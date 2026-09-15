#include <cstddef>
#include "rkllm.h"
#include "rknn_api.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

void check(int status, const char* operation) {
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
    }
}

void text_argument(const std::string& value, const char* name) {
    if (value.empty() || value.find('\0') != std::string::npos) {
        throw std::invalid_argument(std::string(name) + " must be non-empty and contain no NUL");
    }
}

class Library {
    void* handle_ = nullptr;
public:
    explicit Library(const std::string& path) {
        text_argument(path, "Library path");
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle_) {
            throw std::runtime_error(dlerror());
        }
    }
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
    ~Library() {
        if (handle_ && dlclose(handle_) != 0) {
            std::fprintf(stderr, "Native library cleanup failed: %s\n", dlerror());
        }
    }
    template<typename Function>
    Function get(const char* name) {
        dlerror();
        void* address = dlsym(handle_, name);
        if (const char* error = dlerror()) {
            throw std::runtime_error(error);
        }
        return reinterpret_cast<Function>(address);
    }
};

struct Output {
    std::mutex mutex;
    std::condition_variable space;
    std::deque<std::string> chunks;
    std::size_t bytes = 0;
    std::size_t limit = 0;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> failed{false};
    const char* error = "RKLLM callback failed";
    bool finished = false;
    bool has_perf = false;
    bool saw_eos = false;
    std::vector<int32_t> eos_token_ids;
    int generated_tokens = 0;
    RKLLMPerfStat perf{};
};

int result_callback(RKLLMResult* result, void* userdata, LLMCallState state) noexcept {
    auto& output = *static_cast<Output*>(userdata);
    try {
        std::unique_lock<std::mutex> lock(output.mutex);
        if (state == RKLLM_RUN_ERROR) {
            output.failed.store(true);
        } else if (state == RKLLM_RUN_FINISH) {
            output.finished = true;
            if (result) {
                output.perf = result->perf;
                output.has_perf = true;
                output.saw_eos = output.saw_eos || std::find(
                    output.eos_token_ids.begin(), output.eos_token_ids.end(), result->token_id
                ) != output.eos_token_ids.end();
            }
        } else if (state == RKLLM_RUN_NORMAL && result) {
            ++output.generated_tokens;
            output.saw_eos = output.saw_eos || std::find(
                output.eos_token_ids.begin(), output.eos_token_ids.end(), result->token_id
            ) != output.eos_token_ids.end();
            const std::string text = result->text ? result->text : "";
            if (text.size() > output.limit) {
                output.error = "Native callback exceeded the output queue capacity";
                output.failed.store(true);
            } else if (!text.empty()) {
                output.space.wait(lock, [&] {
                    return output.cancelled.load() || output.failed.load()
                        || output.bytes + text.size() <= output.limit;
                });
                if (!output.cancelled.load() && !output.failed.load()) {
                    output.chunks.push_back(text);
                    output.bytes += text.size();
                }
            }
        }
    } catch (...) {
        // No exception may cross the vendor's C callback boundary.
        output.error = "Could not retain native callback output";
        output.failed.store(true);
    }
    return 0;
}

class Generation;

class LLM : public std::enable_shared_from_this<LLM> {
    Library library_;
    std::string model_path_;
    RKLLMParam params_{};
    RKLLMCallback callbacks_{};
    Output initialization_output_;
    LLMHandle handle_ = nullptr;
    std::mutex mutex_;
    std::condition_variable idle_;
    std::weak_ptr<Generation> active_;
    bool closing_ = false;
    std::size_t queue_bytes_;
    std::vector<int32_t> eos_token_ids_;
    int image_embedding_size_;
    decltype(&rkllm_destroy) destroy_;
    decltype(&rkllm_clear_kv_cache) clear_;
    decltype(&rkllm_run) run_;
    decltype(&rkllm_abort) abort_;
    decltype(&rkllm_is_running) running_;
public:
    LLM(const std::string& library_path, const std::string& model_path,
        int context_len, int max_new_tokens, bool ignore_eos, std::size_t queue_bytes,
        std::vector<int32_t> eos_token_ids, bool skip_special_tokens, int image_embedding_size)
        : library_(library_path), model_path_(model_path), queue_bytes_(queue_bytes),
          eos_token_ids_(std::move(eos_token_ids)),
          image_embedding_size_(image_embedding_size),
          destroy_(library_.get<decltype(destroy_)>("rkllm_destroy")),
          clear_(library_.get<decltype(clear_)>("rkllm_clear_kv_cache")),
          run_(library_.get<decltype(run_)>("rkllm_run")),
          abort_(library_.get<decltype(abort_)>("rkllm_abort")),
          running_(library_.get<decltype(running_)>("rkllm_is_running")) {
        text_argument(model_path_, "Model path");
        if (context_len < 1 || context_len > 4096 || max_new_tokens < 1 ||
            max_new_tokens > context_len || queue_bytes == 0 || queue_bytes > 1024 * 1024) {
            throw std::invalid_argument("Invalid context, token limit or native queue capacity");
        }
        if (eos_token_ids_.size() > 32 || std::any_of(
            eos_token_ids_.begin(), eos_token_ids_.end(), [](int32_t token) { return token < 0; }
        )) {
            throw std::invalid_argument("EOS token IDs must be non-negative (at most 32)");
        }
        if (image_embedding_size_ < 0 || image_embedding_size_ > 8192) {
            throw std::invalid_argument("Invalid image embedding width");
        }
        const auto defaults = library_.get<decltype(&rkllm_createDefaultParam)>("rkllm_createDefaultParam");
        const auto init = library_.get<decltype(&rkllm_init)>("rkllm_init");
        const auto set_template = library_.get<decltype(&rkllm_set_chat_template)>("rkllm_set_chat_template");
        params_ = defaults();
        params_.model_path = model_path_.c_str();
        params_.max_context_len = context_len;
        params_.max_new_tokens = max_new_tokens;
        params_.top_k = 20;
        params_.top_p = 0.95F;
        params_.temperature = 0.6F;
        params_.repeat_penalty = 1.1F;
        params_.frequency_penalty = 0;
        params_.presence_penalty = 0;
        params_.skip_special_token = skip_special_tokens;
        params_.ignore_eos_token = ignore_eos;
        params_.is_async = false;
        params_.extend_param.n_batch = 1;
        params_.extend_param.base_domain_id = 1;
        callbacks_.result_callback = result_callback;
        callbacks_.result_userdata = &initialization_output_;
        try {
            check(init(&handle_, &params_, &callbacks_), "rkllm_init");
            if (!handle_ || initialization_output_.failed.load()) {
                throw std::runtime_error("RKLLM initialization returned an invalid handle or callback error");
            }
            check(set_template(handle_, "", "", ""), "rkllm_set_chat_template");
        } catch (...) {
            if (handle_ && destroy_(handle_) != 0) {
                std::fputs("RKLLM initialization cleanup also failed\n", stderr);
            }
            handle_ = nullptr;
            throw;
        }
    }
    ~LLM() {
        try {
            close();
        } catch (const std::exception& error) {
            std::fprintf(stderr, "RKLLM destructor: %s\n", error.what());
        }
    }
    int context_len() const { return params_.max_context_len; }
    std::size_t queue_bytes() const { return queue_bytes_; }
    std::shared_ptr<Generation> request(
        const std::string& prompt, int max_tokens, float temperature, float top_p,
        int top_k, float repeat_penalty, float frequency_penalty, float presence_penalty,
        const py::object& image, int width, int height);
    LLMHandle begin(const std::shared_ptr<Generation>& generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closing_ || !handle_) {
            throw std::runtime_error("LLM is closed");
        }
        if (!active_.expired()) {
            throw std::runtime_error("LLM already has an active generation");
        }
        active_ = generation;
        return handle_;
    }
    void end() {
        std::lock_guard<std::mutex> lock(mutex_);
        active_.reset();
        idle_.notify_all();
    }
    void close();
    friend class Generation;
};

class Generation : public std::enable_shared_from_this<Generation> {
    std::shared_ptr<LLM> model_;
    std::string prompt_;
    std::vector<float> image_;
    std::size_t image_tokens_ = 0;
    int width_ = 0;
    int height_ = 0;
    RKLLMSamplingParam sampling_{};
    int max_tokens_;
    Output output_;
    std::atomic<bool> started_{false};

    void run_native() {
        if (started_.exchange(true)) {
            throw std::runtime_error("Generation.run is single-shot");
        }
        if (output_.cancelled.load()) {
            return;
        }
        const auto handle = model_->begin(shared_from_this());
        try {
            check(model_->clear_(handle, 0, nullptr, nullptr), "rkllm_clear_kv_cache");
            if (!output_.cancelled.load()) {
                RKLLMInput input{};
                input.role = "user";
                input.enable_thinking = false;
                input.input_type = image_.empty() ? RKLLM_INPUT_PROMPT : RKLLM_INPUT_MULTIMODAL;
                if (image_.empty()) {
                    input.prompt_input = prompt_.c_str();
                } else {
                    input.multimodal_input.prompt = prompt_.data();
                    auto& image = input.multimodal_input.image;
                    image.image_embed = image_.data();
                    image.n_image_tokens = image_tokens_;
                    image.n_image = 1;
                    image.image_start = "<|vision_start|>";
                    image.image_end = "<|vision_end|>";
                    image.image_content = "<|image_pad|>";
                    image.image_width = width_;
                    image.image_height = height_;
                }
                RKLLMInferParam inference{};
                inference.mode = RKLLM_INFER_GENERATE;
                inference.keep_history = 0;
                inference.max_new_tokens = max_tokens_;
                inference.sampling_params = &sampling_;
                std::atomic<bool> complete{false};
                std::exception_ptr worker_error;
                int status = -1;
                std::thread worker([&] {
                    try {
                        status = model_->run_(handle, &input, &inference, &output_);
                    } catch (...) {
                        worker_error = std::current_exception();
                    }
                    complete.store(true);
                });
                int abort_status = 0;
                bool aborted = false;
                try {
                    while (!complete.load()) {
                        if ((output_.cancelled.load() || output_.failed.load()) &&
                            !aborted && model_->running_(handle) == 1) {
                            output_.cancelled.store(true);
                            output_.space.notify_all();
                            abort_status = model_->abort_(handle);
                            aborted = true;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    }
                } catch (...) {
                    cancel();
                    if (model_->abort_(handle) != 0) {
                        std::fputs("RKLLM emergency abort failed\n", stderr);
                    }
                    worker.join();
                    throw;
                }
                worker.join();
                if (worker_error) {
                    std::rethrow_exception(worker_error);
                }
                check(abort_status, "rkllm_abort");
                check(status, "rkllm_run");
                if (output_.failed.load()) {
                    throw std::runtime_error(output_.error);
                }
                if (!output_.cancelled.load() && (!output_.finished || !output_.has_perf)) {
                    throw std::runtime_error("RKLLM did not deliver a finish callback with performance data");
                }
            }
        } catch (...) {
            model_->end();
            throw;
        }
        model_->end();
    }
public:
    Generation(std::shared_ptr<LLM> model, std::string prompt, int max_tokens,
               RKLLMSamplingParam sampling, std::vector<float> image,
               std::size_t image_tokens, int width, int height)
        : model_(std::move(model)), prompt_(std::move(prompt)), image_(std::move(image)),
          image_tokens_(image_tokens), width_(width), height_(height),
          sampling_(sampling), max_tokens_(max_tokens) {
        output_.limit = model_->queue_bytes();
        output_.eos_token_ids = model_->eos_token_ids_;
    }
    void cancel() {
        output_.cancelled.store(true);
        output_.space.notify_all();
    }
    py::list read() {
        py::list chunks;
        std::lock_guard<std::mutex> lock(output_.mutex);
        for (const auto& text : output_.chunks) {
            chunks.append(py::bytes(text));
        }
        output_.chunks.clear();
        output_.bytes = 0;
        output_.space.notify_all();
        return chunks;
    }
    py::dict run() {
        const auto keep_alive = shared_from_this();
        {
            py::gil_scoped_release release;
            run_native();
        }
        std::lock_guard<std::mutex> lock(output_.mutex);
        py::dict result;
        result["prompt_tokens"] = output_.has_perf ? output_.perf.prefill_tokens : -1;
        result["completion_tokens"] = output_.has_perf ? output_.perf.generate_tokens : -1;
        result["prefill_ms"] = output_.perf.prefill_time_ms;
        result["generate_ms"] = output_.perf.generate_time_ms;
        result["memory_mb"] = output_.perf.memory_usage_mb;
        result["generated_tokens"] = output_.generated_tokens;
        result["saw_eos"] = output_.saw_eos;
        result["cancelled"] = output_.cancelled.load();
        return result;
    }
};

void LLM::close() {
    std::unique_lock<std::mutex> lock(mutex_);
    closing_ = true;
    if (auto generation = active_.lock()) {
        generation->cancel();
    }
    idle_.wait(lock, [&] { return active_.expired(); });
    if (handle_) {
        const auto handle = std::exchange(handle_, nullptr);
        check(destroy_(handle), "rkllm_destroy");
    }
}

std::shared_ptr<Generation> LLM::request(
    const std::string& prompt, int max_tokens, float temperature, float top_p, int top_k,
    float repeat_penalty, float frequency_penalty, float presence_penalty,
    const py::object& image, int width, int height) {
    text_argument(prompt, "Prompt");
    if (prompt.size() > 8 * 1024 * 1024 || max_tokens < 1 || max_tokens > context_len() ||
        top_k < 1 ||
        !std::isfinite(temperature) || temperature < 0 || temperature > 2 ||
        !std::isfinite(top_p) || top_p <= 0 || top_p > 1 ||
        !std::isfinite(repeat_penalty) || repeat_penalty <= 0 || repeat_penalty > 2 ||
        !std::isfinite(frequency_penalty) || std::abs(frequency_penalty) > 2 ||
        !std::isfinite(presence_penalty) || std::abs(presence_penalty) > 2) {
        throw std::invalid_argument("Invalid request sampling parameters or prompt size");
    }
    std::vector<float> pixels;
    std::size_t image_tokens = 0;
    if (!image.is_none()) {
        if (image_embedding_size_ == 0) {
            throw std::invalid_argument("This model profile does not support image input");
        }
        if (!py::isinstance<py::array>(image)) {
            throw std::invalid_argument("Image embeddings must be a float32 numpy array");
        }
        auto array = py::cast<py::array>(image);
        if (!array.dtype().is(py::dtype::of<float>()) ||
            !(array.flags() & py::array::c_style) || array.ndim() != 2 ||
            array.shape(0) < 1 || array.shape(0) > 4096 ||
            array.shape(1) != image_embedding_size_ ||
            width <= 0 || height <= 0 || width > 2048 || height > 2048 ||
            width % 32 != 0 || height % 32 != 0 ||
            array.shape(0) != (width / 32) * (height / 32) ||
            prompt.find("<image>") == std::string::npos ||
            prompt.find("<image>", prompt.find("<image>") + 1) != std::string::npos) {
            throw std::invalid_argument("Image embeddings, dimensions or placeholder do not match the model profile");
        }
        const auto* data = static_cast<const float*>(array.data());
        pixels.assign(data, data + array.size());
        for (const auto value : pixels) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Image embeddings contain non-finite values");
            }
        }
        image_tokens = static_cast<std::size_t>(array.shape(0));
    } else if (width != 0 || height != 0) {
        throw std::invalid_argument("Image dimensions require image embeddings");
    }
    RKLLMSamplingParam sampling{
        top_k, top_p, temperature, repeat_penalty, frequency_penalty, presence_penalty,
        0, params_.mirostat_tau, params_.mirostat_eta,
    };
    return std::make_shared<Generation>(
        shared_from_this(), prompt, max_tokens, sampling, std::move(pixels),
        image_tokens, width, height);
}

class Vision {
    Library library_;
    std::string model_path_;
    rknn_context context_ = 0;
    std::mutex mutex_;
    int width_ = 0;
    int height_ = 0;
    std::size_t tokens_ = 0;
    std::size_t embedding_ = 0;
    int expected_embedding_size_;
    std::vector<rknn_tensor_attr> output_attrs_;
    decltype(&rknn_destroy) destroy_;
    decltype(&rknn_inputs_set) inputs_;
    decltype(&rknn_run) run_;
    decltype(&rknn_outputs_get) outputs_;
    decltype(&rknn_outputs_release) release_;
public:
    Vision(const std::string& library_path, const std::string& model_path, int embedding_size)
        : library_(library_path), model_path_(model_path),
          expected_embedding_size_(embedding_size),
          destroy_(library_.get<decltype(destroy_)>("rknn_destroy")),
          inputs_(library_.get<decltype(inputs_)>("rknn_inputs_set")),
          run_(library_.get<decltype(run_)>("rknn_run")),
          outputs_(library_.get<decltype(outputs_)>("rknn_outputs_get")),
          release_(library_.get<decltype(release_)>("rknn_outputs_release")) {
        text_argument(model_path_, "Vision model path");
        if (expected_embedding_size_ < 1 || expected_embedding_size_ > 8192) {
            throw std::invalid_argument("Invalid vision embedding width");
        }
        const auto init = library_.get<decltype(&rknn_init)>("rknn_init");
        const auto query = library_.get<decltype(&rknn_query)>("rknn_query");
        const auto cores = library_.get<decltype(&rknn_set_core_mask)>("rknn_set_core_mask");
        try {
            check(init(&context_, model_path_.data(), 0, 0, nullptr), "rknn_init");
            if (!context_) {
                throw std::runtime_error("RKNN returned an invalid context");
            }
            check(cores(context_, RKNN_NPU_CORE_0_1_2), "rknn_set_core_mask");
            rknn_input_output_num counts{};
            check(query(context_, RKNN_QUERY_IN_OUT_NUM, &counts, sizeof(counts)), "rknn_query IO counts");
            if (counts.n_input != 1 || counts.n_output < 1 || counts.n_output > 4) {
                throw std::runtime_error("Unsupported vision input/output count");
            }
            rknn_tensor_attr input{};
            check(query(context_, RKNN_QUERY_INPUT_ATTR, &input, sizeof(input)), "rknn_query input");
            if (input.n_dims != 4 || input.dims[0] != 1 ||
                (input.fmt != RKNN_TENSOR_NCHW && input.fmt != RKNN_TENSOR_NHWC)) {
                throw std::runtime_error("Vision input must be a static batch-one image tensor");
            }
            const bool nchw = input.fmt == RKNN_TENSOR_NCHW;
            const auto channels = input.dims[nchw ? 1 : 3];
            height_ = static_cast<int>(input.dims[nchw ? 2 : 1]);
            width_ = static_cast<int>(input.dims[nchw ? 3 : 2]);
            if (channels != 3 || width_ <= 0 || height_ <= 0 || width_ > 2048 || height_ > 2048 ||
                width_ % 32 != 0 || height_ % 32 != 0) {
                throw std::runtime_error("Unsupported Qwen3.5 vision image dimensions");
            }
            output_attrs_.resize(counts.n_output);
            for (uint32_t index = 0; index < counts.n_output; ++index) {
                auto& attr = output_attrs_[index];
                attr.index = index;
                check(query(context_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)), "rknn_query output");
                if (attr.n_dims < 2 || attr.n_dims > RKNN_MAX_DIMS) {
                    throw std::runtime_error("Vision output is not an embedding matrix");
                }
                for (uint32_t dimension = 0; dimension + 2 < attr.n_dims; ++dimension) {
                    if (attr.dims[dimension] != 1) {
                        throw std::runtime_error("Only batch-one vision embeddings are supported");
                    }
                }
                const auto tokens = attr.dims[attr.n_dims - 2];
                const auto embedding = attr.dims[attr.n_dims - 1];
                if (tokens != static_cast<uint32_t>((width_ / 32) * (height_ / 32)) ||
                    embedding == 0 || embedding > static_cast<uint32_t>(expected_embedding_size_) ||
                    attr.n_elems != static_cast<std::size_t>(tokens) * embedding) {
                    throw std::runtime_error("Vision output shape does not match the image grid");
                }
                if (index && (tokens != tokens_ || embedding != embedding_)) {
                    throw std::runtime_error("Vision deepstack outputs must have matching shapes");
                }
                tokens_ = tokens;
                embedding_ = embedding;
            }
            if (embedding_ * counts.n_output != static_cast<std::size_t>(expected_embedding_size_)) {
                throw std::runtime_error("Vision embedding width does not match the selected model profile");
            }
        } catch (...) {
            if (context_ && destroy_(context_) != 0) {
                std::fputs("RKNN initialization cleanup also failed\n", stderr);
            }
            context_ = 0;
            throw;
        }
    }
    ~Vision() {
        try {
            close();
        } catch (const std::exception& error) {
            std::fprintf(stderr, "RKNN destructor: %s\n", error.what());
        }
    }
    int width() const { return width_; }
    int height() const { return height_; }
    std::size_t image_tokens() const { return tokens_; }
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (context_) {
            const auto context = std::exchange(context_, 0);
            check(destroy_(context), "rknn_destroy");
        }
    }
    py::array_t<float> encode(const py::array& array) {
        if (!array.dtype().is(py::dtype::of<uint8_t>()) || !(array.flags() & py::array::c_style) ||
            array.ndim() != 3 || array.shape(0) != height_ ||
            array.shape(1) != width_ || array.shape(2) != 3) {
            throw std::invalid_argument("Vision input must be contiguous uint8 HWC RGB at the queried size");
        }
        const auto* data = static_cast<const uint8_t*>(array.data());
        std::vector<uint8_t> pixels(data, data + array.size());
        py::array_t<float> result({
            static_cast<py::ssize_t>(tokens_), static_cast<py::ssize_t>(expected_embedding_size_)
        });
        float* destination = result.mutable_data();
        {
            py::gil_scoped_release release_gil;
            std::lock_guard<std::mutex> lock(mutex_);
            if (!context_) {
                throw std::runtime_error("Vision encoder is closed");
            }
            rknn_input input{};
            input.buf = pixels.data();
            input.size = static_cast<uint32_t>(pixels.size());
            input.type = RKNN_TENSOR_UINT8;
            input.fmt = RKNN_TENSOR_NHWC;
            input.pass_through = 0;
            check(inputs_(context_, 1, &input), "rknn_inputs_set");
            check(run_(context_, nullptr), "rknn_run");
            std::vector<rknn_output> outputs(output_attrs_.size());
            for (uint32_t index = 0; index < outputs.size(); ++index) {
                outputs[index].index = index;
                outputs[index].want_float = 1;
            }
            check(outputs_(context_, static_cast<uint32_t>(outputs.size()), outputs.data(), nullptr),
                  "rknn_outputs_get");
            try {
                for (std::size_t index = 0; index < outputs.size(); ++index) {
                    if (!outputs[index].buf || outputs[index].size != tokens_ * embedding_ * sizeof(float)) {
                        throw std::runtime_error("RKNN returned an invalid embedding buffer size");
                    }
                    const auto* source = static_cast<const float*>(outputs[index].buf);
                    for (std::size_t token = 0; token < tokens_; ++token) {
                        std::memcpy(destination + token * expected_embedding_size_ + index * embedding_,
                                    source + token * embedding_, embedding_ * sizeof(float));
                    }
                }
            } catch (...) {
                if (release_(context_, static_cast<uint32_t>(outputs.size()), outputs.data()) != 0) {
                    std::fputs("RKNN output cleanup also failed\n", stderr);
                }
                throw;
            }
            check(release_(context_, static_cast<uint32_t>(outputs.size()), outputs.data()),
                  "rknn_outputs_release");
        }
        return result;
    }
};

}  // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "Owned RKLLM 1.3 / RKNN handles; no Python calls from vendor callbacks.";
    py::class_<Generation, std::shared_ptr<Generation>>(module, "Generation")
        .def("run", &Generation::run)
        .def("read", &Generation::read)
        .def("cancel", &Generation::cancel);
    py::class_<LLM, std::shared_ptr<LLM>>(module, "LLM")
        .def(py::init([](const std::string& library, const std::string& model,
                         int context, int tokens, bool ignore_eos, std::size_t queue_bytes,
                         std::vector<int32_t> eos_token_ids, bool skip_special_tokens,
                         int image_embedding_size) {
            py::gil_scoped_release release;
            return std::make_shared<LLM>(
                library, model, context, tokens, ignore_eos, queue_bytes,
                std::move(eos_token_ids), skip_special_tokens, image_embedding_size);
        }), py::arg("library_path"), py::arg("model_path"), py::arg("context_len") = 4096,
            py::arg("max_new_tokens") = 256, py::arg("ignore_eos") = false,
            py::arg("queue_bytes") = 65536,
            py::arg("eos_token_ids") = std::vector<int32_t>{},
            py::arg("skip_special_tokens") = false,
            py::arg("image_embedding_size") = 2048)
        .def("request", &LLM::request, py::arg("prompt"), py::arg("max_tokens") = 256,
             py::arg("temperature") = 0.6F, py::arg("top_p") = 0.95F, py::arg("top_k") = 20,
             py::arg("repeat_penalty") = 1.1F, py::arg("frequency_penalty") = 0.0F,
             py::arg("presence_penalty") = 0.0F, py::arg("image") = py::none(),
             py::arg("width") = 0, py::arg("height") = 0)
        .def("close", &LLM::close, py::call_guard<py::gil_scoped_release>());
    py::class_<Vision>(module, "Vision")
        .def(py::init([](const std::string& library, const std::string& model, int embedding_size) {
            py::gil_scoped_release release;
            return std::make_unique<Vision>(library, model, embedding_size);
        }), py::arg("library_path"), py::arg("model_path"), py::arg("embedding_size") = 2048)
        .def_property_readonly("width", &Vision::width)
        .def_property_readonly("height", &Vision::height)
        .def_property_readonly("image_tokens", &Vision::image_tokens)
        .def("encode", &Vision::encode)
        .def("close", &Vision::close, py::call_guard<py::gil_scoped_release>());
}
