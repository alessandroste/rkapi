# syntax=docker/dockerfile:1@sha256:ecfaec9ed6d810b56388c508f4121597bfbba70d41a6dfeee4d8cad5f295fc32
FROM ghcr.io/astral-sh/uv:0.10.9@sha256:10902f58a1606787602f303954cea099626a4adb02acbac4c69920fe9d278f82 AS uv
FROM ubuntu/python:3.12-24.04@sha256:87c41e84674e08c72be1ddb73a487574689cf953dada87e716031c39f2fa9fb5 AS runtime-base

FROM ubuntu:24.04@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254 AS tooling
ARG TARGETARCH
ARG DEBIAN_FRONTEND=noninteractive
COPY --from=runtime-base /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY <<'SOURCES' /etc/apt/sources.list.d/ubuntu.sources
Types: deb
URIs: https://snapshot.ubuntu.com/ubuntu/20260912T000000Z/
Suites: noble noble-updates noble-security
Components: main universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
Check-Valid-Until: no
SOURCES
RUN test "$TARGETARCH" = arm64 \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/bin/python3.12 UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 PATH="/opt/venv/bin:$PATH"
WORKDIR /app

FROM tooling AS dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --compile-bytecode

FROM tooling AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ python3.12-dev libstdc++6 libgcc-s1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group build --no-install-project
ARG SDK=https://raw.githubusercontent.com/airockchip/rknn-llm/878f9361fd3afa7e167b7079918918f78d2c1c2a
ADD --checksum=sha256:80596a578f7f8e70df6eda1c2cbead3bfced14623a190258f2bd009a3d1f72cf \
    ${SDK}/rkllm-runtime/Linux/librkllm_api/include/rkllm.h /sdk/include/rkllm.h
ADD --checksum=sha256:c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a \
    ${SDK}/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api/include/rknn_api.h /sdk/include/rknn_api.h
ADD --chmod=0644 --checksum=sha256:6a9e4fc5324c68921c3a900340361e107af7599fe34dc8fa7759b2c5ae22a6e6 \
    ${SDK}/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so /out/librkllmrt.so
ADD --chmod=0644 --checksum=sha256:d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8 \
    ${SDK}/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api/aarch64/librknnrt.so /out/librknnrt.so
COPY native/bindings.cpp native/bindings.cpp
RUN g++ -std=c++17 -O2 -Wall -Wextra -Werror -shared -fPIC -fvisibility=hidden -pthread \
    $(python -m pybind11 --includes) -I/sdk/include native/bindings.cpp -ldl \
    -o "/out/_native$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')" \
    && mkdir -p /out/native-libs /out/native-licenses /out/models \
    && cp -L /usr/lib/aarch64-linux-gnu/libstdc++.so.6 \
        /usr/lib/aarch64-linux-gnu/libgcc_s.so.1 \
        /usr/lib/aarch64-linux-gnu/libgomp.so.1 /out/native-libs/ \
    && for pkg in libstdc++6 libgcc-s1 libgomp1; do \
        cp -L "/usr/share/doc/$pkg/copyright" "/out/native-licenses/$pkg"; \
    done

FROM runtime-base AS runtime
ENV PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOME=/tmp XDG_CACHE_HOME=/tmp/cache \
    RKLLM_LIB_PATH=/runtime/librkllmrt.so RKNN_LIB_PATH=/runtime/librknnrt.so \
    MODEL_PATH=/models/model.rkllm HOST=0.0.0.0 PORT=8001
WORKDIR /app
COPY --from=build /out/native-libs/ /usr/lib/aarch64-linux-gnu/
COPY --from=build /out/native-licenses/ /usr/share/licenses/native-deps/
COPY --from=build /out/models/ /models/
COPY --from=dependencies /opt/venv /opt/venv
COPY --from=build /out/librkllmrt.so /out/librknnrt.so /runtime/
COPY licenses/ /usr/share/licenses/rkllm-api/
COPY app/ app/
COPY --from=build /out/_native*.so app/
COPY main.py LICENSE README.md ./
USER 10001:10001
RUN ["/opt/venv/bin/python", "-c", "import ctypes, hashlib; from pathlib import Path; pins={'qwen35.jinja':'273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80','gemma4.jinja':'0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5'}; assert all(hashlib.sha256((Path('app/templates')/name).read_bytes()).hexdigest()==digest for name,digest in pins.items()), 'Chat template checksum mismatch'; ctypes.CDLL('/runtime/librkllmrt.so'); ctypes.CDLL('/runtime/librknnrt.so'); import app._native; import main"]
EXPOSE 8001
STOPSIGNAL SIGTERM
ENTRYPOINT []
CMD ["/opt/venv/bin/python", "/app/main.py"]
