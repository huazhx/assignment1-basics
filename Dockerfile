FROM python:3.11-slim

# 不缓冲输出（日志即时可见）
ENV PYTHONUNBUFFERED=1

# 安装系统依赖：基础编译工具 + 常见 Python 包依赖（如 cryptography, numpy 等可能需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    xz-utils \
    zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 使用官方推荐方式安装 uv（通过脚本安装预编译二进制）
# 参考：https://docs.astral.sh/uv/getting-started/installation/
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# 验证 uv 是否安装成功（可选）
RUN uv --version

# 如果存在 pyproject.toml 或 requirements.txt，则提前安装依赖
RUN if [ -f pyproject.toml ]; then \
        uv pip install -r <(uv pip compile pyproject.toml); \
    elif [ -f requirements.txt ]; then \
        uv pip install -r requirements.txt; \
    else \
        echo "⚠️ No dependency file found. Skipping Python deps install."; \
    fi

# 复制项目文件
COPY . .

# 默认启动 shell（可按需改为 uv run ...）
CMD ["/bin/bash"]