FROM python:3.11-slim

# 不缓冲输出（日志即时可见）
ENV PYTHONUNBUFFERED=1

# 安装基础工具（可按需删减或增加）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 预装 pip 并安装 uv（仓库 README 推荐使用 uv 管理环境）
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir uv

# 复制源代码
COPY . .

CMD ["/bin/bash"]