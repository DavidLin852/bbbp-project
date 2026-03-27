# ============================================================
# BBB Permeability Prediction Platform - Docker Image
# ============================================================
#
# Build: docker build -t bbb-prediction .
# Run:   docker run -p 8501:8501 bbb-prediction
# ============================================================

# 使用官方Python镜像 (3.9版本，与项目兼容性好)
FROM python:3.9-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /app

# ============================================================
# 安装系统依赖
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # RDKit需要的系统库
    libgl1-mesa-glx \
    libglib2.0-0 \
    libboost-all-dev \
    # Git (用于可能的代码获取)
    git \
    # Wget (用于下载文件)
    wget \
    # 构建工具
    build-essential \
    # 清理 apt 缓存
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 安装Miniconda (推荐方式安装RDKit)
# ============================================================
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# 添加conda到PATH
ENV PATH=/opt/conda/bin:$PATH

# ============================================================
# 创建conda环境并安装RDKit
# ============================================================
RUN conda create -n bbb python=3.9 -y && \
    conda install -n bbb -c conda-forge rdkit -y

# 激活conda环境并安装其他依赖
RUN conda run -n bbb pip install --no-cache-dir \
    streamlit==1.29.0 \
    pandas==2.1.4 \
    numpy==1.26.3 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    xgboost==2.0.3 \
    lightgbm==4.1.0 \
    matplotlib==3.8.2 \
    seaborn==0.13.1 \
    plotly==5.18.0 \
    joblib==1.3.2 \
    tqdm==4.66.1

# 安装PyTorch (CPU版本，减小镜像体积)
RUN conda run -n bbb pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2

# 安装PyTorch Geometric及其依赖
RUN conda run -n bbb pip install --no-cache-dir \
    torch-geometric==2.4.0 \
    torch-scatter==2.1.1 \
    torch-sparse==0.6.17 \
    torch-cluster==1.6.1 \
    torch-spline-conv==1.2.2

# ============================================================
# 复制项目文件
# ============================================================
# 先复制requirements.txt (如果存在)
COPY requirements.txt* /app/

# 复制整个项目
COPY . /app/

# ============================================================
# 创建必要的目录
# ============================================================
RUN mkdir -p /app/data/splits \
             /app/data/raw \
             /app/artifacts/models \
             /app/artifacts/features \
             /app/artifacts/temp_predict \
             /app/outputs \
             /app/outputs/images \
             /app/outputs/docs

# ============================================================
# 设置Streamlit配置
# ============================================================
RUN mkdir -p /root/.streamlit

# Streamlit配置文件
RUN echo "[server]" > /root/.streamlit/config.toml && \
    echo "headless = true" >> /root/.streamlit/config.toml && \
    echo "enableCORS = false" >> /root/.streamlit/config.toml && \
    echo "enableXsrfProtection = true" >> /root/.streamlit/config.toml && \
    echo "port = 8501" >> /root/.streamlit/config.toml

# ============================================================
# 暴露端口
# ============================================================
EXPOSE 8501

# ============================================================
# 启动命令
# ============================================================
# 使用conda环境中的Python运行streamlit
CMD ["conda", "run", "-n", "bbb", "streamlit", "run", "app_bbb_predict.py", "--server.address", "0.0.0.0"]
