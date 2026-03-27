# BBB渗透性预测平台 - Docker部署指南

> **版本**: 1.0
> **更新时间**: 2026-02-24

---

## 目录

1. [为什么使用Docker?](#1-为什么使用docker)
2. [前置要求](#2-前置要求)
3. [快速开始](#3-快速开始)
4. [构建镜像](#4-构建镜像)
5. [运行容器](#5-运行容器)
6. [使用Docker Compose](#6-使用docker-compose)
7. [验证部署](#7-验证部署)
8. [常用命令](#8-常用命令)
9. [故障排除](#9-故障排除)
10. [开发建议](#10-开发建议)

---

## 1. 为什么使用Docker?

| 传统方式 | Docker方式 |
|----------|------------|
| 手动安装RDKit、PyTorch、PyG等复杂依赖 | 一键安装所有依赖 |
| 不同项目环境可能冲突 | 每个项目环境完全隔离 |
| "在我机器上能运行"问题 | 任何机器都能运行 |
| 重新配置环境耗时 | 秒级启动 |
| 难以分享环境 | 镜像分享，环境一致 |

---

## 2. 前置要求

### 2.1 安装Docker

**Windows:**
```powershell
# 使用 winget (推荐)
winget install Docker.DockerDesktop

# 或下载安装包
# https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe
```

**macOS:**
```bash
# 使用 Homebrew
brew install --cask docker

# 或下载安装包
# https://desktop.docker.com/mac/main/amd64/Docker.dmg
```

**Linux (Ubuntu):**
```bash
# 安装Docker
sudo apt-get update
sudo apt-get install -y docker.io

# 启动Docker服务
sudo systemctl start docker
sudo systemctl enable docker

# 将当前用户添加到docker组 (需要重新登录)
sudo usermod -aG docker $USER
```

### 2.2 验证安装

```bash
docker --version
docker-compose --version
```

应该看到类似输出:
```
Docker version 24.0.7
docker-compose version v2.23.0
```

---

## 3. 快速开始

### 3.1 克隆项目

```bash
git clone <repository-url>
cd bbb_project
```

### 3.2 使用Docker Compose启动 (推荐)

```bash
# 首次构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 3.3 访问应用

打开浏览器访问: **http://localhost:8501**

---

## 4. 构建镜像

### 4.1 直接构建

```bash
# 构建镜像 (可能需要10-20分钟)
docker build -t bbb-prediction:latest .

# 查看构建的镜像
docker images | grep bbb
```

### 4.2 使用构建缓存

Docker会缓存构建步骤。如果修改了代码但依赖没变，重新构建会很快。

### 4.3 无缓存构建

```bash
# 强制重新构建
docker build --no-cache -t bbb-prediction:latest .
```

---

## 5. 运行容器

### 5.1 基本运行

```bash
# 运行容器
docker run -d \
  --name bbb-app \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/outputs:/app/outputs \
  bbb-prediction:latest

# 查看运行状态
docker ps

# 查看日志
docker logs -f bbb-app

# 停止容器
docker stop bbb-app

# 删除容器
docker rm bbb-app
```

### 5.2 环境变量

```bash
# 使用自定义环境变量
docker run -d \
  --name bbb-app \
  -p 8501:8501 \
  -e DEFAULT_THRESHOLD=0.65 \
  -e DEFAULT_SEED=0 \
  bbb-prediction:latest
```

### 5.3 资源限制

```bash
# 限制内存和CPU
docker run -d \
  --name bbb-app \
  -p 8501:8501 \
  --memory=4g \
  --cpus=2 \
  bbb-prediction:latest
```

---

## 6. 使用Docker Compose

### 6.1 基本命令

```bash
# 启动服务 (后台运行)
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

# 停止并删除数据卷 (谨慎!)
docker-compose down -v

# 重新构建镜像
docker-compose build --no-cache

# 重启服务
docker-compose restart
```

### 6.2 修改配置后重新构建

```bash
# 修改了Dockerfile或依赖时
docker-compose build
docker-compose up -d
```

### 6.3 查看服务健康状态

```bash
docker-compose ps
```

应该显示:
```
NAME                   IMAGE                    STATUS           PORTS
bbb-prediction-app     bbb-prediction-app       Up (healthy)     0.0.0.0:8501->8501/tcp
```

---

## 7. 验证部署

### 7.1 检查容器运行

```bash
docker ps
```

### 7.2 检查日志

```bash
# 查看最近100行日志
docker logs bbb-prediction-app --tail 100
```

### 7.3 访问Web应用

```bash
# Windows (PowerShell)
Start-Process "http://localhost:8501"

# Linux/macOS
xdg-open http://localhost:8501
# 或
open http://localhost:8501
```

### 7.4 健康检查

```bash
# 使用curl检查
curl -f http://localhost:8501/ || echo "Service not ready"
```

---

## 8. 常用命令

### 8.1 镜像管理

```bash
# 列出镜像
docker images

# 删除未使用的镜像
docker image prune

# 删除指定镜像
docker rmi bbb-prediction:latest
```

### 8.2 容器管理

```bash
# 列出运行中的容器
docker ps

# 列出所有容器
docker ps -a

# 进入容器 (调试用)
docker exec -it bbb-prediction-app /bin/bash

# 查看容器资源使用
docker stats bbb-prediction-app
```

### 8.3 日志管理

```bash
# 查看实时日志
docker-compose logs -f

# 查看最近50行
docker-compose logs --tail=50

# 查看特定服务的日志
docker-compose logs bbb-app
```

### 8.4 数据卷

```bash
# 列出数据卷
docker volume ls

# 查看数据卷详情
docker volume inspect bbb_project_bbb_data
```

---

## 9. 故障排除

### 9.1 端口被占用

```bash
# 查找占用8501端口的进程
# Linux/macOS
lsof -i :8501

# Windows
netstat -ano | findstr :8501

# 更改端口
# 修改 docker-compose.yml 中的 ports:
#   - "8502:8501"
```

### 9.2 内存不足

```bash
# 增加Docker可用内存
# Docker Desktop -> Settings -> Resources

# 或限制容器内存
docker run --memory=2g ...
```

### 9.3 构建失败

```bash
# 清理Docker缓存
docker builder prune

# 重新构建
docker build --no-cache -t bbb-prediction:latest .
```

### 9.4 容器无法启动

```bash
# 查看详细错误
docker logs bbb-prediction-app

# 检查配置文件
docker inspect bbb-prediction-app
```

### 9.5 Windows WSL2问题

```powershell
# 如果使用WSL2后端
wsl --update
wsl --list --verbose

# 重启Docker Desktop
```

### 9.6 macOS问题

```bash
# 给Docker足够资源
# Docker Desktop -> Settings -> Resources

# 如果使用Apple Silicon (M1/M2)
# 确保使用支持arm64的镜像或构建多平台镜像
docker buildx build --platform linux/amd64,linux/arm64 -t bbb-prediction:latest .
```

---

## 10. 开发建议

### 10.1 本地开发工作流

```bash
# 1. 修改代码
# 2. 重新构建 (保持数据卷)
docker-compose build
docker-compose up -d

# 3. 测试
# 4. 重复
```

### 10.2 更新数据/模型

```bash
# 数据已通过数据卷持久化
# 直接替换 data/ 目录下的文件即可

# 模型文件替换 artifacts/ 目录
```

### 10.3 生产部署建议

```bash
# 1. 使用反向代理 (Nginx/Caddy)
# 2. 配置HTTPS
# 3. 设置监控
# 4. 配置日志收集
```

### 10.4 多环境管理

```bash
# 开发环境
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# 生产环境
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## 11. 快速命令参考

```bash
# ===== 一键启动 =====
docker-compose up -d

# ===== 查看状态 =====
docker-compose ps
docker-compose logs -f

# ===== 停止 =====
docker-compose down

# ===== 重新构建 =====
docker-compose build --no-cache
docker-compose up -d

# ===== 进入容器调试 =====
docker exec -it bbb-prediction-app /bin/bash

# ===== 查看资源使用 =====
docker stats

# ===== 清理 =====
docker system prune -a
```

---

## 附录

### A. 目录结构

容器内的目录结构:

```
/app/
├── app_bbb_predict.py    # Streamlit主应用
├── src/                   # 源代码
├── scripts/               # 训练脚本
├── pages/                 # Streamlit页面
├── data/                  # 数据目录 (挂载)
├── artifacts/             # 模型和特征 (挂载)
├── outputs/               # 输出结果 (挂载)
├── requirements.txt       # 依赖列表
├── Dockerfile            # 镜像定义
└── docker-compose.yml    # 编排配置
```

### B. 数据持久化

通过Docker Compose，以下目录会被持久化:

- `bbb_data` → `/app/data`
- `bbb_artifacts` → `/app/artifacts`
- `bbb_outputs` → `/app/outputs`

### C. 常见问题FAQ

**Q: 镜像太大?**
A: 可以使用多阶段构建或使用更小的基础镜像(如python:3.9-slim)

**Q: 构建太慢?**
A: 确保有良好的网络连接，或使用国内镜像源

**Q: 模型文件如何更新?**
A: 替换宿主机 `artifacts/` 目录下的文件即可

---

**文档结束**
