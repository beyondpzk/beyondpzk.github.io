#!/bin/bash
# 一键启动 VitePress 本地预览

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "🚀 启动 VitePress 预览服务器..."
echo "📍 项目目录: $PROJECT_DIR"

# 自动同步侧边栏
echo "🔄 同步侧边栏..."
npm run sync

# 启动开发服务器
echo "✨ 打开 http://localhost:5173/my-blog/"
npm run docs:dev
