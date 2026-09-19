#!/bin/bash
set -euo pipefail
# Commit explicitly reviewed files first; do not stage the entire working tree.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "请先检查并提交需要发布的文件；工作区仍有未提交修改。" >&2
  exit 1
fi
npm run check
npm run docs:build
npm run check:output
git push origin main
