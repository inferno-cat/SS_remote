#!/bin/bash
# 一键拉取 GitHub 项目更新到本地

project_dir="/home/share/zc/code/SS_remote"

cd $project_dir || { echo "目录不存在：$project_dir"; exit 1; }

# 放弃本地未提交的更改
git reset --hard HEAD

# 拉取远程 main 分支
git pull origin main

echo "项目更新完成！"
