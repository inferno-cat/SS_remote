#!/bin/bash
# 该脚本用于一键获取 GitHub 项目更新到服务器本地
# 设置项目目录路径，请根据实际情况修改
project_dir="/home/share/zc/code/SS"
# 切换到项目目录
cd $project_dir
# 拉取最新更新
git pull origin main
echo "项目更新完成！"