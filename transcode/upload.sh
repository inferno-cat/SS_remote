#!/bin/bash
# 该脚本用于一键将本地项目上传到 GitHub
# 设置本地项目路径，请根据实际情况修改
#project_path="D:\rawcode\NewCode\Sub_Net"
project_path="D:\MOVE\SS_remote"
# 切换到项目路径
cd $project_path
# 获取用户输入的提交信息
echo "请输入提交信息："
read commit_message
# 添加所有文件到暂存区
git add .
# 提交修改
git commit -m "$commit_message"
# 推送代码到 GitHub
git push origin main   # 将分支改为 main
echo "代码上传完成！"