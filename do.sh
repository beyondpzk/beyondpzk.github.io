# ===========================================
# --coding:UTF-8 --
# file: do.sh
# description: 
# ===========================================

git pull
git add .
git commit -m $(date +%Y-%m-%d-%H-%M-%S)
git push origin main
