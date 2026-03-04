messages=$1
today=$(date "+%Y-%m-%d %H:%M:%S")
echo "当前日期: $today"
npx prettier . --check
npx prettier --write .
git add .
git commit -m "${today}: ${messages}"
git pull --rebase origin main
git push origin main
