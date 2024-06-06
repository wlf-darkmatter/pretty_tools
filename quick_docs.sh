shell_pwd=$(dirname "$(readlink -f "$0")")
echo 当前目录为:"$shell_pwd"
cd "${shell_pwd}"

#* 生成之前清除掉部分缓存
rm -rf build/html

#* 针对pretty_tools生成文档
cd "${shell_pwd}"
rm -rf ./docs/source/autoapi
sphinx-apidoc -o ./docs/source/autoapi -d 8 ./pretty_tools

cd docs
make html

# 同步到lux服务器上
cd "${shell_pwd}"

passwd=$(cat ./tmp/passwd_lux.txt)

sshpass -p $passwd rsync --delete -avzP ./docs/build/html/* \
    wlf@lux4.x-contion.top:/nas/server/documents_storage/pretty_tools




