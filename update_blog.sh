#!/bin/bash

rm -rf public

hexo gen

cp -r public/* ~/Desktop/whitelok.github.com/

cd ~/Desktop/whitelok.github.com

git add . --all && git commit -m "update" && git push
