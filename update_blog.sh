#!/bin/bash

rm -rf public

hexo gen

cp -r public/* ~/Desktop/whitelok.github.com/
