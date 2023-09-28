# 个人博客hexo代码

## Mac部署

```bash
brew link node
brew uninstall node
brew install node
npm install -g hexo-cli
git clone https://github.com/whitelok/personal-hexo-blog-code.git
git clone https://github.com/whitelok/hexo-theme-cactus-plus.git personal-hexo-blog-code/themes/hexo-theme-cactus-plus
git clone https://github.com/whitelok/whitelok.github.com.git
hexo init blog
cp -r personal-hexo-blog-code/* blog/
```
