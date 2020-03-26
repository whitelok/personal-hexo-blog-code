---
title: 如何修复Xavier Destop 开启Desktop Sharing失败问题
date: 2019-08-05 18:59:00
tags:
---

 - OS version：R32.1 Ubuntu 18.04
 - 问题描述：在开启Desktop Sharing的时候（[如何开启Ubuntu远程桌面](https://www.jianshu.com/p/817c30f934d8)）

```bash
sudo apt update
sudo apt install vino
gsettings set org.gnome.Vino prompt-enabled false
gsettings set org.gnome.Vino require-encryption false
nmcli connection show
dconf write /org/gnome/settings-daemon/plugins/sharing/vino-server/enabled-connections "['<UUID of the ethernet>']"
export DISPLAY=:0
/usr/lib/vino/vino-server
```
