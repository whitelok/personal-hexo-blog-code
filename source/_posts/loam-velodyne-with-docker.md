---
title: 手把手教你用Docker看自动驾驶的激光点云：loam_velodyne激光点云可视化工具
date: 2019-08-30 11:33:46
tags:
---

# 目标
 - 在现有的Ubuntu系统上部署ROS耗时长而且冲突很多
 - 而且由于将ROS安装到Ubuntu 16.04上会遇到ROS依赖包与catkin_make冲突的问题，将ROS部署到Docker的container中，外部机器不受影响，避免包冲突
 - container运行结果直接输出到host主机上
 - 效果图如下
 ![loam-velodyne-with-docker-image-1](http://whitelok.github.io/resources/loam-velodyne-with-docker-image-1.gif)
 - loam_velodyne的[项目地址](https://github.com/laboshinl/loam_velodyne)

# 安装&部署&运行
 1. 先在host上安装Docker， 并运行：`xhost +`
 2. 拉取melodic版本的image：`sudo docker pull osrf/ros:melodic-desktop-full`
 3. 下载完后运行起container：`sudo docker run -itd -v [需要共享的路径]:[需要共享的路径] -e DISPLAY=unix$display -e LC_ALL=C.UTF-8 -e --ipc=host --privileged --net host --shm-size 4G --device /dev/dri --cap-add sys_ptrace osrf/ros:melodic-desktop-full`
 4. 然后进入到Docker：`sudo docker exec -it [container id] bash`
 5. 新建工程编译目录：`mkdir -p ~/catkin_ws/src`
 6. 进入工程源码目录：`cd ~/catkin_ws/src`
 7. 下载velodyne driver：`git clone https://github.com/ros-drivers/velodyne.git`
 8. 下载loam_velodyne：`git clone https://github.com/laboshinl/loam_velodyne.git`
 9. 根据loam_velodyne的issue，修改loam_velodyne源码：

```
    vim loam_velodyne/CMakeLists.txt
    注释掉：add_definitions( -march=native )
```

10. 回到源码的根目录：`cd ~/catkin_ws`
11. 编译源码：`catkin_make -DCMAKE_BUILD_TYPE=Release`
12. 开启3个Terminal
13. 在第1个terminal运行：`source /ros_entrypoint.sh; roscore > /dev/null`
14. 在第2个terminal运行：

```
    source /ros_entrypoint.sh; 
    source ~/catkin_ws/devel/setup.bash; 
    cd ~/catkin_ws;
    roslaunch loam_velodyne loam_velodyne.launch
```

 15. 此时host界面已经出现如效果图一样的UI，只是里面暂时没有点云而已
 16. 在第3个terminal运行：
 
 ```
 source /ros_entrypoint.sh; 
 source ~/catkin_ws/devel/setup.bash; 
 cd ~/catkin_ws;
 roslaunch velodyne_pointcloud VLP16_points.launch pcap:="$HOME/Downloads/velodyne.pcap"
 注意pcap数据可以在此处下载：https://data.kitware.com/#folder/5b7fff608d777f06857cb539
```

 17. 完成。如果还是不行，请在下面留言
