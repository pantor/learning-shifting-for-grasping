# Bin Picking

Let a robot learn how to grasp objects out of a bin by itself. This task called bin picking is of high industrial importance, and oftentimes we don't know the geometry of the grasped object. Therefore, the robot tries to learn grasping via try and error. It needs around 20000 grasp tries using active learning to grasp reliably. This repo inclused a range of extensions: Specific grasps, side grasps, reactive grasps and pushing of objects.


## Structure

The overall structure is as follows:
 - *Database Server* There is a database server for collecting and uploading data and images to another computer. The server has a web interface for showing all actions for all databases in a given server and to show the latest action live. 
 - *Include / Src* The main C++ project for controlling the robot and infer the next best action. It uses the TensorFlow C++ API and MoveIt! for robot control. The camera driver is also included, either via direct access or and optional ensenso node. The latter is helpful because the camera needs a long time to connect and crashes sometimes afterwards.
 - *Learning*
 - *Scripts* It is recommended to export to PYTHONPATH in `.bashrc`: `export PYTHONPATH=$PYTHONPATH:$HOME/Documents/bin_picking/scripts`
 - *Test*

This project is a ROS package with launch files and a package.xml. The ROS node /move_group is set to respawn=true. This enables to call rosnode kill /move_group to restart it.


## Start

For an easy start, run `sh terminal-setup.sh` for a complete terminal setup. Then run `roslaunch bin_picking moveit.launch`, `roslaunch bin_picking bin_picking.launch`, check the database server or start a jupyter notebook.


## Config




## Installation

For the robotic hardware, make sure to load `launch/gripper-config.json` as the Franka end-effector configuration. Currently, following dependencies need to be installed:
- ROS Kinetic
- libfranka & franka_ros
- TensorFlow (with C++)
- EnsensoSDK
- Eigen unsupported (Euler angles, 3.4.0 dev)
- yaml-cpp
- Cereal
- curl & cpr
- Catch2


## Robot Learning Database

A database, server and viewer for research around robotic grasping.

![database-screenshot](doc/database-screenshot.png?raw=true)

The robot learning database is based on SQLite, Flask, Vue.js. It shows the entire action database as well as live actions. It can also delete recorded actions. The server can be started via `python3 database/app.py`, afterwards open [localhost](127.0.0.1:8080) in your browser.


## Training


