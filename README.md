# Robot Learning of Shifting Objects for Grasping in Cluttered Environments

This repository contains additional information for the submitted paper "Robot Learning of Shifting Objects for Grasping in Cluttered Environments" for IROS 2019. This code lets a robot learn how to grasp objects out of a bin by itself. As traditional approached oftentimes need the 3d model of the object, the robot in this project learns grasping in a self-supervised manner by try and error. Our focus relies on the data-efficiency of the learning process: Currently, it needs around 20000 grasp and around 3000 shift attempts to reliably empty a bin with a grasp rate of over 95%. Shifting is essential for bin picking, as it allows the robot to empty a bin completely.

Click the image for a **video** (or [here](https://drive.google.com/file/d/1-IE4kr5ICFjxqHVggU8nzf9ZE3paAUtz/view?usp=sharing))!
[![Watch the video](doc/overall-system-wide.jpg?raw=true)](https://drive.google.com/file/d/1-IE4kr5ICFjxqHVggU8nzf9ZE3paAUtz/view?usp=sharing)

Our overall setup of a Franka Panda robotic arm including the standard force-feedback gripper, an Ensenso stereo camera, custom 3D-printed gripper jaws with anti-slip tape, and two industrial bins with objects. The robot learns first grasping and then shifting objects to explicitly increase grasp success. The first sections give a short introduction into the source code. Later, we present more information about the paper, i.a. a more detailed evaluation.


## Structure

The overall structure is as follows:
 - *Database Server* There is a database server for collecting and uploading data and images to another computer. The server has a web interface for showing all actions for all databases in a given server and to show the latest action live. 
 - *Include / Src* The main C++ project for controlling the robot and infer the next best action. It uses the TensorFlow C++ API and MoveIt! for robot control. The camera driver is also included, either via direct access or and optional ensenso node. The latter is helpful because the camera needs a long time to connect and crashes sometimes afterwards.
 - *Scripts* It is recommended to export to PYTHONPATH in `.bashrc`: `export PYTHONPATH=$PYTHONPATH:$HOME/Documents/bin_picking/scripts`

This project is a ROS package with launch files and a package.xml. The ROS node /move_group is set to respawn=true. This enables to call rosnode kill /move_group to restart it.


## Running the Demo

After installing all dependencies (see next section), run both `roslaunch bin_picking moveit.launch` and `roslaunch bin_picking bin_picking.launch`. For recording, check the database server and the corresponding web interface. 


## Models

TensorFlow models (via the tf.saver API) for both grasping and pushing are in the `models` directory.


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
