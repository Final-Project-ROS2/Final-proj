# Project Setup and Troubleshooting Guide

This README provides detailed step-by-step procedures for various tasks related to the project.  
Use the **Table of Contents** below to navigate to the specific guide you need.

---

## 📖 Table of Contents
- [1. Connecting to ChulaWifi on Ubuntu 22.04](#1-connecting-to-chulawifi-on-ubuntu-2204)
- [2. Running Gazebo Simulation for Testing](#2-running-gazebo-simulation-for-testing)
- [3. ROS2 Node with Python Virtual Environment](#3-ros2-vision-node-with-python-virtual-environment)
- [4. (More sections to be added...)](#3-more-sections-to-be-added)
---

## 1. Connecting to ChulaWifi on Ubuntu 22.04

Follow these steps to connect your Ubuntu 22.04 system to **ChulaWifi**:

1. Click the **set of icons** at the **top right corner** of the screen.
2. Select **Wi-Fi / Network Settings**.
3. Choose **ChulaWifi** from the list of available networks.
4. If it **auto-connects**, you’re done! ✅  
   Otherwise, proceed with the following configuration:
   - **Authentication:** Protected EAP (PEAP)  
   - **Anonymous identity:** *(leave blank)*  
   - **CA certificate:** Tick the **"No CA certificate required"** box  
   - **PEAP version:** Automatic  
   - **Inner authentication:** MSCHAPv2  
   - **Username:** Your ChulaWifi username (default: your student ID)  
   - **Password:** Your ChulaWifi password (default: same as REG Chula password)
5. Click **Connect**.

Once configured correctly, your Ubuntu system should automatically reconnect to **ChulaWifi** in the future.

---

## 2. Running Gazebo Simulation for Testing

Follow these steps to start up the **Gazebo simulation** environment with the correct world and utility nodes:

1. **Build all packages**
   - Open a terminal.
   - Navigate to your workspace:
     ```bash
     cd ~/final_project_ws
     ```
   - Build all packages:
     ```bash
     colcon build
     ```

2. **Source the default virtual environment**
   - Open a **second terminal**.
   - Go to the same workspace:
     ```bash
     cd ~/final_project_ws
     ```
   - Source the default virtual environment:
     ```bash
     source ./venv/bin/activate
     ```
   - Verify activation by checking that the terminal prompt begins with:

     ```
     (venv)
     ```
     
3. **Source the workspace**
   - In the same terminal, source the environment setup:
     ```bash
     source install/setup.bash
     ```

4. **Launch Gazebo with the UR5 setup**
   - In the same terminal, run:
     ```bash
     ros2 launch ur_yt_sim spawn_ur5_camera_gripper_moveit.launch.py
     ```
   - This will start **Gazebo** with the appropriate world environment, UR5 robot arm, camera, gripper, and supporting utilities.

5. **Verify simulation environment**
   - Wait for all models (robot arm, tables, items) to load successfully.
   - You can now begin testing.

6. **Important ROS topics**
   - The raw RGB image from the depth camera is published to:
     ```
     /camera/image_raw
     ```
   - The depth image data is published to:
     ```
     /camera/depth/image_raw
     ```
     
## 3. ROS2 Vision Node with Python Virtual Environment

Follow these steps to create and run a **ROS2 Python node** inside a **virtual environment**:

1. **Navigate to the source folder**
   ```bash
   cd ~/final_project_ws/src
   ```

2. **Create a Python ROS2 package**

   ```bash
   ros2 pkg create --build-type ament_python --license Apache-2.0 <package-name>
   ```

3. **Open the package in VS Code**

   ```bash
   code <package-name>
   ```

4. **Modify the `setup.cfg`**

   * Add the following lines at the **top** of the file:

     ```ini
     [build_scripts]
     executable=/usr/bin/env python3
     ```

5. **Edit your package and create Python nodes** as usual.

6. **Build the workspace**

   * Open a new terminal:

     ```bash
     cd ~/final_project_ws
     colcon build
     ```

7. **Activate the vision virtual environment**

   * Open another new terminal:

     ```bash
     cd ~/final_project_ws
     source ./vision_venv/bin/activate
     ```
   * Verify activation by checking that the terminal prompt begins with:

     ```
     (vision_venv)
     ```

8. **Source the ROS2 workspace**

   ```bash
   source install/setup.bash
   ```

9. **Run your Python node**

   ```bash
   ros2 run <package-name> <executable-name>
   ```

✅ You are now running your ROS2 Python node inside an isolated **virtual environment**, ensuring dependency consistency and easier package management.

---

## 4. (More sections to be added)

Additional guides will be added here in the future, such as:

* Setting up ROS2 environment
* Configuring MoveIt2 for UR robots
* Dual boot setup for simulation and real hardware
* And more...


---


