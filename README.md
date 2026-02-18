# Project Setup and Troubleshooting Guide

This README provides detailed step-by-step procedures for various tasks related to the project.
Use the **Table of Contents** below to navigate to the specific guide you need.

---

## 📖 Table of Contents

* [1. Connecting to ChulaWifi on Ubuntu 22.04](#1-connecting-to-chulawifi-on-ubuntu-2204)
* [2. Running Gazebo Simulation for Testing](#2-running-gazebo-simulation-for-testing)
* [3. ROS2 Node with Python Virtual Environment](#3-ros2-vision-node-with-python-virtual-environment)
* [4. Connecting to UR Arm](#4-connecting-to-ur-arm)
* [5. Fixing Blocking Node (Nested Service/Action Calls)](#5-fixing-blocking-node-nested-serviceaction-calls)
* [6. Building and Running the Project](#6-building-and-running-the-project)
* [7. Checking Depth Camera Connection](#7-checking-depth-camera-connection)
* [8. (More sections to be added...)](#8-more-sections-to-be-added)

---

## 1. Connecting to ChulaWifi on Ubuntu 22.04

Follow these steps to connect your Ubuntu 22.04 system to **ChulaWifi**:

1. Click the **set of icons** at the **top right corner** of the screen.
2. Select **Wi-Fi / Network Settings**.
3. Choose **ChulaWifi** from the list of available networks.
4. If it **auto-connects**, you’re done! ✅
   Otherwise, proceed with the following configuration:

   * **Authentication:** Protected EAP (PEAP)
   * **Anonymous identity:** *(leave blank)*
   * **CA certificate:** Tick the **"No CA certificate required"** box
   * **PEAP version:** Automatic
   * **Inner authentication:** MSCHAPv2
   * **Username:** Your ChulaWifi username (default: your student ID)
   * **Password:** Your ChulaWifi password (default: same as REG Chula password)
5. Click **Connect**.

Once configured correctly, your Ubuntu system should automatically reconnect to **ChulaWifi** in the future.

---

## 2. Running Gazebo Simulation for Testing

Follow these steps to start up the **Gazebo simulation** environment with the correct world and utility nodes:

1. **Build all packages**

   * Open a terminal.
   * Navigate to your workspace:

     ```bash
     cd ~/final_project_ws
     ```
   * Source the `vision_venv` virtual environment:

     ```bash
     source ./vision_venv/bin/activate
     ```
   * Verify activation by checking that the terminal prompt begins with:

     ```
     (vision_venv)
     ```
   * Build all packages:

     ```bash
     colcon build
     ```

2. **Source the `vision_venv` virtual environment**

   * Open a **second terminal**.
   * Go to the same workspace:

     ```bash
     cd ~/final_project_ws
     ```
   * Source the `vision_venv` virtual environment:

     ```bash
     source ./vision_venv/bin/activate
     ```
   * Verify activation by checking that the terminal prompt begins with:

     ```
     (vision_venv)
     ```

3. **Source the workspace**

   * In the same terminal, source the environment setup:

     ```bash
     source install/setup.bash
     ```

4. **Launch Gazebo with the UR5 setup**

   * In the same terminal, run:

     ```bash
     ros2 launch ur_yt_sim final_project.launch.py
     ```
   * This will start **Gazebo** with the appropriate world environment, UR5 robot arm, camera, gripper, and supporting utilities.

5. **Verify simulation environment**

   * Wait for all models (robot arm, tables, items) to load successfully.
   * You can now begin testing.

6. **Important ROS topics**

   * The raw RGB image from the depth camera is published to:

     ```
     /camera/image_raw
     ```
   * The depth image data is published to:

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

---

## 4. Connecting to UR Arm

Follow these steps to safely connect and control the **Universal Robots (UR) arm** using ROS2:

### **Step 1 — Prepare the UR Arm**

1. Plug the UR arm into a **wall socket** (avoid adapters with unstable power).
2. Connect your **computer to the UR arm** using a LAN cable.
3. On the teach pendant:

   * Press the **silver start button** next to the **red e-stop**.
   * Press the **power button** (bottom-left of the screen).
   * Press **On**, then **Start**, then **Exit**.

### **Step 2 — Configure Network & Verify Connectivity**

1. On the teach pendant, open **Menu (top-right)** → **Settings** → **System** → **Network**.
2. Ensure the following:

   * **Network mode:** Static Address
   * **IP address:** `10.10.0.60`
   * **Subnet mask:** `255.255.0.0`
3. Disable Ethernet/IP adapter:

   * Go to: **Installation** → **Fieldbus** → **Ethernet/IP** → **Disable**
4. Check expected host (computer) IP:

   * **Installation** → **URCaps** → **External Control**
   * Host IP: `10.10.0.5`
   * Host name: `10.10.0.5`
   * Custom port: `50002`

### **Step 3 — Configure Ubuntu Static IP (IMPORTANT)**

If not configured correctly, Ubuntu may show:

```
Activation of network connection failed
```

This usually means your PC is not in the same subnet as the robot.

Since the robot IP is:

```
10.10.0.60
Subnet: 255.255.0.0
```

Your computer must be:

```
10.10.X.X
```

Recommended:

```
10.10.0.5
```

---

#### Configure via Ubuntu GUI (20.04 / 22.04 / 24.04)

1. Open **Settings**

2. Go to **Network**

3. Under **Wired**, click the ⚙ (gear icon)

4. Go to the **IPv4** tab

5. Change:

```
Automatic (DHCP)
```

to

```
Manual
```

6. Click **Add** and enter:

| Field   | Value                         |
| ------- | ----------------------------- |
| Address | 10.10.0.5                     |
| Netmask | 255.255.0.0                   |
| Gateway | (leave blank if direct cable) |

Leave **DNS blank**.

7. Click **Apply**
8. Turn the wired connection **Off → On**

---

#### Verify Configuration

Open terminal:

```bash
ip addr
```

You should see:

```
inet 10.10.0.5/16
```

Test connectivity:

```bash
ping 10.10.0.60
```

You should receive replies.

### **Step 4 — Run the External Control Program (ROS2)**

1. On your computer:

   ```bash
   cd ~/final_project_ws
   source install/setup.bash
   ros2 launch ur_yt_sim final_project.launch.py real_hardware:=true
   ```
2. On the UR teach pendant:

   * Open **Program** → **URCaps** → **External Control**
   * Press the **Run** button (lower-right corner)

Your UR arm should now be controlled via ROS2.

---

## 5. Fixing Blocking Node (Nested Service/Action Calls)

ROS2 developers often encounter a very common issue:
A node provides a **service or action**, and inside its callback, that same node tries to call **other services or actions**.
This leads to **deadlocks**, where:

* The first request works
* Every subsequent request hangs
* The node becomes unresponsive
* `spin_once()` loops inside callbacks make the problem worse

This happens because ROS2 callbacks **cannot re-enter** unless explicitly configured.

---

### ⭐ **Root Cause**

When a node:

* handles a service callback **and**
* inside that callback, calls another async service and waits with a loop

The executor **cannot** run the nested service response callback, because the thread is stuck inside the parent callback.

This results in complete deadlock.

---

### ✅ Correct Fix — Use Synchronous Calls + Reentrant Callback Group + MultiThreadedExecutor

ROS2 provides a clean and safe pattern for handling nested service calls **without blocking the node**.

#### ✔️ Use a **ReentrantCallbackGroup**

Allows callbacks in the same group to run concurrently.

#### ✔️ Use a **MultiThreadedExecutor**

Allows simultaneous execution of nested callbacks in separate threads.

#### ✔️ Use **synchronous service calls** (`client.call()`)

Not async calls inside callbacks.

No `spin_once()`, no polling loops, no manual spinning.

---

### 📌 Code Pattern (Working Solution)

```python
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import rclpy
from rclpy.node import Node

class MyServiceNode(Node):
    def __init__(self):
        super().__init__('my_service_node')

        # Create callback group that allows reentrancy
        self.callback_group = ReentrantCallbackGroup()

        # Service server
        self.my_service = self.create_service(
            MyServiceType,
            '/my_service',
            self.service_callback,
            callback_group=self.callback_group
        )

        # Service client
        self.other_service_client = self.create_client(
            OtherServiceType,
            '/other_service',
            callback_group=self.callback_group
        )

    def service_callback(self, request, response):
        # ❌ Wrong: async call + manual spin loop → deadlock
        # future = self.other_service_client.call_async(req)
        # while not future.done():
        #     rclpy.spin_once(self, timeout_sec=0.1)

        # ✅ Correct: synchronous call (blocking + safe)
        result = self.other_service_client.call(request)

        # Process result
        response.data = result.data
        return response


def main():
    rclpy.init()

    node = MyServiceNode()

    # Use MultiThreadedExecutor to enable nested callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

---

### 🧠 Fix Guide to use in Prompt for LLM to Fix Your Code

```markdown
# ROS2 Nested Service Calls - Deadlock Fix

## Problem Description
When a ROS2 service callback needs to call other services (nested service calls), using **async calls with manual spinning** creates a deadlock:
- The service callback blocks while waiting for nested service responses
- Calling `rclpy.spin_once(self, ...)` inside the callback fails because the main callback is already executing
- Subsequent requests to the parent service are never processed

## Solution: Use Synchronous Service Calls

Replace async service calls (`call_async()`) with synchronous calls (`call()`) inside service callbacks.

### Setup Requirements
1. Use `ReentrantCallbackGroup` for services that make nested calls
2. Use `MultiThreadedExecutor` to allow concurrent execution

### Code Pattern

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class MyServiceNode(Node):
    def __init__(self):
        super().__init__('my_service_node')
        
        # Create reentrant callback group
        self.callback_group = ReentrantCallbackGroup()
        
        # Create service server with callback group
        self.my_service = self.create_service(
            MyServiceType,
            '/my_service',
            self.service_callback,
            callback_group=self.callback_group
        )
        
        # Create service clients with same callback group
        self.other_service_client = self.create_client(
            OtherServiceType,
            '/other_service',
            callback_group=self.callback_group
        )
    
    def service_callback(self, request, response):
        # ❌ WRONG: Async call with manual spinning (causes deadlock)
        # future = self.other_service_client.call_async(req)
        # while not future.done():
        #     rclpy.spin_once(self, timeout_sec=0.1)
        # result = future.result()
        
        # ✅ CORRECT: Synchronous call
        result = self.other_service_client.call(req)
        
        # Process result and return response
        response.data = result.data
        return response

def main():
    rclpy.init()
    node = MyServiceNode()
    
    # Use MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

### Key Points

- **Synchronous calls** (`client.call()`) block but work correctly with `ReentrantCallbackGroup` and `MultiThreadedExecutor`
- The multi-threaded executor allows nested service calls to execute on different threads
- No manual spinning needed - the executor handles everything
- Service callbacks remain simple and readable

### When to Use This Pattern

- Service callbacks that need to call other services
- Action callbacks that call services
- Any callback that requires nested/chained service requests
- Scenarios where service orchestration is needed

---

## 6. Building and Running the Project

This project requires **four separate terminals** to build and run all components properly:

1. **Build terminal**
2. **Main launch terminal**
3. **Microphone connector terminal (or Action calling terminal)**
4. **Response monitoring terminal**

> **Important:** All terminals should start in the same workspace:
>
> ```bash
> cd ~/final_project_ws
> ```

---

### 1. **Build the workspace**

* Open the **first terminal**.
* Navigate to the workspace:

  ```bash
  cd ~/final_project_ws
  ```
* Activate the vision virtual environment:

  ```bash
  source ./vision_venv/bin/activate
  ```
* Build the project:

  ```bash
  colcon build
  ```
* Wait for the build process to complete successfully before proceeding.

---

### 2. **Set up the microphone environment**

* Open a **second terminal**.
* Go to the workspace:

  ```bash
  cd ~/final_project_ws
  ```
* Activate the microphone virtual environment:

  ```bash
  source ./venv/bin/activate
  ```
* Verify activation by checking that the terminal prompt begins with:

  ```
  (venv)
  ```

---

### 3. **Run the project**

The project can be executed in **three different modes**, depending on whether simulation or real hardware is used. Select the mode by supplying the appropriate value to the `mode` launch argument.

---

#### **Mode 1 (sim): Fully in simulation**

* In the **main launch terminal**, activate the vision virtual environment:

  ```bash
  source ./vision_venv/bin/activate
  ```

* In the **main launch terminal**, run:

  ```bash
  ros2 launch ur_yt_sim final_project.launch.py mode:=sim
  ```

* In the **microphone connector terminal**, run:

  ```bash
  ros2 launch asr asr.launch.py
  ```

> 💡 The project can still be used **without the microphone** by directly calling ROS 2 actions.

---

#### **Mode 2 (cam): Simulation world + real depth camera**

* In the **main launch terminal**, activate the vision virtual environment:

  ```bash
  source ./vision_venv/bin/activate
  ```

* In the **main launch terminal**, run:

  ```bash
  ros2 launch ur_yt_sim final_project.launch.py mode:=cam
  ```

* Microphone setup remains the same as in **Mode 1**:

  ```bash
  ros2 launch asr asr.launch.py
  ```

---

#### **Mode 3 (real): Real hardware**

* In the **main launch terminal**, activate the vision virtual environment:

  ```bash
  source ./vision_venv/bin/activate
  ```

* In the **main launch terminal**, run:

  ```bash
  ros2 launch ur_yt_sim final_project.launch.py mode:=real
  ```

* Microphone setup remains the same as in **Mode 1**:

  ```bash
  ros2 launch asr asr.launch.py
  ```

---

### 4. **Main launch file arguments**

The main launch file (`final_project.launch.py`) supports the following arguments:

| Argument          | Type   | Default                        | Description                                                          |
| ----------------- | ------ | ------------------------------ | -------------------------------------------------------------------- |
| `mode`            | string | `sim`                          | Select launch mode from `sim`, `cam`, and `real`                     |
| `tcp_offset`      | bool   | `false`                        | Apply tcp offset to z coordinate when true                           |
| `pddl`            | bool   | `false`                        | Use PDDL-based planning when true; use LLM-based planning when false |
| `world_file`      | string | `test_world_find_object.world` | Gazebo world file (must be located in `ur_yt_sim/worlds`)            |
| `use_ollama`      | bool   | `false`                        | Use local Ollama LLM instead of Google Gemini                        |
| `real_hardware`   | bool   | `false`                        | Use real robot hardware instead of Gazebo simulation                 |
| `real_camera`     | bool   | `false`                        | Use a real depth camera instead of the simulated camera              |
| `confirm`         | bool   | `true`                         | Require user confirmation before plan execution                      |
| `is_home`         | bool   | `true`                         | Set initial robot state for PDDL mode                                |
| `is_ready`        | bool   | `false`                        | Set initial robot state for PDDL mode                                |
| `gripper_is_open` | bool   | `true`                         | Set initial gripper state for PDDL mode                              |

---

### 5. **Monitoring the response**

* In the **response monitoring terminal**, run the following command:

  ```bash
  ros2 topic echo /response
  ```

The planning node responses will be printed in this terminal

---

### 6. **Using the project**

There are **2** ways to send command to the UR Arm.

#### **1. Vocal**

Make sure that you've launch the ASR node (by running `ros2 launch asr asr.launch.py` in the **microphone terminal** if you haven't). Then simply speak your command.

#### **2. Action Call**

With this method, you do not need to launch the ASR node. Simply run the following command

```bash
ros2 action send_goal /prompt_high_level custom_interfaces/action/Prompt "{prompt: '<command>'}"
```

Replacing the \<command\> with your command. For example

```bash
ros2 action send_goal /prompt_high_level custom_interfaces/action/Prompt "{prompt: 'Move to ready'}"
```

Make sure the double-quotes and single-quotes are present.

---

## 7. Checking Depth Camera Connection

To check whether the depth camera is connected to the computer correctly or not, do the following

* Open 3 terminals
* For all 3 terminals, navigate to the final project workspace by running
  ```bash
  cd ~/final_project_ws/
  ```
* In the first terminal, build the project by running
  ```bash
  colcon build
  ```
* In the second terminal, source the project, then run the depth camera publisher by running
  ```
  source install/setup.bash
  ros2 run depth_camera intel_pub
  ```
* In the third terminal, source the project, then run the depth camera subscriber by running
  ```
  source install/setup.bash
  ros2 run depth_camera intel_sub
  ```

---

## 8. (More sections to be added)

Additional guides will be added here in the future, such as:

* Setting up ROS2 environment
* Configuring MoveIt2 for UR robots
* Dual boot setup for simulation and real hardware
* And more...

---