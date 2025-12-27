---
title: "Weeks 6-7: Robot Simulation with Gazebo"
description: "Gazebo simulation environment setup, URDF and SDF robot description formats, physics simulation and sensor simulation"
keywords: ["gazebo", "simulation", "urdf", "sdf", "physics", "sensors"]
sidebar_position: 3
---

# Weeks 6-7: Robot Simulation with Gazebo

Welcome to the simulation module of the Physical AI & Humanoid Robotics course! These two weeks focus on creating realistic robot simulations using Gazebo, a powerful physics-based simulation environment. You'll learn to build and configure digital twins of robots that accurately represent their real-world counterparts.

## Learning Objectives

By the end of these two weeks, you will be able to:
- Set up and configure the Gazebo simulation environment
- Create robot models using URDF and SDF formats
- Configure physics properties and collision detection
- Simulate various sensor systems (LiDAR, cameras, IMUs)
- Integrate Gazebo with ROS 2 for robot control
- Understand the importance of simulation in robotics development

## Prerequisites

- Completion of Weeks 1-5 (Physical AI foundations and ROS 2)
- Basic understanding of 3D coordinate systems
- Familiarity with XML for robot description formats
- Ubuntu 22.04 with ROS 2 Humble installed

## Week 6: Gazebo Simulation Environment

### Day 1: Introduction to Gazebo

#### What is Gazebo?

Gazebo is a 3D dynamic simulator with the ability to accurately and efficiently simulate populations of robots in complex indoor and outdoor environments. It provides:
- High-fidelity physics simulation using ODE (Open Dynamics Engine)
- High-quality graphics rendering using OGRE
- Support for various sensors (cameras, LiDAR, IMUs, etc.)
- Integration with ROS 2 for robot control and communication

#### Installing Gazebo

Gazebo comes with ROS 2 Humble, but you can install additional components:

```bash
sudo apt update
sudo apt install ros-humble-gazebo-*
sudo apt install gazebo
```

#### Basic Gazebo Environment

Create your first Gazebo world:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="physical_ai_world">
    <!-- Include a default ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include a default sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Your robot will be spawned here -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.4 0.2 -1.0</direction>
    </light>
  </world>
</sdf>
```

#### Launching Gazebo

```bash
# Launch Gazebo with a default world
gazebo

# Launch with a specific world file
gazebo /path/to/your/world.world
```

### Day 2: URDF and SDF Fundamentals

#### URDF vs SDF

**URDF (Unified Robot Description Format)**:
- XML format for describing robot structure and kinematics
- Used primarily in ROS for robot description
- Focuses on kinematic chains and joint relationships
- Limited physics properties

**SDF (Simulation Description Format)**:
- XML format for describing simulation environments
- Used by Gazebo for physics simulation
- Includes detailed physics properties, materials, and sensors
- Can include multiple robots and static objects

#### URDF to SDF Conversion

Gazebo can automatically convert URDF to SDF, but you can also manually create SDF models for more control:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="simple_robot">
    <!-- Base link -->
    <link name="base_link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.01</iyy>
          <iyz>0.0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>

      <visual name="visual">
        <geometry>
          <box>
            <size>0.5 0.5 0.2</size>
          </box>
        </geometry>
        <material>
          <ambient>0.8 0.2 0.1 1</ambient>
          <diffuse>0.8 0.2 0.1 1</diffuse>
        </material>
      </visual>

      <collision name="collision">
        <geometry>
          <box>
            <size>0.5 0.5 0.2</size>
          </box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
```

### Day 3: Robot Model Creation

#### Creating a Simple Humanoid Base

Create a basic humanoid torso model in URDF:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base torso -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.004"/>
    </inertial>
  </link>

  <joint name="head_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.35"/>
  </joint>
</robot>
```

### Day 4: Gazebo Integration with ROS 2

#### Launch Files for Simulation

Create a launch file to spawn your robot in Gazebo:

```python
#!/usr/bin/env python3
# launch/gazebo_simulation.py

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/gazebo.launch.py'
        ])
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': open(os.path.join(
                get_package_share_directory('your_robot_description'),
                'urdf',
                'simple_humanoid.urdf'
            )).read()
        }]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'simple_humanoid',
            '-x', '0',
            '-y', '0',
            '-z', '0.5'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])
```

### Day 5: Hands-on Exercise - Basic Robot in Gazebo

#### Exercise: Create and Simulate a Simple Robot

1. Create a URDF file for a simple robot with a base and one joint
2. Create a launch file to spawn the robot in Gazebo
3. Verify that the robot appears in the simulation
4. Test basic movement if joints are configured

## Week 7: Advanced Simulation Concepts

### Day 6: Physics Simulation and Collision Detection

#### Physics Properties in SDF

Physics properties control how objects behave in the simulation:

```xml
<world name="physical_ai_world">
  <!-- Physics engine configuration -->
  <physics name="ode" type="ode">
    <gravity>0 0 -9.8</gravity>
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1.0</real_time_factor>
    <real_time_update_rate>1000</real_time_update_rate>
    <ode>
      <solver>
        <type>quick</type>
        <iters>10</iters>
        <sor>1.3</sor>
      </solver>
      <constraints>
        <cfm>0.0</cfm>
        <erp>0.2</erp>
        <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
        <contact_surface_layer>0.001</contact_surface_layer>
      </constraints>
    </ode>
  </physics>

  <!-- Your models here -->
</world>
```

#### Collision Detection Parameters

Fine-tune collision detection for humanoid robots:

```xml
<collision name="base_collision">
  <geometry>
    <box>
      <size>0.3 0.2 0.5</size>
    </box>
  </geometry>
  <surface>
    <contact>
      <ode>
        <soft_cfm>0.000001</soft_cfm>
        <soft_erp>0.2</soft_erp>
        <kp>1e+13</kp>
        <kd>1.0</kd>
        <max_vel>100.0</max_vel>
        <min_depth>0.001</min_depth>
      </ode>
    </contact>
    <friction>
      <ode>
        <mu>1.0</mu>
        <mu2>1.0</mu2>
      </ode>
    </friction>
  </surface>
</collision>
```

### Day 7: Sensor Simulation

#### Camera Simulation

Add a camera sensor to your robot:

```xml
<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.05 0.05 0.05"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="0.05 0.05 0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
  </inertial>
</link>

<joint name="camera_joint" type="fixed">
  <parent link="head"/>
  <child link="camera_link"/>
  <origin xyz="0.05 0 0" rpy="0 0 0"/>
</joint>

<gazebo reference="camera_link">
  <sensor type="camera" name="camera1">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
      <topic_name>camera/image_raw</topic_name>
    </plugin>
  </sensor>
</gazebo>
```

#### IMU Sensor Simulation

Add an IMU sensor for balance and orientation:

```xml
<gazebo reference="base_link">
  <sensor type="imu" name="imu_sensor">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <topic_name>imu/data</topic_name>
      <body_name>base_link</body_name>
      <frame_name>base_link</frame_name>
      <update_rate>100</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

#### LiDAR Sensor Simulation

Add a LiDAR sensor for navigation:

```xml
<gazebo reference="base_link">
  <sensor type="ray" name="lidar_sensor">
    <always_on>true</always_on>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-1.570796</min_angle>
          <max_angle>1.570796</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_plugin" filename="libgazebo_ros_laser.so">
      <topic_name>scan</topic_name>
      <frame_name>base_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### Day 8: Environment Creation

#### Creating Complex Environments

Build a training environment for humanoid robots:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_training">
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Obstacles for navigation training -->
    <model name="obstacle_1">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 1.0</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 1.0</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.166667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.166667</iyy>
            <iyz>0</iyz>
            <izz>0.166667</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Furniture for human-like environment -->
    <model name="table">
      <pose>0 2 0.4 0 0 0</pose>
      <link name="table_top">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.8 0.02</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.8 0.02</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.8 0.6 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>1.06667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1.41667</iyy>
            <iyz>0</iyz>
            <izz>2.43333</izz>
          </inertia>
        </inertial>
      </link>
      <link name="leg_1">
        <pose>-0.4 -0.3 -0.39 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.3 0.1 1</ambient>
            <diffuse>0.5 0.3 0.1 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.01667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.01667</iyy>
            <iyz>0</iyz>
            <izz>0.00125</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### Day 9: Integration with Unity

#### Introduction to Unity for Robot Visualization

Unity can be used alongside Gazebo for enhanced visualization:

- **Gazebo**: Handles physics simulation and sensor data
- **Unity**: Provides high-fidelity graphics and human-robot interaction visualization
- **ROS 2 Bridge**: Synchronizes data between both environments

#### Unity-ROS 2 Integration

Unity can connect to ROS 2 using the Unity Robotics Hub:

1. Install the Unity Robotics Hub package
2. Configure ROS 2 connection settings
3. Create visualization assets that mirror Gazebo simulation
4. Synchronize robot states between both environments

### Day 10: Practical Exercise - Complete Humanoid Simulation

#### Exercise: Build a Complete Humanoid Simulation

Create a complete humanoid robot with:
1. Full body URDF with joints
2. Multiple sensors (camera, IMU, LiDAR)
3. Physics properties tuned for humanoid movement
4. A training environment with obstacles
5. ROS 2 integration for control

## Assessment and Learning Verification

### Week 6 Assessment
1. **Technical Skills**: Create a URDF model of a simple robot and spawn it in Gazebo
2. **Understanding**: Explain the difference between URDF and SDF formats
3. **Application**: Launch a robot in Gazebo and verify sensor data publication

### Week 7 Assessment
1. **Integration**: Build a humanoid robot model with multiple sensors
2. **Problem Solving**: Create a custom environment for robot training
3. **Analysis**: Compare simulation vs. real-world robot behavior

## Resources and Further Reading

### Required Reading
- Gazebo Documentation: http://gazebosim.org/
- URDF Tutorials: http://wiki.ros.org/urdf/Tutorials
- SDF Specification: http://sdformat.org/

### Recommended Resources
- Gazebo ROS Packages: https://github.com/ros-simulation/gazebo_ros_pkgs
- Robot State Publisher: http://wiki.ros.org/robot_state_publisher
- TF2 Transform Library: http://wiki.ros.org/tf2

## Next Steps

After completing Weeks 6-7, you'll have a solid foundation in robot simulation using Gazebo. In the next module (Weeks 8-10), we'll explore the NVIDIA Isaac platform for advanced perception and AI-powered robotics applications.

The next module will cover:
- NVIDIA Isaac SDK and Isaac Sim
- AI-powered perception and manipulation
- Reinforcement learning for robot control
- Sim-to-real transfer techniques

Continue to the [Weeks 8-10: NVIDIA Isaac Platform](/docs/weekly-breakdown/weeks-8-10-nvidia-isaac) module to build on your simulation skills with advanced AI capabilities.