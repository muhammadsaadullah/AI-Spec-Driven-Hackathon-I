---
title: "URDF for Humanoids"
description: "Understanding and creating URDF files for humanoid robots in ROS 2"
keywords: ["urdf", "humanoid", "robot description", "ros2", "kinematics", "robotics"]
sidebar_position: 4
---

# URDF for Humanoids

URDF (Unified Robot Description Format) is an XML format used in ROS to describe robot models. For humanoid robots, URDF defines the physical structure, kinematic chain, and visual properties of the robot.

## Learning Objectives

By the end of this module, you will be able to:
- Create and structure URDF files for humanoid robots
- Define joints, links, and their properties for bipedal locomotion
- Understand the kinematic chain of humanoid robots
- Visualize and validate URDF models in ROS 2
- Apply best practices for humanoid robot description

## Prerequisites

- ROS 2 fundamentals
- Basic understanding of robot kinematics
- XML familiarity

## URDF Structure for Humanoids

A humanoid robot URDF typically includes:
- A base/torso link
- Head, arms (with shoulders, elbows, wrists), and legs (with hips, knees, ankles)
- Proper joint definitions for locomotion and manipulation
- Visual and collision properties

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Hip joint and link -->
  <joint name="hip_joint" type="fixed">
    <parent link="base_link"/>
    <child link="hip_link"/>
    <origin xyz="0 0 -0.25" rpy="0 0 0"/>
  </joint>

  <link name="hip_link">
    <visual>
      <geometry>
        <box size="0.25 0.25 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.25 0.25 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

## Humanoid Joint Types and Ranges

### Hip Joints (6 DOF)
- Hip yaw, pitch, roll for balance and locomotion
- Typically revolute or continuous joints
- Range of motion critical for stable walking

### Knee Joints (1 DOF)
- Flexion/extension only
- Revolute joints with limited range
- Critical for shock absorption and power transfer

### Ankle Joints (2-3 DOF)
- Pitch and roll for balance
- May include limited yaw
- Important for terrain adaptation

### Shoulder Joints (3 DOF)
- Pitch, roll, yaw for manipulation
- Spherical joint approximated with multiple revolute joints

## Creating Humanoid Kinematic Chains

### Leg Chain Example
```xml
<!-- Left Leg Chain -->
<joint name="left_hip_yaw_joint" type="revolute">
  <parent link="hip_link"/>
  <child link="left_thigh_link"/>
  <origin xyz="0.0 0.1 -0.05" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
</joint>

<link name="left_thigh_link">
  <visual>
    <geometry>
      <cylinder radius="0.05" length="0.4"/>
    </geometry>
    <origin xyz="0 0 -0.2"/>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.05" length="0.4"/>
    </geometry>
    <origin xyz="0 0 -0.2"/>
  </collision>
  <inertial>
    <mass value="3.0"/>
    <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.05"/>
  </inertial>
</link>

<joint name="left_knee_joint" type="revolute">
  <parent link="left_thigh_link"/>
  <child link="left_shin_link"/>
  <origin xyz="0.0 0.0 -0.4" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-2.0" upper="0.1" effort="100" velocity="1.0"/>
</joint>

<link name="left_shin_link">
  <visual>
    <geometry>
      <cylinder radius="0.04" length="0.4"/>
    </geometry>
    <origin xyz="0 0 -0.2"/>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.04" length="0.4"/>
    </geometry>
    <origin xyz="0 0 -0.2"/>
  </collision>
  <inertial>
    <mass value="2.0"/>
    <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.02"/>
  </inertial>
</link>

<joint name="left_ankle_joint" type="revolute">
  <parent link="left_shin_link"/>
  <child link="left_foot_link"/>
  <origin xyz="0.0 0.0 -0.4" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-0.5" upper="0.5" effort="50" velocity="1.0"/>
</joint>

<link name="left_foot_link">
  <visual>
    <geometry>
      <box size="0.2 0.1 0.05"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="0.2 0.1 0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.01"/>
  </inertial>
</link>
```

## Visualizing URDF Models

### Using RViz
```bash
# Launch robot state publisher with your URDF
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="your_urdf_content"

# Visualize in RViz
ros2 run rviz2 rviz2
```

### Using Joint State Publisher GUI
```bash
# For interactive joint manipulation
ros2 run joint_state_publisher_gui joint_state_publisher_gui
```

## Humanoid-Specific Considerations

### Center of Mass
- Critical for stable locomotion
- Should be positioned within the support polygon
- Model mass distribution accurately

### Balance and Stability
- Ankle joints crucial for balance control
- Hip joints important for whole-body balance
- Proper inertial properties for dynamics simulation

### Manipulation Capabilities
- Shoulder and elbow joints for reaching
- Wrist joints for tool manipulation
- Hand models for grasping (optional)

## Best Practices for Humanoid URDF

### File Organization
- Separate URDF into multiple files (head.urdf.xacro, arm.urdf.xacro, etc.)
- Use Xacro macros for parameterization
- Include proper naming conventions

### Performance Optimization
- Use simplified collision geometry where possible
- Limit visual mesh complexity
- Optimize joint limits for realistic motion

### Validation
- Check for proper kinematic chain closure
- Verify joint limits and ranges
- Test in simulation before physical implementation

## Advanced Topics

### Transmission Elements
For actuator simulation and control:
```xml
<transmission name="left_hip_yaw_trans">
  <type>transmission_interface/SimpleTansmission</type>
  <joint name="left_hip_yaw_joint">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_hip_yaw_motor">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Gazebo Integration
Include Gazebo-specific elements for simulation:
```xml
<gazebo reference="base_link">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
</gazebo>
```

## Next Steps

After mastering URDF for humanoids, explore [Gazebo Simulation Setup](/docs/modules/gazebo-unity/simulation-setup) to learn how to simulate your humanoid robot model in a physics environment.