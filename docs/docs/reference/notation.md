---
title: "Notation Guide"
description: "Notation guide for the Physical AI & Humanoid Robotics course"
keywords: ["notation", "mathematical notation", "robotics", "symbols", "conventions"]
sidebar_position: 2
---

# Notation Guide

This guide provides the mathematical notation and conventions used throughout the Physical AI & Humanoid Robotics course.

## Mathematical Notation

- **Scalars**: Lowercase letters (e.g., `x`, `y`, `t`)
- **Vectors**: Lowercase bold letters (e.g., **v**, **p**, **q**)
- **Matrices**: Uppercase bold letters (e.g., **R**, **T**, **J**)
- **Sets**: Uppercase blackboard bold letters (e.g., ℝ, ℕ, ℤ)

## Robotics-Specific Notation

- **q**: Joint angles vector
- **q̇**: Joint velocities vector
- **q̈**: Joint accelerations vector
- **J**: Jacobian matrix
- **T**: Transformation matrix
- **R**: Rotation matrix
- **p**: Position vector
- **v**: Velocity vector
- **ω**: Angular velocity vector
- **τ**: Torque vector
- **F**: Force vector

## Coordinate Frames

- **World frame**: W (subscript W, e.g., **p**^W for position in world frame)
- **Base frame**: B (subscript B, e.g., **T**^B_W for transform from world to base)
- **End-effector frame**: E (subscript E, e.g., **p**^E for position in end-effector frame)
- **Camera frame**: C (subscript C, e.g., **p**^C for position in camera frame)

## Time Notation

- `t`: Continuous time
- `k`: Discrete time step
- `Δt`: Time step duration

## Common Abbreviations

- **CoM**: Center of Mass
- **DOF**: Degrees of Freedom
- **HRI**: Human-Robot Interaction
- **IMU**: Inertial Measurement Unit
- **LIDAR**: Light Detection and Ranging
- **LLM**: Large Language Model
- **SLAM**: Simultaneous Localization and Mapping
- **URDF**: Unified Robot Description Format
- **VLA**: Vision-Language-Action
- **ZMP**: Zero Moment Point

## ROS-Specific Notation

- **Topics**: Prefixed with `/` (e.g., `/joint_states`, `/cmd_vel`)
- **Nodes**: Named descriptively (e.g., `robot_state_publisher`, `joint_state_publisher`)
- **Parameters**: Defined in launch files or parameter servers

## Units

- **Length**: Meters (m)
- **Time**: Seconds (s)
- **Mass**: Kilograms (kg)
- **Force**: Newtons (N)
- **Torque**: Newton-meters (Nm)
- **Velocity**: Meters per second (m/s)
- **Acceleration**: Meters per second squared (m/s²)
- **Angular Velocity**: Radians per second (rad/s)
- **Angular Acceleration**: Radians per second squared (rad/s²)
- **Angle**: Radians (rad)

## Mathematical Operators

- `∇`: Gradient operator
- `∫`: Integration
- `∑`: Summation
- `×`: Cross product
- `·`: Dot product
- `||·||`: Norm (magnitude)
- `∈`: Element of
- `→`: Approaches or transforms to
- `≜`: Defined as

## Conventions

### Mathematical Expressions

Mathematical expressions in this course follow standard mathematical notation:

- Vector components are written with subscripts: **v** = [v_x, v_y, v_z]
- Matrix elements are written with double subscripts: **R** = [r_11, r_12, r_13; r_21, r_22, r_23; r_31, r_32, r_33]
- Derivatives are written with dots for time derivatives: q̇ = dq/dt

### Code Notation

- ROS package names: `package_name`
- ROS topics: `/topic_name`
- ROS messages: `package_name/MessageName`
- Python modules: `module_name`
- Functions: `function_name()`
- Variables: `variable_name`

### Figures and Diagrams

- Coordinate systems: Right-handed, with Z pointing up
- Joint angles: Positive rotations follow the right-hand rule
- Forces: Drawn as arrows in the direction of action
- Reference frames: Labeled with origin point and axis directions