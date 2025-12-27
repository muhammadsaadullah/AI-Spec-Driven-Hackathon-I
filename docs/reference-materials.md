---
title: "Reference Materials"
description: "Glossary of terms and notation guide for the Physical AI & Humanoid Robotics course"
keywords: ["glossary", "notation", "terminology", "reference", "robotics", "ai"]
sidebar_position: 8
---

# Reference Materials

This reference guide provides definitions for key terms and notation used throughout the Physical AI & Humanoid Robotics course.

## Glossary

### A

**Actuator**: A component of a robot that converts energy into physical motion. Examples include servo motors, hydraulic cylinders, and pneumatic actuators.

**Artificial Intelligence (AI)**: The simulation of human intelligence processes by machines, especially computer systems. In robotics, AI enables robots to perceive, reason, and act autonomously.

**Autonomous**: Operating independently without human intervention. An autonomous robot can perform tasks without direct human control.

### B

**Balance Control**: Systems and algorithms that maintain a robot's stability, particularly important for bipedal humanoid robots.

**Bipedal**: Having two legs for locomotion. Humanoid robots are typically bipedal to navigate human environments.

### C

**Center of Mass (CoM)**: The point where the mass of an object is concentrated. Critical for robot balance and stability.

**Computer Vision**: A field of AI that enables computers to interpret and understand visual information from the world.

**Control Theory**: The study of how to influence the behavior of dynamical systems. In robotics, control theory is used to make robots move as desired.

**Conversational AI**: AI systems that can understand and respond to human language in a natural, conversational manner.

### D

**Deep Learning**: A subset of machine learning that uses neural networks with multiple layers to model complex patterns in data.

**Degrees of Freedom (DOF)**: The number of independent parameters that define the configuration of a mechanical system. A humanoid robot typically has 20-50+ DOF.

**Digital-to-Physical Transition**: The process of moving from digital AI systems to embodied intelligence that operates in physical space.

**Dynamics**: The study of forces and torques and their effect on motion. Robot dynamics are crucial for control and simulation.

### E

**Embodied Intelligence**: Intelligence that emerges from the interaction between an agent and its physical environment. The physical form influences cognitive processes.

**End-Effector**: The device at the end of a robotic arm that interacts with the environment, such as a gripper or tool.

**Environment Interaction**: The process by which an intelligent agent perceives and acts upon its physical environment.

### F

**Forward Kinematics**: The process of calculating the position and orientation of a robot's end-effector based on its joint angles.

**Force Control**: Control systems that regulate the forces applied by a robot during interaction with objects or the environment.

### G

**Gazebo**: A 3D dynamic simulator for robotics that provides accurate and efficient simulation of robot populations in complex environments.

**Generative AI**: AI systems that can generate new content, such as text, images, or actions, based on learned patterns.

### H

**Human-Robot Interaction (HRI)**: The study of interactions between humans and robots, focusing on design, evaluation, and implementation of robotic systems for human use.

**Humanoid Robot**: A robot with a body structure similar to that of a human, typically with a head, torso, two arms, and two legs.

### I

**Inverse Kinematics**: The process of calculating the joint angles required to achieve a desired position and orientation of a robot's end-effector.

**Isaac Sim**: NVIDIA's simulation environment built on Omniverse for robotics development, offering photorealistic simulation and synthetic data generation.

**Isaac ROS**: Hardware-accelerated ROS 2 packages for perception and navigation, optimized for NVIDIA GPUs.

**Intelligent Agent**: An autonomous entity that perceives its environment and takes actions to achieve goals.

### J

**Joint**: A connection between two or more links in a robot that allows relative motion between them.

### K

**Kinematics**: The study of motion without considering the forces that cause the motion. Includes forward and inverse kinematics.

### L

**Large Language Model (LLM)**: Advanced AI models that can understand and generate human-like text based on vast training datasets.

**LIDAR (Light Detection and Ranging)**: A remote sensing method that uses light in the form of a pulsed laser to measure distances.

**Locomotion**: The ability to move from one place to another. For humanoid robots, this typically refers to walking.

### M

**Manipulation**: The ability to handle or control objects using robot hands or end-effectors.

**Middleware**: Software that provides common services and capabilities to applications beyond what's offered by the operating system.

**Mobile Robot**: A robot that can move around in its environment, as opposed to fixed-location robots.

### N

**Navigation**: The ability of a robot to move through its environment to reach a desired location.

**Natural Language Processing (NLP)**: A field of AI focused on the interaction between computers and human language.

**Neural Network**: A computing system inspired by the human brain, used in machine learning to recognize patterns.

### P

**Perception**: The ability of a robot to interpret sensory information from its environment.

**Physical AI**: AI systems that operate in physical space and interact with the real world, as opposed to digital-only AI.

**Proprioceptive Sensors**: Sensors that measure internal robot state, such as joint encoders and IMUs.

**Proximal**: Referring to parts of the robot closer to the base or torso (opposite of distal).

### R

**ROS (Robot Operating System)**: A flexible framework for writing robot software, providing services like hardware abstraction and message passing.

**ROS 2**: The second generation of the Robot Operating System, designed for production environments.

**Rigid Body Dynamics**: The study of the motion of interconnected rigid bodies under the action of external forces.

**Robot Operating System (ROS)**: A collection of tools, libraries, and conventions for building robot applications.

### S

**Sensor Fusion**: The process of combining data from multiple sensors to improve perception accuracy.

**Sim-to-Real Transfer**: The process of transferring behaviors learned in simulation to real-world robots.

**SLAM (Simultaneous Localization and Mapping)**: The computational problem of constructing or updating a map of an unknown environment while simultaneously keeping track of an agent's location within it.

**State Estimation**: The process of estimating the internal state of a system from noisy measurements.

**Supervised Learning**: A machine learning approach where the model learns from labeled training data.

### T

**Torque**: A rotational force that causes rotation around an axis. Important for controlling robot joints.

**Transform**: A mathematical operation that converts coordinates from one frame to another in robotics.

**Trustworthy AI**: AI systems that are reliable, safe, and ethically aligned.

### U

**URDF (Unified Robot Description Format)**: An XML format for representing robot models in ROS.

**Unity**: A real-time 3D development platform that can be used for robot visualization and simulation.

**Unstructured Environment**: An environment without predefined paths or structures, requiring robots to navigate dynamically.

### V

**VLA (Vision-Language-Action)**: Systems that integrate visual perception, language understanding, and physical action.

**Vision System**: The components and algorithms that enable a robot to perceive and interpret visual information.

**Visual SLAM**: SLAM using visual sensors such as cameras instead of or in addition to other sensors.

### W

**Whole-Body Control**: Control strategies that consider the entire robot body to achieve tasks while respecting constraints.

**Workspace**: The space within which a robot can operate, defined by its kinematic constraints.

## Notation Guide

### Mathematical Notation

- **Scalars**: Lowercase letters (e.g., `x`, `y`, `t`)
- **Vectors**: Lowercase bold letters (e.g., **v**, **p**, **q**)
- **Matrices**: Uppercase bold letters (e.g., **R**, **T**, **J**)
- **Sets**: Uppercase blackboard bold letters (e.g., ℝ, ℕ, ℤ)

### Robotics-Specific Notation

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

### Coordinate Frames

- **World frame**: W (subscript W, e.g., **p**^W for position in world frame)
- **Base frame**: B (subscript B, e.g., **T**^B_W for transform from world to base)
- **End-effector frame**: E (subscript E, e.g., **p**^E for position in end-effector frame)
- **Camera frame**: C (subscript C, e.g., **p**^C for position in camera frame)

### Time Notation

- `t`: Continuous time
- `k`: Discrete time step
- `Δt`: Time step duration

### Common Abbreviations

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

### ROS-Specific Notation

- **Topics**: Prefixed with `/` (e.g., `/joint_states`, `/cmd_vel`)
- **Nodes**: Named descriptively (e.g., `robot_state_publisher`, `joint_state_publisher`)
- **Parameters**: Defined in launch files or parameter servers

### Units

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

### Mathematical Operators

- `∇`: Gradient operator
- `∫`: Integration
- `∑`: Summation
- `×`: Cross product
- `·`: Dot product
- `||·||`: Norm (magnitude)
- `∈`: Element of
- `→`: Approaches or transforms to
- `≜`: Defined as

## Acronyms and Initialisms

| Acronym | Full Form | Description |
|---------|-----------|-------------|
| AI | Artificial Intelligence | Simulation of human intelligence processes by machines |
| CoM | Center of Mass | Point where mass is concentrated in an object |
| DOF | Degrees of Freedom | Independent parameters defining system configuration |
| GPT | Generative Pre-trained Transformer | Type of large language model |
| HRI | Human-Robot Interaction | Study of interactions between humans and robots |
| IMU | Inertial Measurement Unit | Sensor measuring acceleration and rotation |
| Isaac | Intelligent System for Autonomous mObile Manipulation | NVIDIA's robotics platform |
| LIDAR | Light Detection and Ranging | Remote sensing method using pulsed laser light |
| LLM | Large Language Model | Advanced AI model for natural language processing |
| NLP | Natural Language Processing | Field of AI focused on computer-human language interaction |
| ROS | Robot Operating System | Flexible framework for writing robot software |
| SLAM | Simultaneous Localization and Mapping | Computational problem of mapping while localizing |
| URDF | Unified Robot Description Format | XML format for representing robot models |
| VLA | Vision-Language-Action | Systems integrating vision, language, and action |
| ZMP | Zero Moment Point | Point where net moment of ground reaction forces is zero |

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

This reference guide should be consulted whenever encountering unfamiliar terminology or notation in the course materials. Understanding these concepts is fundamental to mastering Physical AI and Humanoid Robotics.