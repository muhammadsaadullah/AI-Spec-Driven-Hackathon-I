---
title: "ROS 2 Fundamentals"
description: "Learn the fundamentals of ROS 2 architecture, core concepts, and development practices"
keywords: ["ros2", "robotics", "middleware", "nodes", "topics"]
sidebar_position: 2
---

# ROS 2 Fundamentals

ROS 2 (Robot Operating System 2) is the middleware framework that provides services designed for a heterogeneous computer cluster, including hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more.

## Learning Objectives

By the end of this module, you will be able to:
- Understand ROS 2 architecture and core concepts
- Create and manage ROS 2 nodes, topics, services, and actions
- Build ROS 2 packages with Python
- Use launch files and parameter management
- Apply ROS 2 best practices for humanoid robotics

## Prerequisites

- Basic Python programming knowledge
- Understanding of robotics concepts
- Familiarity with Linux command line

## ROS 2 Architecture

ROS 2 uses a client library implementation approach that allows multiple implementations of the ROS client library API. The architecture is built around:

- **DDS (Data Distribution Service)**: Provides the underlying message passing infrastructure
- **RMW (ROS Middleware)**: Abstraction layer between ROS 2 and DDS implementations
- **ROS Client Libraries**: Language-specific libraries (rclcpp, rclpy) that provide the ROS API

## Core Concepts

### Nodes
Nodes are the fundamental building blocks of ROS 2 applications. They encapsulate functionality and communicate with other nodes through topics, services, and actions.

```python
# Example: Basic ROS 2 Node in Python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Minimal node started')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics
Topics enable asynchronous communication between nodes using a publish/subscribe pattern.

### Services
Services provide synchronous request/response communication between nodes.

### Actions
Actions provide goal-oriented communication with feedback and status updates.

## Building ROS 2 Packages with Python

To create a Python package for ROS 2:

1. Create the package structure:
```bash
ros2 pkg create --build-type ament_python my_robot_package
```

2. Add your Python nodes to the `my_robot_package/my_robot_package/` directory

3. Update the `setup.py` file to include your executables

4. Build the package:
```bash
colcon build --packages-select my_robot_package
```

## Launch Files and Parameter Management

Launch files allow you to start multiple nodes with a single command and manage parameters:

```python
# launch/my_launch_file.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='my_node',
            name='my_node',
            parameters=[
                {'param_name': 'param_value'}
            ]
        )
    ])
```

## Best Practices for Humanoid Robotics

- Use consistent naming conventions for topics and services
- Implement proper error handling and logging
- Structure your code for modularity and reusability
- Use standard ROS message types when possible
- Follow TF (Transform) conventions for coordinate frames

## Next Steps

After mastering these fundamentals, proceed to learn about [ROS 2 Nodes, Topics, and Services](/docs/modules/ros2/nodes-topics-services) to dive deeper into the communication mechanisms.