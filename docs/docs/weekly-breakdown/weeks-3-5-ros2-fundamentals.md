---
title: "Weeks 3-5: ROS 2 Fundamentals"
description: "ROS 2 architecture, core concepts, packages, launch files, and parameter management for humanoid robotics"
keywords: ["ros2", "robotics", "middleware", "nodes", "topics", "services", "humanoid"]
sidebar_position: 2
---

# Weeks 3-5: ROS 2 Fundamentals

This module provides a comprehensive introduction to ROS 2 (Robot Operating System 2), the middleware framework that enables communication between different software components in robotic systems. We'll focus on concepts and techniques specifically relevant to humanoid robotics.

## Learning Objectives

By the end of these three weeks, you will be able to:
- Understand ROS 2 architecture and core concepts
- Create and manage ROS 2 nodes with Python
- Implement publisher-subscriber communication
- Design and implement service-based communication
- Build ROS 2 packages with proper structure
- Configure and use launch files for complex systems
- Manage parameters and coordinate frames for humanoid robots

## Prerequisites

- Basic Python programming knowledge
- Understanding of Physical AI concepts (Weeks 1-2)
- Familiarity with Linux command line
- Basic understanding of robotics concepts

## Week 3: ROS 2 Architecture and Core Concepts

### Day 1: Introduction to ROS 2

#### What is ROS 2?

ROS 2 is the next generation of the Robot Operating System, designed to address limitations of ROS 1 and provide enterprise-ready robotics software development capabilities.

**Key Features of ROS 2:**
- **Real-Time Support**: Deterministic communication for time-critical applications
- **Multi-Robot Support**: Better tools for coordinating multiple robots
- **Security**: Built-in security features for safe robot deployment
- **ROS 1 Compatibility**: Tools for integrating with existing ROS 1 systems

#### ROS 2 Architecture

**DDS (Data Distribution Service)**
- Underlying communication layer
- Provides publish/subscribe and request/response communication
- Implements Quality of Service (QoS) policies
- Supports multiple implementations (Fast DDS, Cyclone DDS, RTI Connext)

**RMW (ROS Middleware)**
- Abstraction layer between ROS 2 and DDS implementations
- Enables middleware independence
- Provides consistent interface across DDS vendors

**Client Libraries**
- **rclcpp**: C++ client library
- **rclpy**: Python client library
- **rcl**: Core C library
- **rclc**: C library for embedded systems

#### Quality of Service (QoS) Policies

QoS policies define how messages are delivered between nodes:

```python
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

# Example QoS profiles for different scenarios

# Reliable communication for critical data
critical_qos = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE
)

# Best-effort for streaming data
streaming_qos = QoSProfile(
    history=QoSHistoryPolicy.KEEP_ALL,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE
)
```

### Day 2: Nodes and Basic Communication

#### Creating a ROS 2 Node

A ROS 2 node is the fundamental building block of a ROS 2 system. Here's a minimal example:

```python
#!/usr/bin/env python3
# minimal_node.py

import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Minimal node started')

        # Declare parameters
        self.declare_parameter('param_example', 'default_value')

        # Example timer
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        param_value = self.get_parameter('param_example').value
        self.get_logger().info(f'Timer callback {self.i}, param: {param_value}')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)

    minimal_node = MinimalNode()

    try:
        rclpy.spin(minimal_node)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Publisher Implementation

Creating a publisher to send messages:

```python
#!/usr/bin/env python3
# talker.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')

        # Create publisher
        self.publisher = self.create_publisher(String, 'chatter', 10)

        # Create timer to publish periodically
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)

    talker = Talker()

    try:
        rclpy.spin(talker)
    except KeyboardInterrupt:
        pass
    finally:
        talker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Subscriber Implementation

Creating a subscriber to receive messages:

```python
#!/usr/bin/env python3
# listener.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):
    def __init__(self):
        super().__init__('listener')

        # Create subscriber
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)

    listener = Listener()

    try:
        rclpy.spin(listener)
    except KeyboardInterrupt:
        pass
    finally:
        listener.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Day 3: Services and Actions

#### Services

Services provide request-response communication:

**Service Server:**
```python
#!/usr/bin/env python3
# minimal_service.py

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    try:
        rclpy.spin(minimal_service)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Service Client:**
```python
#!/usr/bin/env python3
# minimal_client.py

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        future = self.cli.call_async(self.req)
        return future

def main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClient()
    future = minimal_client.send_request(1, 2)

    try:
        rclpy.spin_until_future_complete(minimal_client, future)
        response = future.result()
        minimal_client.get_logger().info(f'Result: {response.sum}')
    except KeyboardInterrupt:
        pass
    finally:
        minimal_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Actions

Actions provide goal-oriented communication with feedback:

**Action Server:**
```python
#!/usr/bin/env python3
# fibonacci_action_server.py

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node

from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            self.get_logger().info(f'Publishing feedback: {feedback_msg.sequence}')

            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Returning result: {result.sequence}')

        return result

def main(args=None):
    rclpy.init(args=args)

    fibonacci_action_server = FibonacciActionServer()

    try:
        rclpy.spin(fibonacci_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        fibonacci_action_server.destroy()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Day 4: Parameters and Configuration

#### Parameter Management

ROS 2 provides a robust parameter system for configuration:

```python
#!/usr/bin/env python3
# parameter_demo.py

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType

class ParameterDemoNode(Node):
    def __init__(self):
        super().__init__('parameter_demo')

        # Declare parameters with default values and descriptions
        self.declare_parameter('robot_name', 'humanoid_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_mode', True)
        self.declare_parameter('joint_limits', [1.57, 1.57, 1.57])  # radians

        # Set up parameter change callback
        self.add_on_set_parameters_callback(self.parameters_callback)

        # Example timer to demonstrate parameter usage
        self.timer = self.create_timer(1.0, self.timer_callback)

    def parameters_callback(self, params):
        """Callback for parameter changes"""
        for param in params:
            self.get_logger().info(f'Parameter {param.name} changed to {param.value}')

        return SetParametersResult(successful=True)

    def timer_callback(self):
        robot_name = self.get_parameter('robot_name').value
        max_vel = self.get_parameter('max_velocity').value
        safety_mode = self.get_parameter('safety_mode').value

        self.get_logger().info(
            f'Robot: {robot_name}, Max Velocity: {max_vel}, Safety: {safety_mode}'
        )

def main(args=None):
    rclpy.init(args=args)

    parameter_demo = ParameterDemoNode()

    try:
        rclpy.spin(parameter_demo)
    except KeyboardInterrupt:
        pass
    finally:
        parameter_demo.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Using Parameter Files

Create parameter files to manage configuration:

```yaml
# config/robot_params.yaml
parameter_demo:
  ros__parameters:
    robot_name: "humanoid_v1"
    max_velocity: 0.5
    safety_mode: true
    joint_limits:
      - 1.57
      - 1.57
      - 1.57
    camera_resolution: [640, 480]
```

Loading parameters in a node:

```python
#!/usr/bin/env python3
# parameter_loader.py

import rclpy
from rclpy.node import Node
import yaml

class ParameterLoader(Node):
    def __init__(self):
        super().__init__('parameter_loader')

        # Load parameters from file
        self.load_parameters_from_file('config/robot_params.yaml')

    def load_parameters_from_file(self, file_path):
        """Load parameters from a YAML file"""
        try:
            with open(file_path, 'r') as file:
                params = yaml.safe_load(file)

            # Set parameters
            for node_name, node_params in params.items():
                if node_name == self.get_name():
                    for param_name, param_value in node_params.get('ros__parameters', {}).items():
                        self.declare_parameter(param_name, param_value)

        except FileNotFoundError:
            self.get_logger().error(f'Parameter file {file_path} not found')

def main(args=None):
    rclpy.init(args=args)

    param_loader = ParameterLoader()

    try:
        rclpy.spin(param_loader)
    except KeyboardInterrupt:
        pass
    finally:
        param_loader.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Day 5: Hands-on Exercise - Basic ROS 2 Node

Create a complete example that demonstrates multiple ROS 2 concepts:

```python
#!/usr/bin/env python3
# humanoid_controller.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'status', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        # Services
        self.safety_srv = self.create_service(
            SetBool, 'set_safety_mode', self.set_safety_mode_callback)

        # Parameters
        self.declare_parameter('max_linear_velocity', 0.5)
        self.declare_parameter('max_angular_velocity', 0.3)
        self.declare_parameter('safety_enabled', True)

        # Timers
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Internal state
        self.joint_positions = {}
        self.safety_mode = self.get_parameter('safety_enabled').value

        self.get_logger().info('Humanoid Controller initialized')

    def joint_state_callback(self, msg):
        """Process joint state messages"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def set_safety_mode_callback(self, request, response):
        """Handle safety mode service calls"""
        self.safety_mode = request.data
        self.get_logger().info(f'Safety mode set to: {self.safety_mode}')

        response.success = True
        response.message = f'Safety mode updated to: {self.safety_mode}'
        return response

    def control_loop(self):
        """Main control loop"""
        # Example: Publish status
        status_msg = String()
        status_msg.data = f'Joints: {len(self.joint_positions)}, Safety: {self.safety_mode}'
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)

    controller = HumanoidController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Week 4: Building ROS 2 Packages

### Day 6: Package Structure and Creation

#### ROS 2 Package Structure

A typical ROS 2 package follows this structure:

```
my_robot_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml            # Package metadata
├── setup.py              # Build configuration for Python
├── setup.cfg             # Python setup configuration
├── my_robot_package/     # Python package directory
│   ├── __init__.py      # Python package initialization
│   └── my_node.py       # Python node implementation
├── src/                  # C++ source files
│   └── my_node.cpp      # C++ node implementation
├── include/              # C++ header files
├── launch/               # Launch files
├── config/               # Configuration files
├── test/                 # Test files
└── README.md            # Package documentation
```

#### Creating a Package

Create a Python package using the command line:

```bash
# Create a Python package
ros2 pkg create --build-type ament_python my_humanoid_control

# Create a C++ package
ros2 pkg create --build-type ament_cmake my_humanoid_control_cpp
```

#### package.xml

The `package.xml` file contains package metadata:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_humanoid_control</name>
  <version>0.0.0</version>
  <description>Package for humanoid robot control</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <exec_depend>rclpy</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>std_srvs</exec_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

#### setup.py

The `setup.py` file defines how the Python package is built:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_humanoid_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        # Include all config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='user@example.com',
    description='Package for humanoid robot control',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'humanoid_controller = my_humanoid_control.humanoid_controller:main',
            'sensor_processor = my_humanoid_control.sensor_processor:main',
            'motion_planner = my_humanoid_control.motion_planner:main',
        ],
    },
)
```

### Day 7: Complex Node Implementation

Create a more sophisticated humanoid control node:

```python
# my_humanoid_control/humanoid_controller.py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import JointState, Imu
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import math
from typing import Dict, List

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Initialize state
        self.joint_positions = {}
        self.imu_data = None
        self.trajectory_active = False
        self.safety_enabled = True

        # Configure QoS profiles
        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        cmd_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', cmd_qos)
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory', cmd_qos)
        self.status_pub = self.create_publisher(String, '/status', cmd_qos)
        self.safety_pub = self.create_publisher(Bool, '/safety_status', cmd_qos)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, sensor_qos)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, sensor_qos)

        # Parameters
        self.declare_parameter('control_frequency', 100)  # Hz
        self.declare_parameter('max_linear_velocity', 0.5)
        self.declare_parameter('max_angular_velocity', 0.3)
        self.declare_parameter('safety_threshold_angle', 0.5)  # radians
        self.declare_parameter('joint_names', [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ])

        # Timers
        self.control_timer = self.create_timer(
            1.0 / self.get_parameter('control_frequency').value,
            self.control_loop
        )

        # Get joint names from parameters
        self.joint_names = self.get_parameter('joint_names').value

        self.get_logger().info('Humanoid Controller initialized')

    def joint_state_callback(self, msg: JointState):
        """Process joint state messages"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def imu_callback(self, msg: Imu):
        """Process IMU messages"""
        self.imu_data = msg

    def control_loop(self):
        """Main control loop"""
        # Check safety conditions
        if self.safety_enabled and self.is_unsafe_condition():
            self.emergency_stop()
            return

        # Publish status
        status_msg = String()
        status_msg.data = f'Joints: {len(self.joint_positions)}, Trajectory: {self.trajectory_active}'
        self.status_pub.publish(status_msg)

        # Publish safety status
        safety_msg = Bool()
        safety_msg.data = not self.is_unsafe_condition()
        self.safety_pub.publish(safety_msg)

    def is_unsafe_condition(self) -> bool:
        """Check if current state is unsafe"""
        if self.imu_data is None:
            return False  # No data to evaluate

        # Check orientation (if tilt is too large)
        roll, pitch = self.get_orientation_from_imu()

        threshold = self.get_parameter('safety_threshold_angle').value
        return abs(roll) > threshold or abs(pitch) > threshold

    def get_orientation_from_imu(self) -> tuple:
        """Extract roll and pitch from IMU quaternion"""
        if self.imu_data is None:
            return 0.0, 0.0

        # Convert quaternion to roll/pitch/yaw
        # Simplified conversion for pitch and roll
        q = self.imu_data.orientation
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(sinp)

        return roll, pitch

    def emergency_stop(self):
        """Emergency stop procedure"""
        # Stop all movement
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        # Send zero trajectory
        if self.joint_trajectory_pub:
            zero_trajectory = self.create_zero_trajectory()
            self.joint_trajectory_pub.publish(zero_trajectory)

        self.get_logger().warn('EMERGENCY STOP ACTIVATED')

    def create_zero_trajectory(self) -> JointTrajectory:
        """Create a trajectory that zeros out all joint velocities"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        # Set positions to current (or desired safe) positions
        for joint_name in self.joint_names:
            current_pos = self.joint_positions.get(joint_name, 0.0)
            point.positions.append(current_pos)
            point.velocities.append(0.0)  # Zero velocity
            point.accelerations.append(0.0)  # Zero acceleration

        point.time_from_start = Duration(sec=0, nanosec=100000000)  # 0.1 seconds
        trajectory.points.append(point)

        return trajectory

def main(args=None):
    rclpy.init(args=args)

    controller = HumanoidController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Humanoid Controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Day 8: TF (Transform) and Coordinate Systems

TF (Transform) is crucial for humanoid robotics to manage coordinate frames:

```python
# my_humanoid_control/tf_manager.py

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
import tf_transformations

class TFManager(Node):
    def __init__(self):
        super().__init__('tf_manager')

        # Initialize transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        # Timer for broadcasting transforms
        self.timer = self.create_timer(0.05, self.broadcast_transforms)  # 20 Hz

        # Store joint positions
        self.joint_positions = {}

    def joint_state_callback(self, msg):
        """Process joint state messages and store positions"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def broadcast_transforms(self):
        """Broadcast coordinate frame transforms"""
        # Example: Broadcast transforms for a simplified humanoid
        transforms = self.calculate_transforms()

        for transform in transforms:
            self.tf_broadcaster.sendTransform(transform)

    def calculate_transforms(self) -> list:
        """Calculate transforms based on joint positions"""
        transforms = []

        # Base link to left leg
        t_left_leg = TransformStamped()
        t_left_leg.header.stamp = self.get_clock().now().to_msg()
        t_left_leg.header.frame_id = 'base_link'
        t_left_leg.child_frame_id = 'left_leg_link'

        # Calculate transform based on joint angles (simplified)
        hip_angle = self.joint_positions.get('left_hip_joint', 0.0)
        t_left_leg.transform.translation.x = 0.0
        t_left_leg.transform.translation.y = -0.1  # Left leg offset
        t_left_leg.transform.translation.z = -0.1  # Hip height offset
        t_left_leg.transform.rotation = self.angle_to_quaternion(hip_angle, 'y')

        transforms.append(t_left_leg)

        # Add more transforms for other joints...

        return transforms

    def angle_to_quaternion(self, angle, axis):
        """Convert rotation angle to quaternion"""
        from geometry_msgs.msg import Quaternion

        if axis == 'x':
            q = tf_transformations.quaternion_from_euler(angle, 0, 0)
        elif axis == 'y':
            q = tf_transformations.quaternion_from_euler(0, angle, 0)
        elif axis == 'z':
            q = tf_transformations.quaternion_from_euler(0, 0, angle)
        else:
            q = [0, 0, 0, 1]  # Identity quaternion

        quat_msg = Quaternion()
        quat_msg.x = q[0]
        quat_msg.y = q[1]
        quat_msg.z = q[2]
        quat_msg.w = q[3]

        return quat_msg

def main(args=None):
    rclpy.init(args=args)

    tf_manager = TFManager()

    try:
        rclpy.spin(tf_manager)
    except KeyboardInterrupt:
        pass
    finally:
        tf_manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Day 9: Parameter Validation and Testing

Add parameter validation and configuration:

```python
# my_humanoid_control/parameter_validator.py

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, IntegerRange, FloatingPointRange
from rcl_interfaces.srv import SetParameters

class ParameterValidator(Node):
    def __init__(self):
        super().__init__('parameter_validator')

        # Declare parameters with validation
        self.declare_parameter_with_validation()

        # Set up parameter validation callback
        self.add_on_set_parameters_callback(self.validate_parameters)

        # Create service to check parameter validity
        self.param_check_srv = self.create_service(
            SetParameters, 'check_parameters', self.check_parameters_callback)

    def declare_parameter_with_validation(self):
        """Declare parameters with specific constraints"""
        # Velocity parameters with ranges
        velocity_desc = ParameterDescriptor(
            description='Maximum linear velocity (m/s)',
            floating_point_range=[FloatingPointRange(from_value=0.0, to_value=2.0)]
        )
        self.declare_parameter('max_linear_velocity', 0.5, descriptor=velocity_desc)

        # Safety angle parameters
        angle_desc = ParameterDescriptor(
            description='Safety threshold angle (radians)',
            floating_point_range=[FloatingPointRange(from_value=0.1, to_value=1.57)]
        )
        self.declare_parameter('safety_threshold_angle', 0.5, descriptor=angle_desc)

        # Joint count parameter
        joint_desc = ParameterDescriptor(
            description='Number of joints to control',
            integer_range=[IntegerRange(from_value=6, to_value=50)]
        )
        self.declare_parameter('joint_count', 12, descriptor=joint_desc)

    def validate_parameters(self, parameters):
        """Validate parameter changes"""
        from rcl_interfaces.msg import SetParametersResult

        result = SetParametersResult()
        result.successful = True

        for param in parameters:
            if param.name == 'max_linear_velocity':
                if not (0.0 <= param.value <= 2.0):
                    result.successful = False
                    result.reason = f'Velocity must be between 0 and 2, got {param.value}'
                    break
            elif param.name == 'safety_threshold_angle':
                if not (0.1 <= param.value <= 1.57):
                    result.successful = False
                    result.reason = f'Safety angle must be between 0.1 and 1.57, got {param.value}'
                    break

        return result

    def check_parameters_callback(self, request, response):
        """Service callback to check if parameters are valid"""
        # This would validate all current parameters
        # For now, just return success
        response.successful = [True] * len(request.parameters)
        response.reasons = ['Valid'] * len(request.parameters)
        return response

def main(args=None):
    rclpy.init(args=args)

    validator = ParameterValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        pass
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Day 10: Comprehensive Package Exercise

Create a complete package with multiple nodes and configuration:

```python
# my_humanoid_control/safety_manager.py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Bool, String
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Duration

class SafetyManager(Node):
    def __init__(self):
        super().__init__('safety_manager')

        # Configuration
        self.declare_parameter('safety_period', 0.05)  # 20 Hz safety checks
        self.declare_parameter('fall_threshold', 0.785)  # ~45 degrees
        self.declare_parameter('collision_threshold', 0.3)  # meters
        self.declare_parameter('velocity_threshold', 1.0)  # m/s

        # State
        self.imu_data = None
        self.joint_states = None
        self.safety_violation = False
        self.emergency_stop_active = False

        # QoS profiles
        sensor_qos = QoSProfile(depth=10)

        # Publishers
        self.safety_pub = self.create_publisher(Bool, '/safety_status', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        self.cmd_stop_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/safety_status', 10)

        # Subscribers
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, sensor_qos)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, sensor_qos)

        # Safety timer
        self.safety_timer = self.create_timer(
            self.get_parameter('safety_period').value,
            self.safety_check
        )

        self.get_logger().info('Safety Manager initialized')

    def imu_callback(self, msg):
        """Handle IMU data for orientation checks"""
        self.imu_data = msg

    def joint_callback(self, msg):
        """Handle joint state data for position checks"""
        self.joint_states = msg

    def safety_check(self):
        """Perform safety checks and react to violations"""
        violations = []

        # Check orientation (fall detection)
        if self.imu_data:
            roll, pitch = self.get_orientation_from_imu()
            threshold = self.get_parameter('fall_threshold').value

            if abs(roll) > threshold or abs(pitch) > threshold:
                violations.append(f'Fall detected: Roll={roll:.2f}, Pitch={pitch:.2f}')

        # Check for safety violations
        if violations:
            self.safety_violation = True
            self.trigger_safety_response(violations)
        else:
            self.safety_violation = False
            self.emergency_stop_active = False

        # Publish safety status
        safety_msg = Bool()
        safety_msg.data = not self.safety_violation
        self.safety_pub.publish(safety_msg)

        status_msg = String()
        status_msg.data = f'Safety: {not self.safety_violation}, Violations: {len(violations)}'
        self.status_pub.publish(status_msg)

    def get_orientation_from_imu(self):
        """Extract roll and pitch from IMU quaternion"""
        if not self.imu_data:
            return 0.0, 0.0

        # Extract roll and pitch from quaternion (simplified)
        q = self.imu_data.orientation
        import math

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(sinp)

        return roll, pitch

    def trigger_safety_response(self, violations):
        """Trigger safety response when violations occur"""
        self.get_logger().warn(f'Safety violation detected: {", ".join(violations)}')

        # Activate emergency stop
        if not self.emergency_stop_active:
            self.emergency_stop_active = True

            # Stop all movement
            stop_cmd = Twist()
            self.cmd_stop_pub.publish(stop_cmd)

            # Publish emergency stop signal
            emergency_msg = Bool()
            emergency_msg.data = True
            self.emergency_stop_pub.publish(emergency_msg)

def main(args=None):
    rclpy.init(args=args)

    safety_manager = SafetyManager()

    try:
        rclpy.spin(safety_manager)
    except KeyboardInterrupt:
        safety_manager.get_logger().info('Safety Manager shutting down')
    finally:
        safety_manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Week 5: Launch Files and Advanced Configuration

### Day 11: Launch Files and Composition

#### Basic Launch Files

Create launch files to start multiple nodes at once:

```python
# launch/humanoid_system.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='humanoid_robot',
        description='Name of the robot'
    )

    # Controller node
    controller_node = Node(
        package='my_humanoid_control',
        executable='humanoid_controller',
        name='humanoid_controller',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('my_humanoid_control'),
                'config',
                'controller_params.yaml'
            ]),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    # Safety manager node
    safety_node = Node(
        package='my_humanoid_control',
        executable='safety_manager',
        name='safety_manager',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('my_humanoid_control'),
                'config',
                'safety_params.yaml'
            ]),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    # TF manager node
    tf_node = Node(
        package='my_humanoid_control',
        executable='tf_manager',
        name='tf_manager',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    # Return the launch description
    return LaunchDescription([
        declare_use_sim_time,
        declare_robot_name,
        controller_node,
        safety_node,
        tf_node,
    ])
```

#### Advanced Launch Configuration

Create more complex launch files with conditional logic:

```python
# launch/humanoid_with_gazebo.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    robot_name = LaunchConfiguration('robot_name')

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_use_rviz_cmd = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to start RViz'
    )

    declare_robot_name_cmd = DeclareLaunchArgument(
        'robot_name',
        default_value='humanoid_robot',
        description='Name of the robot to spawn'
    )

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', robot_name,
            '-x', '0.0',
            '-y', '0.0',
            '-z', '1.0'
        ],
        output='screen',
    )

    # Humanoid controller
    controller_node = Node(
        package='my_humanoid_control',
        executable='humanoid_controller',
        name='humanoid_controller',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('my_humanoid_control'),
                'config',
                'controller_params.yaml'
            ]),
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # RViz node (conditional)
    rviz_node = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d', PathJoinSubstitution([
                FindPackageShare('my_humanoid_control'),
                'rviz',
                'humanoid_config.rviz'
            ])
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_robot_name_cmd)

    # Add nodes and launch files
    ld.add_action(gazebo)
    ld.add_action(robot_state_publisher)
    ld.add_action(spawn_entity)
    ld.add_action(controller_node)
    ld.add_action(rviz_node)

    return ld
```

### Day 12: Parameter Configuration Files

Create comprehensive parameter configuration files:

```yaml
# config/controller_params.yaml
humanoid_controller:
  ros__parameters:
    control_frequency: 100
    max_linear_velocity: 0.5
    max_angular_velocity: 0.3
    safety_threshold_angle: 0.5
    joint_names:
      - "left_hip_joint"
      - "left_knee_joint"
      - "left_ankle_joint"
      - "right_hip_joint"
      - "right_knee_joint"
      - "right_ankle_joint"
      - "left_shoulder_joint"
      - "left_elbow_joint"
      - "right_shoulder_joint"
      - "right_elbow_joint"
    pid_gains:
      linear:
        kp: 1.0
        ki: 0.1
        kd: 0.05
      angular:
        kp: 2.0
        ki: 0.2
        kd: 0.1
    trajectory_config:
      max_velocity: 0.8
      max_acceleration: 1.0
      min_trajectory_duration: 0.1
```

```yaml
# config/safety_params.yaml
safety_manager:
  ros__parameters:
    safety_period: 0.05
    fall_threshold: 0.785  # 45 degrees
    collision_threshold: 0.3
    velocity_threshold: 1.0
    joint_limit_threshold: 0.1
    emergency_stop_duration: 5.0
    recovery_attempts: 3
    recovery_timeout: 10.0
```

### Day 13: System Integration and Testing

Create integration tests and launch configurations:

```python
# test/test_integration.py

import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class IntegrationTestNode(Node):
    def __init__(self):
        super().__init__('integration_test_node')

        # Create subscribers to verify system behavior
        self.status_sub = self.create_subscription(
            String, '/status', self.status_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

        # Store test data
        self.status_messages = []
        self.joint_messages = []
        self.test_passed = False

    def status_callback(self, msg):
        self.status_messages.append(msg.data)

    def joint_callback(self, msg):
        self.joint_messages.append(msg)

class TestHumanoidSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_system_integration(self):
        """Test that all system components are communicating properly"""
        node = IntegrationTestNode()
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        # Run the test for a few seconds to collect messages
        start_time = node.get_clock().now()
        timeout = node.get_clock().now().nanoseconds + 5 * 1000000000  # 5 seconds

        while node.get_clock().now().nanoseconds < timeout:
            executor.spin_once(timeout_sec=0.1)

            # Check if we have received messages
            if len(node.status_messages) > 0 and len(node.joint_messages) > 0:
                # If we're receiving both status and joint messages, system is integrated
                node.test_passed = True
                break

        # Verify that messages were received
        self.assertTrue(node.test_passed,
                       "System integration test failed - no communication detected")
        self.assertGreater(len(node.status_messages), 0,
                         "No status messages received")
        self.assertGreater(len(node.joint_messages), 0,
                         "No joint state messages received")

        node.destroy_node()

if __name__ == '__main__':
    unittest.main()
```

### Day 14: Performance Monitoring and Debugging

Create tools for system monitoring:

```python
# my_humanoid_control/system_monitor.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from rclpy.qos import QoSProfile
import time
from collections import deque

class SystemMonitor(Node):
    def __init__(self):
        super().__init__('system_monitor')

        # Configuration
        self.declare_parameter('monitor_period', 1.0)  # seconds
        self.declare_parameter('message_history_size', 100)

        # Publishers
        self.status_pub = self.create_publisher(String, '/system_status', 10)
        self.cpu_usage_pub = self.create_publisher(Float32, '/cpu_usage', 10)

        # Message tracking
        self.message_counts = {}
        self.message_times = {}

        # Monitor timer
        self.monitor_timer = self.create_timer(
            self.get_parameter('monitor_period').value,
            self.monitor_callback
        )

        # Message history
        self.message_history = deque(
            maxlen=self.get_parameter('message_history_size').value
        )

        self.get_logger().info('System Monitor initialized')

    def monitor_callback(self):
        """Monitor system performance and publish status"""
        # Calculate message rates
        message_rates = self.calculate_message_rates()

        # Create status message
        status_msg = String()
        status_msg.data = f"Messages/sec: {message_rates}, Status: NOMINAL"
        self.status_pub.publish(status_msg)

        # Log performance
        self.get_logger().info(f"System status: {status_msg.data}")

    def calculate_message_rates(self):
        """Calculate message rates for monitored topics"""
        # This is a simplified version
        # In practice, you'd monitor actual message rates
        rates = {}

        # Simulate some rates
        rates['/joint_states'] = 100  # Hz
        rates['/imu/data'] = 200     # Hz
        rates['/cmd_vel'] = 50       # Hz

        return rates

def main(args=None):
    rclpy.init(args=args)

    monitor = SystemMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        monitor.get_logger().info('System Monitor shutting down')
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Day 15: Final Assessment and Review

#### Week 5 Exercise: Complete System Launch

Create a complete system launch that integrates all components:

```python
# launch/complete_humanoid_system.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import TimerAction

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    log_level = LaunchConfiguration('log_level')

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_log_level_cmd = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level for nodes'
    )

    # Robot controller
    controller_node = Node(
        package='my_humanoid_control',
        executable='humanoid_controller',
        name='humanoid_controller',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('my_humanoid_control'),
                'config',
                'controller_params.yaml'
            ]),
            {'use_sim_time': use_sim_time}
        ],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen'
    )

    # Safety manager
    safety_node = Node(
        package='my_humanoid_control',
        executable='safety_manager',
        name='safety_manager',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('my_humanoid_control'),
                'config',
                'safety_params.yaml'
            ]),
            {'use_sim_time': use_sim_time}
        ],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen'
    )

    # TF manager
    tf_node = Node(
        package='my_humanoid_control',
        executable='tf_manager',
        name='tf_manager',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen'
    )

    # System monitor
    monitor_node = Node(
        package='my_humanoid_control',
        executable='system_monitor',
        name='system_monitor',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen'
    )

    # Add startup order dependencies using event handlers
    delayed_safety = RegisterEventHandler(
        OnProcessStart(
            target_action=controller_node,
            on_start=[
                TimerAction(
                    period=2.0,
                    actions=[safety_node]
                )
            ]
        )
    )

    delayed_tf = RegisterEventHandler(
        OnProcessStart(
            target_action=controller_node,
            on_start=[
                TimerAction(
                    period=1.0,
                    actions=[tf_node]
                )
            ]
        )
    )

    delayed_monitor = RegisterEventHandler(
        OnProcessStart(
            target_action=safety_node,
            on_start=[
                TimerAction(
                    period=0.5,
                    actions=[monitor_node]
                )
            ]
        )
    )

    # Return the launch description
    return LaunchDescription([
        declare_use_sim_time_cmd,
        declare_log_level_cmd,
        controller_node,
        delayed_safety,
        delayed_tf,
        delayed_monitor,
    ])
```

## Assessment and Learning Verification

### Week 3 Assessment

1. **Conceptual Understanding**: Explain the difference between ROS 1 and ROS 2 architecture
2. **Implementation**: Create a publisher-subscriber pair for exchanging sensor data
3. **Analysis**: Compare different QoS policies and their appropriate use cases

### Week 4 Assessment

1. **Technical Skills**: Create a complete ROS 2 package with multiple nodes
2. **Problem Solving**: Implement a safety manager node that monitors system state
3. **Integration**: Design and implement TF transforms for a humanoid robot

### Week 5 Assessment

1. **System Integration**: Create a launch file that starts a complete humanoid control system
2. **Configuration**: Implement parameter validation and management system
3. **Testing**: Write integration tests for the complete system

## Resources and Further Reading

### Required Reading
- "Programming Robots with ROS" by Morgan Quigley, Brian Gerkey, and William Smart
- "Effective Robotics Programming with ROS" by Anil Mahtani, Alejandro Hernandez Cordero, and Luis Sánchez Crespo
- ROS 2 Documentation: https://docs.ros.org/

### Tutorials
- ROS 2 Tutorials: https://docs.ros.org/en/humble/Tutorials.html
- Navigation 2: https://navigation.ros.org/
- Robot Framework Migration Guide: https://docs.ros.org/en/humble/The-ROS2-Project/Migration-Guide.html

### Tools
- RViz: Robot visualization
- rqt: ROS GUI tools
- ros2 bag: Data recording and playback
- ros2 run: Node execution
- ros2 launch: System launch

## Next Steps

After completing Weeks 3-5, you now have a solid foundation in ROS 2 fundamentals specifically relevant to humanoid robotics. In the next module (Weeks 6-7), we'll dive into robot simulation with Gazebo, building on the ROS 2 knowledge you've gained to create realistic simulation environments for your humanoid robots.

The next module will cover:
- Gazebo simulation environment setup
- Physics simulation and collision detection
- Sensor simulation for humanoid robots
- Integration with ROS 2 control systems

Continue to the [Weeks 6-7: Robot Simulation with Gazebo](/docs/weekly-breakdown/weeks-6-7-gazebo-simulation) module to apply your ROS 2 knowledge to realistic simulation environments.