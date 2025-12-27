---
title: "ROS 2 Nodes, Topics, and Services"
description: "Deep dive into ROS 2 communication mechanisms: nodes, topics, and services"
keywords: ["ros2", "nodes", "topics", "services", "communication", "middleware"]
sidebar_position: 3
---

# ROS 2 Nodes, Topics, and Services

This module covers the fundamental communication mechanisms in ROS 2: nodes for encapsulating functionality, topics for asynchronous message passing, and services for synchronous request/response communication.

## Learning Objectives

By the end of this module, you will be able to:
- Create and configure ROS 2 nodes with proper lifecycle management
- Implement publisher-subscriber communication using topics
- Design and implement request-response communication using services
- Apply quality of service (QoS) policies for reliable communication
- Use ROS 2 tools for debugging and monitoring communication

## Prerequisites

- ROS 2 Fundamentals knowledge
- Python programming experience
- Basic understanding of robotics systems

## Nodes in ROS 2

Nodes are the fundamental building blocks of ROS 2 applications. They encapsulate functionality and provide an execution context for your code.

### Node Lifecycle

ROS 2 nodes follow a lifecycle that includes:
- Unconfigured → Inactive → Active → Finalized

```python
import rclpy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn

class LifecycleNodeExample(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_node_example')

    def on_configure(self, state):
        self.get_logger().info('Configuring node...')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating node...')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating node...')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info('Cleaning up node...')
        return TransitionCallbackReturn.SUCCESS
```

## Topics - Publisher/Subscriber Pattern

Topics enable asynchronous communication between nodes using a publish/subscribe pattern.

### Creating a Publisher

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1
```

### Creating a Subscriber

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')
```

### Quality of Service (QoS) Settings

QoS policies control the behavior of publishers and subscribers:

```python
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

# Example: Reliable communication with keep-all history
qos_profile = QoSProfile(
    history=QoSHistoryPolicy.KEEP_ALL,
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE
)

publisher = self.create_publisher(String, 'topic_name', qos_profile)
```

## Services - Request/Response Pattern

Services provide synchronous communication with request/response semantics.

### Creating a Service Server

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {response.sum}')
        return response
```

### Creating a Service Client

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Best Practices for Humanoid Robotics

### Naming Conventions
- Use descriptive, consistent names for topics and services
- Follow the convention: `/robot_name/module_name/message_type`
- Example: `/humanoid_robot/leg_controller/joint_states`

### Error Handling
- Always check if services are available before calling them
- Implement timeout mechanisms for service calls
- Handle connection failures gracefully

### Performance Considerations
- Choose appropriate QoS settings based on your application needs
- Use appropriate message types for your data
- Consider message frequency to avoid network congestion

## ROS 2 Communication Tools

### Command Line Tools
- `ros2 topic list` - List all topics
- `ros2 topic echo <topic_name>` - Print messages from a topic
- `ros2 service list` - List all services
- `ros2 node list` - List all active nodes
- `ros2 run <package> <executable>` - Run a node

### Monitoring Communication
```bash
# Monitor topic traffic
ros2 topic hz /chatter

# Monitor message contents
ros2 topic echo /chatter

# Call a service from command line
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"
```

## Integration with Humanoid Systems

For humanoid robotics, communication patterns typically include:
- Joint state publishers for each limb
- Sensor data streams (IMU, force/torque, camera feeds)
- Control command services for motion planning
- TF (Transform) broadcasters for coordinate frames

## Next Steps

After mastering nodes, topics, and services, explore [URDF for Humanoids](/docs/modules/ros2/urdf-humanoids) to learn how to describe robot structure and kinematics in ROS 2.