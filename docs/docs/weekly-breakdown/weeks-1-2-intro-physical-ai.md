---
title: "Weeks 1-2: Introduction to Physical AI"
description: "Foundations of Physical AI and embodied intelligence, from digital AI to robots that understand physical laws"
keywords: ["physical ai", "embodied intelligence", "foundations", "robotics", "ai"]
sidebar_position: 1
---

# Weeks 1-2: Introduction to Physical AI

Welcome to the first module of the Physical AI & Humanoid Robotics course! These initial weeks establish the foundational concepts of Physical AI and embodied intelligence, exploring how AI systems can function in the physical world and comprehend physical laws.

## Learning Objectives

By the end of these two weeks, you will be able to:
- Define Physical AI and embodied intelligence
- Understand the transition from digital AI to physical AI
- Identify key challenges in bridging digital and physical domains
- Recognize the importance of humanoid form in human-centered environments
- Describe fundamental sensor systems used in robotics
- Explain the relationship between AI models and physical embodiment

## Prerequisites

- Basic understanding of AI and machine learning concepts
- Fundamental knowledge of physics (Newtonian mechanics)
- Programming experience (Python preferred)
- Familiarity with Linux command line (Ubuntu 22.04)

## Week 1: Foundations of Physical AI

### Day 1: Introduction to Physical AI

#### What is Physical AI?

Physical AI represents a paradigm shift from AI models confined to digital environments to embodied intelligence that operates in physical space. Unlike traditional AI that processes data in virtual environments, Physical AI systems must:

- **Perceive physical reality**: Use sensors to understand the three-dimensional world
- **Interact with physical objects**: Apply forces, manipulate objects, navigate spaces
- **Comprehend physical laws**: Understand gravity, friction, momentum, and material properties
- **Adapt to physical constraints**: Work within the limitations of real-world physics

#### Embodied Intelligence

Embodied intelligence is the concept that intelligence emerges from the interaction between an agent and its physical environment. Key principles include:

- **Embodiment**: The physical form influences cognitive processes
- **Environment interaction**: Intelligence is shaped by environmental feedback
- **Morphological computation**: Physical properties contribute to intelligent behavior
- **Situatedness**: Intelligence is context-dependent and situation-specific

#### The Digital vs. Physical Divide

| Digital AI | Physical AI |
|------------|-------------|
| Processes virtual data | Interacts with real objects |
| No physical constraints | Subject to physical laws |
| Perfect information | Noisy, incomplete sensor data |
| Instantaneous operations | Time-delayed physical actions |
| Unlimited parallelism | Resource-limited actuators |

### Day 2: The Transition from Digital to Physical

#### Challenges in the Digital-to-Physical Transition

1. **Reality Gap**: The difference between simulated and real-world performance
2. **Sensor Noise**: Real sensors provide imperfect, noisy data
3. **Actuator Limitations**: Physical actuators have limited precision and speed
4. **Latency Issues**: Physical actions take time, creating temporal delays
5. **Safety Constraints**: Physical systems must operate safely in human environments

#### Simulation-to-Reality Transfer (Sim-to-Real)

Sim-to-Real transfer techniques help bridge the reality gap:

- **Domain Randomization**: Training in varied simulated environments
- **System Identification**: Modeling real-world dynamics
- **Adaptive Control**: Adjusting behavior based on real-world feedback
- **Fine-tuning**: Adjusting simulation parameters to match reality

#### The Role of Humanoid Form

Humanoid robots are particularly well-suited for human-centered environments because they:

- **Share our physical form**: Can navigate human-designed spaces
- **Enable natural interaction**: Humans are adapted to interact with human-like forms
- **Leverage human knowledge**: Can utilize environments designed for humans
- **Provide abundant training data**: Human behavior provides rich training examples

### Day 3: Physical Laws and Robot Dynamics

#### Newtonian Mechanics in Robotics

Robot dynamics are governed by fundamental physical laws:

**Newton's First Law (Inertia)**
- A robot at rest stays at rest unless acted upon by an external force
- A robot in motion continues in motion unless acted upon by an external force

**Newton's Second Law (F = ma)**
- Force equals mass times acceleration
- Essential for understanding robot motion and control

**Newton's Third Law (Action-Reaction)**
- For every action, there is an equal and opposite reaction
- Critical for understanding contact forces and locomotion

#### Key Physical Concepts for Robotics

**Center of Mass (CoM)**
- The point where the robot's mass is concentrated
- Critical for balance and stability
- Calculated as: CoM = Σ(mᵢ × rᵢ) / Σmᵢ

**Moment of Inertia**
- Resistance to rotational motion
- Affects how robots rotate and balance
- Depends on mass distribution

**Support Polygon**
- The area where ground reaction forces can be applied
- For stable standing, CoM must remain within the support polygon

**Zero Moment Point (ZMP)**
- Point where the net moment of ground reaction forces is zero
- Critical for dynamic balance in walking robots

### Day 4: Sensor Systems Overview

#### Types of Sensors in Robotics

**Proprioceptive Sensors** (Internal)
- Joint encoders: Measure joint angles
- Inertial Measurement Units (IMUs): Measure acceleration and rotation
- Force/torque sensors: Measure forces at contact points
- Motor current sensors: Indicate load and effort

**Exteroceptive Sensors** (External Environment)
- Cameras: Visual information
- LiDAR: Distance measurements
- Depth sensors: 3D scene information
- Tactile sensors: Contact and pressure information

#### Sensor Characteristics

**Accuracy vs. Precision**
- **Accuracy**: How close measurements are to true values
- **Precision**: How consistent repeated measurements are

**Noise and Resolution**
- **Noise**: Random variations in sensor readings
- **Resolution**: Smallest detectable change in measurement

**Bandwidth and Latency**
- **Bandwidth**: Frequency range of reliable measurements
- **Latency**: Time delay between event and measurement

#### Common Sensor Systems

**Inertial Measurement Unit (IMU)**
- **Components**: Accelerometer, gyroscope, magnetometer
- **Purpose**: Measure orientation, angular velocity, linear acceleration
- **Applications**: Balance, navigation, motion tracking
- **Limitations**: Drift over time, sensitive to vibration

**LiDAR (Light Detection and Ranging)**
- **Principle**: Measures distance using laser pulses
- **Resolution**: High precision distance measurements
- **Applications**: Mapping, obstacle detection, navigation
- **Limitations**: Expensive, sensitive to weather, limited in close proximity

**RGB-D Cameras**
- **Components**: Color camera + depth sensor
- **Purpose**: Visual information + 3D scene understanding
- **Applications**: Object recognition, scene understanding, navigation
- **Limitations**: Affected by lighting, limited range

### Day 5: Hands-on Introduction

#### Setting Up the Development Environment

Before diving into Physical AI concepts, we need to establish the development environment:

1. **Ubuntu 22.04 LTS**: The recommended operating system
2. **ROS 2 Humble Hawksbill**: The Robot Operating System
3. **Python 3.10+**: Primary programming language
4. **Development tools**: Git, VS Code, etc.

#### First Physical AI Exercise: Robot State Monitoring

Create a simple system to monitor and visualize robot state:

```python
#!/usr/bin/env python3
# robot_state_monitor.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
import numpy as np

class RobotStateMonitor(Node):
    def __init__(self):
        super().__init__('robot_state_monitor')

        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        # Publisher for center of mass estimate
        self.com_pub = self.create_publisher(Float32, '/center_of_mass', 10)

        # Timer for periodic updates
        self.timer = self.create_timer(0.1, self.update_callback)

        # Robot state storage
        self.joint_positions = {}
        self.joint_velocities = {}

        self.get_logger().info("Robot State Monitor initialized")

    def joint_callback(self, msg):
        """Callback for joint state messages"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]

    def calculate_center_of_mass(self):
        """Simple center of mass calculation"""
        # This is a simplified example
        # In practice, you'd need robot URDF and mass properties
        if 'left_leg_joint' in self.joint_positions and 'right_leg_joint' in self.joint_positions:
            left_pos = self.joint_positions['left_leg_joint']
            right_pos = self.joint_positions['right_leg_joint']

            # Simplified CoM estimate
            com_estimate = (left_pos + right_pos) / 2.0
            return com_estimate
        return 0.0

    def update_callback(self):
        """Periodic update callback"""
        com = self.calculate_center_of_mass()

        # Publish CoM estimate
        com_msg = Float32()
        com_msg.data = com
        self.com_pub.publish(com_msg)

        # Log state
        self.get_logger().info(f"CoM estimate: {com:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = RobotStateMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Week 2: Humanoid Robotics Landscape

### Day 6: Overview of Humanoid Robotics

#### Historical Development

Humanoid robotics has evolved through several generations:

**First Generation (1970s-1990s)**
- Early walking machines
- Limited mobility and functionality
- Primarily research platforms

**Second Generation (2000s-2010s)**
- Improved balance and locomotion
- More sophisticated control systems
- Commercial applications (ASIMO, QRIO)

**Third Generation (2010s-Present)**
- Advanced AI integration
- Natural human interaction
- Practical applications (Pepper, NAO, Atlas)

#### Current State of Humanoid Robotics

**Research Platforms**
- Boston Dynamics Atlas: Advanced mobility and manipulation
- Honda ASIMO: Human interaction and navigation
- Toyota HSR: Service robotics applications

**Commercial Platforms**
- SoftBank Pepper: Customer service and entertainment
- Aldebaran NAO: Education and research
- UBTECH Jimu: Consumer robotics

**Emerging Platforms**
- Agility Robotics Digit: Logistics and delivery
- Figure AI: General-purpose humanoid robots
- Sanctuary AI: Industrial applications

#### Key Challenges in Humanoid Robotics

1. **Balance and Locomotion**
   - Maintaining stability during movement
   - Adapting to uneven terrain
   - Handling external disturbances

2. **Manipulation and Dexterity**
   - Fine motor control
   - Object recognition and grasping
   - Tool use and manipulation

3. **Human-Robot Interaction**
   - Natural communication
   - Social behavior
   - Trust and acceptance

4. **Integration and Autonomy**
   - Sensor fusion
   - Real-time decision making
   - Task planning and execution

### Day 7: Sensor Systems in Depth

#### LIDAR Systems

**Principle of Operation**
LiDAR systems work by emitting laser pulses and measuring the time it takes for the light to return after reflecting off objects:

```
Distance = (Speed of Light × Time of Flight) / 2
```

**Types of LiDAR**
- **Mechanical**: Rotating mirrors/lasers, 360° coverage
- **Solid State**: No moving parts, more reliable
- **Flash**: Illuminates entire scene at once

**Applications in Humanoid Robotics**
- Environment mapping
- Obstacle detection
- Navigation and path planning
- Human detection and tracking

#### Depth Cameras

**Time-of-Flight (ToF) Cameras**
- Measure distance using light travel time
- Good for medium distances
- Sensitive to ambient light

**Structured Light Cameras**
- Project known light patterns
- Analyze deformation to calculate depth
- Good accuracy at close range

**Stereo Vision**
- Use two cameras to calculate depth via triangulation
- No special hardware required
- Computationally intensive

#### IMU Systems for Humanoid Balance

**Accelerometer**
- Measures linear acceleration
- Can detect gravity vector for orientation
- Sensitive to motion and vibration

**Gyroscope**
- Measures angular velocity
- Good for detecting rotation
- Drifts over time

**Magnetometer**
- Measures magnetic field
- Provides absolute orientation reference
- Susceptible to magnetic interference

### Day 8: Force and Torque Sensing

#### Importance of Force Sensing in Humanoids

Force and torque sensing is crucial for:
- **Balance control**: Detecting ground reaction forces
- **Safe interaction**: Limiting forces during contact
- **Grasping**: Controlling grip strength
- **Locomotion**: Managing foot-ground contact

#### Force/Torque Sensors

**Six-Axis Force/Torque Sensors**
- Measure 3 forces (X, Y, Z) and 3 torques (roll, pitch, yaw)
- Used at joints and end-effectors
- Enable precise force control

**Tactile Sensors**
- Distributed pressure sensing
- Provide rich contact information
- Enable dexterous manipulation

**Foot Pressure Sensors**
- Detect center of pressure
- Monitor balance during walking
- Provide feedback for gait control

#### Force Control Principles

**Impedance Control**
- Control the relationship between force and position
- Creates virtual springs and dampers
- Allows compliant interaction

**Admittance Control**
- Control motion in response to applied forces
- Creates virtual mass-spring-damper systems
- Useful for contact-rich tasks

### Day 9: Kinematics and Degrees of Freedom

#### Humanoid Robot Kinematics

**Degrees of Freedom (DOF)**
- Number of independent parameters defining position
- More DOF = greater flexibility but complexity
- Humanoid robots typically have 20-50+ DOF

**Human Comparison**
- Human body: ~700 DOF (simplified to ~200 for control)
- Human arm: ~7 DOF
- Human hand: ~22 DOF

**Typical Humanoid Configuration**
- **Head**: 3 DOF (pitch, yaw, roll)
- **Arms**: 7 DOF each (shoulder: 3, elbow: 1, wrist: 3)
- **Torso**: 3-6 DOF
- **Legs**: 6-7 DOF each
- **Feet**: 2-6 DOF each

#### Forward and Inverse Kinematics

**Forward Kinematics**
- Calculate end-effector position from joint angles
- Deterministic, always has solution
- Used for position verification

**Inverse Kinematics**
- Calculate joint angles from desired end-effector position
- May have multiple solutions or no solution
- Critical for motion planning

#### Kinematic Constraints

**Joint Limits**
- Physical limitations on joint angles
- Critical for safety and hardware protection
- Must be considered in motion planning

**Singularity Avoidance**
- Configurations where robot loses DOF
- Can cause control problems
- Must be avoided in planning

### Day 10: Practical Exercise - Sensor Integration

#### Exercise: Multi-Sensor Fusion for State Estimation

Create a system that combines multiple sensors to estimate robot state:

```python
#!/usr/bin/env python3
# sensor_fusion_demo.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32
import numpy as np
from scipy.spatial.transform import Rotation as R

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

        # Publishers
        self.orientation_pub = self.create_publisher(Vector3, '/estimated_orientation', 10)
        self.balance_pub = self.create_publisher(Float32, '/balance_metric', 10)

        # State variables
        self.imu_orientation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.joint_positions = {}

        # Fusion parameters
        self.imu_weight = 0.7  # Weight for IMU in fusion
        self.kinematic_weight = 0.3  # Weight for kinematic estimate

        self.get_logger().info("Sensor Fusion Node initialized")

    def imu_callback(self, msg):
        """Handle IMU data"""
        # Convert quaternion to Euler angles
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        rotation = R.from_quat(quat)
        euler = rotation.as_euler('xyz', degrees=False)

        self.imu_orientation = np.array(euler)

    def joint_callback(self, msg):
        """Handle joint state data"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def estimate_balance(self):
        """Estimate balance based on sensor fusion"""
        # Simple balance metric based on orientation
        # In practice, this would use more sophisticated algorithms

        # Calculate roll and pitch angles
        roll, pitch, yaw = self.imu_orientation

        # Balance metric: 0 = perfectly balanced, higher = less balanced
        balance_metric = abs(roll) + abs(pitch)

        return min(balance_metric, 1.0)  # Clamp to [0, 1]

    def publish_state(self):
        """Publish fused state estimates"""
        # Publish orientation estimate
        orientation_msg = Vector3()
        orientation_msg.x = float(self.imu_orientation[0])  # roll
        orientation_msg.y = float(self.imu_orientation[1])  # pitch
        orientation_msg.z = float(self.imu_orientation[2])  # yaw
        self.orientation_pub.publish(orientation_msg)

        # Publish balance metric
        balance_msg = Float32()
        balance_msg.data = self.estimate_balance()
        self.balance_pub.publish(balance_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()

    # Timer for publishing state
    node.create_timer(0.05, node.publish_state)  # 20 Hz

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Assessment and Learning Verification

### Week 1 Assessment

1. **Conceptual Understanding**: Explain the difference between digital AI and Physical AI
2. **Application**: Describe three challenges in transferring AI from simulation to reality
3. **Analysis**: Why is the humanoid form advantageous for human-centered environments?

### Week 2 Assessment

1. **Technical Skills**: Implement a basic sensor fusion algorithm combining IMU and joint data
2. **Problem Solving**: Design a simple balance controller using sensor feedback
3. **Integration**: Explain how multiple sensor types work together in a humanoid robot

## Resources and Further Reading

### Required Reading
- "Introduction to Robotics: Mechanics and Control" by John J. Craig
- "Robotics: Control, Sensing, Vision, and Intelligence" by Fu, Gonzalez, and Lee
- "Handbook of Robotics" edited by Siciliano and Khatib

### Recommended Resources
- ROS 2 documentation: https://docs.ros.org/
- OpenAI Gym Robotics: https://robotics.farama.org/
- NVIDIA Isaac Sim documentation: https://docs.omniverse.nvidia.com/

### Simulation Environment
- Gazebo: http://gazebosim.org/
- PyBullet: https://pybullet.org/
- Webots: https://cyberbotics.com/

## Next Steps

After completing Weeks 1-2, you will have established a solid foundation in Physical AI concepts. In the next module (Weeks 3-5), we'll dive deep into ROS 2 fundamentals, building the software infrastructure needed to control and operate humanoid robots.

The next module will cover:
- ROS 2 architecture and core concepts
- Building ROS 2 packages with Python
- Launch files and parameter management
- TF (Transform) for coordinate system management

Continue to the [Weeks 3-5: ROS 2 Fundamentals](/docs/weekly-breakdown/weeks-3-5-ros2-fundamentals) module to build on these foundational concepts.