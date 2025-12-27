---
title: "Gazebo Simulation Setup"
description: "Setting up Gazebo for humanoid robot simulation and physics-based environments"
keywords: ["gazebo", "simulation", "physics", "robotics", "humanoid", "ros2"]
sidebar_position: 2
---

# Gazebo Simulation Setup

Gazebo is a physics-based simulation environment that provides realistic rendering, physics simulation, and sensor simulation capabilities. This module covers setting up Gazebo for humanoid robot simulation.

## Learning Objectives

By the end of this module, you will be able to:
- Install and configure Gazebo for humanoid robotics
- Set up a simulation environment with realistic physics
- Integrate Gazebo with ROS 2 using gazebo_ros_pkgs
- Launch humanoid robots in simulation
- Configure sensors for realistic perception

## Prerequisites

- ROS 2 fundamentals
- URDF knowledge
- Basic understanding of physics simulation

## Gazebo Installation

### System Requirements
- Ubuntu 22.04 LTS (recommended)
- GPU with OpenGL 3.3+ support
- Minimum 8GB RAM (16GB+ recommended)
- Multi-core processor

### Installation Steps

```bash
# Install Gazebo Garden (recommended for ROS 2 Humble)
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control ros-humble-gazebo-plugins

# Install Gazebo classic if needed
sudo apt install ros-humble-gazebo-dev ros-humble-gazebo-plugins ros-humble-gazebo-ros

# Verify installation
gz sim --version
```

## Basic Gazebo Integration with ROS 2

### Launching Gazebo with ROS 2 Bridge

```xml
<!-- launch/gazebo_simulation.launch.py -->
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():
    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('my_robot_gazebo'),
                'worlds',
                'basic.world'
            ])
        }.items()
    )

    return LaunchDescription([
        gazebo,
    ])
```

### Spawning a Robot in Gazebo

```python
# Python script to spawn a robot
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity

class SpawnRobot(Node):
    def __init__(self):
        super().__init__('spawn_robot')
        self.cli = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def spawn(self, robot_name, robot_xml, initial_pose):
        req = SpawnEntity.Request()
        req.name = robot_name
        req.xml = robot_xml
        req.initial_pose = initial_pose
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()
```

## Physics Configuration

### World File Structure

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_world">
    <!-- Physics engine -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Include ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add your humanoid robot here -->
    <model name="humanoid_robot">
      <!-- Robot definition -->
    </model>
  </world>
</sdf>
```

### Physics Parameters for Humanoid Locomotion

For stable humanoid simulation, use these parameters:

```xml
<physics name="humanoid_physics" type="ode">
  <max_step_size>0.001</max_step_size>  <!-- Small steps for stability -->
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>
  <gravity>0 0 -9.80665</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>100</iters>  <!-- More iterations for stability -->
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.000001</cfm>  <!-- Constraint force mixing -->
      <erp>0.2</erp>      <!-- Error reduction parameter -->
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Sensor Simulation

### Camera Sensors
```xml
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <camera name="head">
      <horizontal_fov>1.089</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
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

### IMU Sensors
```xml
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
    <topic>__default_topic__</topic>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <frame_name>imu_link</frame_name>
      <topic_name>imu/data</topic_name>
      <body_name>imu_link</body_name>
    </plugin>
  </sensor>
</gazebo>
```

### Force/Torque Sensors
```xml
<gazebo>
  <joint name="left_foot_joint">
    <sensor name="left_foot_force_torque" type="force_torque">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <force_torque>
        <frame>sensor</frame>
        <measure_direction>child_to_parent</measure_direction>
      </force_torque>
    </sensor>
  </joint>
</gazebo>
```

## Humanoid-Specific Simulation Considerations

### Ground Contact
For stable bipedal locomotion, ensure proper ground contact:
- Use appropriate friction coefficients (typically 0.8-1.0 for rubber feet)
- Set proper collision geometry for feet
- Use appropriate damping and stiffness parameters

### Balance and Stability
- Lower simulation time step for more stable physics
- Tune PID controllers for joint position/effort control
- Consider using Gazebo's built-in balance controller plugins

### Real-time Performance
- Reduce visual complexity for faster simulation
- Use simpler collision meshes
- Limit update rates for sensors that don't need high frequency

## Launching Humanoid Simulation

### Complete Launch File Example

```python
# launch/humanoid_simulation.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')

    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='humanoid_robot',
        description='Name of the robot to spawn'
    )

    # Launch Gazebo
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

    # Spawn robot
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

    return LaunchDescription([
        declare_use_sim_time,
        declare_robot_name,
        gazebo,
        robot_state_publisher,
        spawn_entity,
    ])
```

## Troubleshooting Common Issues

### Robot Falling Through Ground
- Check collision geometry definition
- Verify physics parameters
- Ensure proper mass and inertial properties

### Joint Control Issues
- Verify transmission definitions in URDF
- Check controller configurations
- Validate joint limits and ranges

### Performance Problems
- Reduce visual complexity
- Lower update rates where possible
- Simplify collision meshes

## Best Practices

### Model Optimization
- Use simplified collision geometry for performance
- Include realistic mass and inertial properties
- Test simulation stability with your specific robot

### Controller Integration
- Use ros2_control for proper hardware interface
- Implement appropriate safety limits
- Include proper error handling

### Simulation Validation
- Compare simulation behavior with theoretical models
- Validate sensor outputs for realism
- Test edge cases and failure scenarios

## Next Steps

After setting up Gazebo simulation, explore [Physics and Collision Simulation](/docs/modules/gazebo-unity/physics-collision) to learn advanced physics modeling and collision detection techniques for humanoid robots.