---
title: "Sensor Simulation"
description: "Simulating realistic sensors for humanoid robots in Gazebo: LiDAR, cameras, IMUs, and more"
keywords: ["sensors", "gazebo", "simulation", "lidar", "camera", "imu", "humanoid", "robotics"]
sidebar_position: 4
---

# Sensor Simulation

This module covers simulating realistic sensors for humanoid robots in Gazebo, including LiDAR, cameras, IMUs, and other sensors essential for perception and control in physical AI applications.

## Learning Objectives

By the end of this module, you will be able to:
- Configure and simulate various sensor types in Gazebo
- Set up realistic sensor parameters for humanoid robotics
- Integrate sensors with ROS 2 message formats
- Calibrate and validate sensor outputs
- Handle sensor fusion and data processing

## Prerequisites

- Gazebo simulation setup knowledge
- ROS 2 integration with Gazebo
- Basic understanding of sensor principles

## Camera Sensors

### RGB Camera Configuration
```xml
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <camera name="head_camera">
      <horizontal_fov>1.089</horizontal_fov>  <!-- 62.4 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
      <topic_name>camera/image_raw</topic_name>
      <hack_baseline>0.07</hack_baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
    </plugin>
  </sensor>
</gazebo>
```

### Depth Camera Configuration
```xml
<gazebo reference="depth_camera_link">
  <sensor name="depth_camera" type="depth">
    <camera name="depth_head_camera">
      <horizontal_fov>1.089</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.01</stddev>
      </noise>
    </camera>
    <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <baseline>0.2</baseline>
      <always_on>true</always_on>
      <update_rate>30.0</update_rate>
      <camera_name>depth_camera</camera_name>
      <frame_name>depth_camera_optical_frame</frame_name>
      <point_cloud_topic>depth_camera/points</point_cloud_topic>
      <depth_image_topic>depth_camera/depth/image_raw</depth_image_topic>
      <depth_image_camera_info_topic>depth_camera/depth/camera_info</depth_image_camera_info_topic>
      <point_cloud_update_rate>10</point_cloud_update_rate>
      <point_cloud_cutoff>0.5</point_cloud_cutoff>
      <point_cloud_cutoff_max>3.0</point_cloud_cutoff_max>
      <Cx>320.5</Cx>
      <Cy>240.5</Cy>
      <focal_length>320.0</focal_length>
    </plugin>
  </sensor>
</gazebo>
```

### Stereo Camera Configuration
```xml
<gazebo reference="left_camera_link">
  <sensor name="stereo_left" type="camera">
    <camera name="left">
      <horizontal_fov>1.089</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
    </camera>
  </sensor>
</gazebo>

<gazebo reference="right_camera_link">
  <sensor name="stereo_right" type="camera">
    <camera name="right">
      <horizontal_fov>1.089</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
    </camera>
  </sensor>
</gazebo>
```

## LiDAR Sensors

### 2D LiDAR Configuration
```xml
<gazebo reference="laser_link">
  <sensor name="laser" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1.0</resolution>
          <min_angle>-3.14159</min_angle>  <!-- -π -->
          <max_angle>3.14159</max_angle>    <!-- π -->
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>10.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/laser</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>laser_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### 3D LiDAR Configuration (Velodyne-style)
```xml
<gazebo reference="velodyne_link">
  <sensor name="velodyne" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>1800</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
        <vertical>
          <samples>16</samples>
          <resolution>1</resolution>
          <min_angle>-0.2618</min_angle>  <!-- -15 degrees -->
          <max_angle>0.2618</max_angle>   <!-- 15 degrees -->
        </vertical>
      </scan>
      <range>
        <min>0.2</min>
        <max>100.0</max>
        <resolution>0.001</resolution>
      </range>
    </ray>
    <plugin name="velodyne_controller" filename="libgazebo_ros_velodyne_gpu_laser.so">
      <topic_name>velodyne_points</topic_name>
      <frame_name>velodyne_link</frame_name>
      <min_range>0.2</min_range>
      <max_range>100.0</max_range>
      <gaussian_noise>0.008</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

## IMU Sensors

### IMU Configuration
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
      <update_rate>100</update_rate>
      <gaussian_noise>0.001</gaussian_noise>
      <accel_gaussian_noise>0.017</accel_gaussian_noise>
      <rate_gaussian_noise>0.001</rate_gaussian_noise>
      <topic_name>imu/data</topic_name>
      <serviceName>imu_service</serviceName>
    </plugin>
  </sensor>
</gazebo>
```

### Accelerometer Configuration
```xml
<gazebo reference="accelerometer_link">
  <sensor name="accelerometer" type="accelerometer">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <plugin name="accelerometer_plugin" filename="libgazebo_ros_imu_sensor.so">
      <topic_name>accelerometer/data</topic_name>
      <frame_name>accelerometer_link</frame_name>
      <update_rate>100</update_rate>
      <gaussian_noise>0.017</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

## Force/Torque Sensors

### Force/Torque Sensor Configuration
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

<!-- Plugin for force/torque sensor -->
<gazebo reference="left_foot_joint">
  <sensor name="ft_sensor" type="force_torque">
    <plugin name="ft_plugin" filename="libgazebo_ros_ft_sensor.so">
      <topic_name>left_foot/force_torque</topic_name>
      <joint_name>left_foot_joint</joint_name>
      <update_rate>100</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

## GPS and Localization Sensors

### GPS Sensor Configuration
```xml
<gazebo reference="gps_link">
  <sensor name="gps" type="gps">
    <always_on>true</always_on>
    <update_rate>1</update_rate>
    <plugin name="gps_plugin" filename="libgazebo_ros_gps.so">
      <topic_name>gps/fix</topic_name>
      <frame_name>gps_link</frame_name>
      <update_rate>1</update_rate>
      <gaussian_noise>0.1</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

## Sensor Fusion and Processing

### Creating Sensor Processing Nodes

```python
# sensor_fusion.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan, Image
from geometry_msgs.msg import PointStamped
import numpy as np

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Subscribers for different sensors
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.camera_sub = self.create_subscription(Image, 'camera/image_raw', self.camera_callback, 10)

        # Publisher for fused data
        self.fused_pub = self.create_publisher(PointStamped, 'fused_sensor_data', 10)

        # Initialize sensor data storage
        self.imu_data = None
        self.scan_data = None
        self.camera_data = None

    def imu_callback(self, msg):
        self.imu_data = msg
        self.process_sensor_data()

    def scan_callback(self, msg):
        self.scan_data = msg
        self.process_sensor_data()

    def camera_callback(self, msg):
        self.camera_data = msg
        self.process_sensor_data()

    def process_sensor_data(self):
        # Implement sensor fusion logic
        if self.imu_data and self.scan_data:
            # Example: Combine IMU orientation with LiDAR data
            fused_point = PointStamped()
            fused_point.header.stamp = self.get_clock().now().to_msg()
            fused_point.header.frame_id = "fused_frame"

            # Process and combine sensor data
            # This is a simplified example
            fused_point.point.x = self.scan_data.ranges[0] if self.scan_data.ranges[0] > 0 else 0.0
            fused_point.point.y = self.imu_data.orientation.y
            fused_point.point.z = self.imu_data.linear_acceleration.z

            self.fused_pub.publish(fused_point)

def main(args=None):
    rclpy.init(args=args)
    sensor_fusion_node = SensorFusionNode()
    rclpy.spin(sensor_fusion_node)
    sensor_fusion_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Calibration

### Camera Calibration
```bash
# Use camera calibration tools
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025 image:=/camera/image_raw camera:=/camera
```

### IMU Calibration
- Static calibration for bias correction
- Dynamic calibration for scale factors
- Temperature compensation if needed

### LiDAR Calibration
- Extrinsics: Position and orientation relative to robot
- Intrinsics: Internal parameters (if applicable)
- Multi-sensor alignment

## Performance Considerations

### Update Rate Optimization
- Balance accuracy vs. performance
- Different sensors need different update rates
- Consider computational load

### Noise Modeling
- Add realistic noise to sensor outputs
- Model sensor-specific noise characteristics
- Validate noise levels against real sensors

### Bandwidth Management
- Consider data throughput requirements
- Implement data compression if needed
- Use appropriate message types

## Sensor Validation

### Data Quality Checks
- Verify sensor data ranges
- Check for missing or corrupted data
- Validate timestamp synchronization

### Comparison with Real Sensors
- Compare simulation vs. real sensor outputs
- Validate noise characteristics
- Check response times and delays

### Environmental Validation
- Test sensors in different environments
- Validate behavior under various conditions
- Check for sensor interference

## Humanoid-Specific Sensor Placement

### Head Sensors
- Cameras for vision-based perception
- Microphones for voice interaction
- IMU for head orientation

### Body Sensors
- IMUs for balance and orientation
- Force/torque sensors in feet for balance
- Joint position sensors for kinematics

### Hand Sensors
- Tactile sensors for manipulation
- Force sensors for grip control
- Cameras for object recognition

## Best Practices

### Sensor Configuration
1. Use realistic sensor parameters based on actual hardware
2. Implement appropriate noise models
3. Consider computational performance
4. Validate sensor data quality

### Integration
- Ensure proper coordinate frame transformations
- Synchronize sensor timestamps
- Implement proper error handling
- Provide sensor health monitoring

### Documentation
- Document sensor specifications and parameters
- Include calibration procedures
- Provide troubleshooting guides
- Maintain sensor configuration templates

## Advanced Topics

### Multi-Sensor Simulation
Simulate coordinated multi-sensor systems:
- Stereo vision for depth perception
- Multi-LiDAR systems for 360° coverage
- Sensor arrays for redundancy

### Dynamic Sensor Environments
- Moving objects affecting sensor data
- Changing lighting conditions
- Weather effects on sensors

### Sensor Failure Simulation
- Simulate sensor degradation
- Model complete sensor failures
- Test fault-tolerant algorithms

## Next Steps

After mastering sensor simulation, explore [NVIDIA Isaac Sim](/docs/modules/nvidia-isaac/isaac-sim) to learn about advanced photorealistic simulation and synthetic data generation for humanoid robots.