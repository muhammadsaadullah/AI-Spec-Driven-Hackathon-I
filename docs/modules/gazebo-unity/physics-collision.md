---
title: "Physics and Collision Simulation"
description: "Advanced physics simulation and collision detection for humanoid robots in Gazebo"
keywords: ["physics", "collision", "gazebo", "simulation", "humanoid", "robotics"]
sidebar_position: 3
---

# Physics and Collision Simulation

This module covers advanced physics simulation and collision detection techniques essential for realistic humanoid robot simulation in Gazebo. Proper physics modeling is crucial for stable locomotion and realistic interactions.

## Learning Objectives

By the end of this module, you will be able to:
- Configure advanced physics parameters for humanoid stability
- Implement realistic collision detection and response
- Model ground contact physics for bipedal locomotion
- Optimize physics simulation for performance
- Debug common physics-related issues in humanoid simulation

## Prerequisites

- Gazebo simulation setup knowledge
- Basic understanding of physics simulation
- ROS 2 integration with Gazebo

## Advanced Physics Configuration

### Physics Engine Selection

Gazebo supports multiple physics engines, each with different characteristics:

#### ODE (Open Dynamics Engine)
- Best for humanoid locomotion
- Good balance of stability and performance
- Recommended for contact-rich scenarios

```xml
<physics name="ode_physics" type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>
  <gravity>0 0 -9.80665</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>1000</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.000001</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

#### Bullet Physics
- Good for complex contact scenarios
- Better for soft body simulation
- May require more tuning for humanoid stability

#### DART Physics
- Advanced multi-body dynamics
- Good for complex articulated systems
- Experimental support in Gazebo

### Time Step Configuration

For stable humanoid simulation:
- Use small time steps (0.001s or smaller)
- Balance stability vs. performance
- Consider using adaptive time stepping

```xml
<physics name="stable_physics">
  <max_step_size>0.001</max_step_size>  <!-- 1ms time step -->
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>  <!-- 1000 Hz -->
</physics>
```

## Collision Detection and Response

### Collision Geometry Types

#### Box Collision
Best for simple, stable contacts:
```xml
<collision name="foot_collision">
  <geometry>
    <box>
      <size>0.2 0.1 0.02</size>  <!-- 20cm x 10cm x 2cm -->
    </box>
  </geometry>
  <surface>
    <friction>
      <ode>
        <mu>0.8</mu>  <!-- Coefficient of friction -->
        <mu2>0.8</mu2>
        <slip1>0.0</slip1>
        <slip2>0.0</slip2>
      </ode>
    </friction>
    <bounce>
      <restitution_coefficient>0.01</restitution_coefficient>  <!-- Minimal bounce -->
      <threshold>100000</threshold>
    </bounce>
    <contact>
      <ode>
        <soft_cfm>0.000001</soft_cfm>
        <soft_erp>0.2</soft_erp>
        <kp>1000000000000.0</kp>  <!-- Penetration correction stiffness -->
        <kd>1.0</kd>  <!-- Damping coefficient -->
        <max_vel>100.0</max_vel>
        <min_depth>0.001</min_depth>  <!-- Penetration tolerance -->
      </ode>
    </contact>
  </surface>
</collision>
```

#### Cylinder Collision
Good for limbs with consistent contact areas:
```xml
<collision name="thigh_collision">
  <geometry>
    <cylinder>
      <radius>0.05</radius>  <!-- 5cm radius -->
      <length>0.4</length>   <!-- 40cm length -->
    </cylinder>
  </geometry>
  <!-- Surface properties similar to box collision -->
</collision>
```

#### Mesh Collision
For complex shapes (use sparingly for performance):
```xml
<collision name="head_collision">
  <geometry>
    <mesh>
      <uri>package://my_robot_description/meshes/head_collision.dae</uri>
    </mesh>
  </geometry>
</collision>
```

### Friction Modeling for Humanoid Locomotion

For stable bipedal walking:
- Foot-ground friction: 0.8-1.0 (for grip)
- Joint friction: Minimal (to avoid energy loss)
- Use anisotropic friction for directional properties

```xml
<surface>
  <friction>
    <ode>
      <mu>0.9</mu>      <!-- Forward/backward friction -->
      <mu2>0.8</mu2>    <!-- Lateral friction -->
      <fdir1>1 0 0</fdir1>  <!-- Direction of anisotropic friction -->
    </ode>
  </friction>
</surface>
```

## Ground Contact Physics

### Ground Surface Properties
```xml
<model name="ground_plane">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <plane>
          <normal>0 0 1</normal>
          <size>100 100</size>
        </plane>
      </geometry>
      <surface>
        <friction>
          <ode>
            <mu>0.9</mu>
            <mu2>0.9</mu2>
          </ode>
        </friction>
        <bounce>
          <restitution_coefficient>0.0</restitution_coefficient>
        </bounce>
        <contact>
          <ode>
            <soft_cfm>0.000001</soft_cfm>
            <soft_erp>0.2</soft_erp>
            <kp>1000000000000.0</kp>
            <kd>1.0</kd>
            <max_vel>100.0</max_vel>
            <min_depth>0.001</min_depth>
          </ode>
        </contact>
      </surface>
    </collision>
  </link>
</model>
```

### Foot-Ground Interaction
Critical for stable walking:
- Use multiple contact points per foot
- Configure appropriate contact stiffness
- Consider terrain adaptation

```xml
<link name="left_foot">
  <!-- Multiple collision elements for better contact -->
  <collision name="left_foot_front_collision">
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
    <geometry>
      <box><size>0.1 0.08 0.01</size></box>
    </geometry>
  </collision>
  <collision name="left_foot_back_collision">
    <origin xyz="-0.05 0 0" rpy="0 0 0"/>
    <geometry>
      <box><size>0.1 0.08 0.01</size></box>
    </geometry>
  </collision>
  <!-- Add heel and toe contacts as needed -->
</link>
```

## Humanoid-Specific Physics Challenges

### Center of Mass Stability
- Keep CoM within support polygon during walking
- Model realistic mass distribution
- Consider dynamic CoM adjustments

### Balance and Control
- Use PD controllers with appropriate gains
- Implement whole-body control strategies
- Consider feedback from IMU and force/torque sensors

### Collision Avoidance
- Configure self-collision detection
- Set appropriate safety margins
- Implement collision-free motion planning

## Performance Optimization

### Simplified Collision Models
Use simpler collision geometry for performance:
```xml
<!-- Instead of complex mesh collision -->
<collision name="complex_shape_collision">
  <!-- Use multiple simple shapes -->
  <collision name="shape_part1">
    <geometry><box><size>0.1 0.1 0.1</size></box></geometry>
  </collision>
  <collision name="shape_part2">
    <geometry><cylinder><radius>0.05</radius><length>0.2</length></cylinder></geometry>
  </collision>
</collision>
```

### Adaptive Simulation Parameters
- Adjust time step based on contact events
- Use different parameters for different simulation phases
- Implement performance monitoring

### Multi-threading
Enable physics multi-threading for complex scenarios:
```xml
<physics name="multithreaded_physics" type="ode">
  <ode>
    <thread_position_correction>true</thread_position_correction>
    <solver>
      <threads>4</threads>  <!-- Use multiple threads -->
    </solver>
  </ode>
</physics>
```

## Debugging Physics Issues

### Common Problems and Solutions

#### Robot Falling Through Ground
1. Check collision geometry definition
2. Verify mass and inertial properties
3. Adjust contact parameters (CFM/ERP)
4. Ensure proper gravity settings

#### Unstable Joint Behavior
1. Check joint limits and types
2. Verify transmission configurations
3. Adjust controller parameters
4. Review physics time step

#### Excessive Jittering
1. Increase solver iterations
2. Adjust ERP and CFM values
3. Verify collision geometry
4. Check for over-constrained systems

### Physics Debugging Tools

#### Contact Visualization
Enable contact force visualization:
```bash
gz sim --render-engine ogre --gui-config contact_visualization.config
```

#### Physics Statistics
Monitor simulation performance:
```bash
# Check real-time factor and update rates
gz topic -e -t /stats
```

## Integration with Control Systems

### Real-time Control Considerations
- Account for simulation latency
- Synchronize control loops with physics updates
- Implement appropriate buffering

### Sensor Integration
- Configure sensor update rates appropriately
- Account for sensor noise and delay
- Implement sensor fusion algorithms

## Advanced Topics

### Soft Contact Modeling
For more realistic contact behavior:
```xml
<surface>
  <contact>
    <ode>
      <soft_cfm>0.0001</soft_cfm>  <!-- Compliance factor -->
      <soft_erp>0.1</soft_erp>     <!-- Error reduction -->
      <kp>1000000000.0</kp>        <!-- Stiffness -->
      <kd>100.0</kd>               <!-- Damping -->
    </ode>
  </contact>
</surface>
```

### Terrain Simulation
Model different ground types:
- Grass: Lower friction, higher compliance
- Concrete: Higher friction, lower compliance
- Sand: Lower friction, higher compliance

### Multi-body Dynamics
Handle complex interactions:
- Multiple contact points
- Simultaneous collisions
- Joint coupling effects

## Best Practices

### Physics Parameter Tuning
1. Start with conservative parameters
2. Gradually adjust for stability
3. Validate against real-world behavior
4. Document parameter choices

### Model Validation
- Compare simulation results with theoretical models
- Validate against physical robot behavior
- Test edge cases and failure scenarios
- Document validation procedures

### Performance Monitoring
- Monitor real-time factor consistently
- Track simulation stability metrics
- Profile computational performance
- Optimize based on usage patterns

## Next Steps

After mastering physics and collision simulation, explore [Sensor Simulation](/docs/modules/gazebo-unity/sensors-simulation) to learn how to simulate realistic sensors for your humanoid robot's perception system.