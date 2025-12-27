---
title: "Weeks 11-12: Humanoid Robot Development"
description: "Humanoid robot kinematics and dynamics, bipedal locomotion and balance control, manipulation and grasping with humanoid hands, natural human-robot interaction design"
keywords: ["humanoid", "kinematics", "dynamics", "locomotion", "balance", "manipulation", "interaction"]
sidebar_position: 5
---

# Weeks 11-12: Humanoid Robot Development

Welcome to the humanoid robot development module of the Physical AI & Humanoid Robotics course! These two weeks focus on the specialized challenges and techniques required for developing and controlling humanoid robots. You'll learn about the unique aspects of humanoid kinematics, dynamics, locomotion, and manipulation that make these robots both fascinating and challenging.

## Learning Objectives

By the end of these two weeks, you will be able to:
- Understand and implement humanoid robot kinematics and inverse kinematics
- Design and implement bipedal locomotion and balance control systems
- Develop manipulation and grasping strategies for humanoid hands
- Create natural human-robot interaction systems
- Apply advanced control techniques for humanoid stability
- Integrate perception systems for humanoid navigation and manipulation

## Prerequisites

- Completion of Weeks 1-10 (Physical AI foundations, ROS 2, Gazebo, and Isaac)
- Strong understanding of linear algebra and calculus
- Basic knowledge of control theory
- Experience with Python and ROS 2
- Understanding of robot kinematics from previous modules

## Week 11: Humanoid Kinematics and Locomotion

### Day 1: Humanoid Robot Kinematics

#### Humanoid Robot Structure

Humanoid robots are designed to mimic human body structure with specific kinematic chains:

```
                    Head
                     |
              [Neck Joint]
                     |
                   Torso
                  /     \
        [Shoulder]       [Shoulder]
            |               |
         Upper Arm       Upper Arm
            |               |
        [Elbow Joint]   [Elbow Joint]
            |               |
         Lower Arm       Lower Arm
            |               |
        [Wrist Joint]   [Wrist Joint]
            |               |
          Hand           Hand
            |               |
        [Hip Joint]   [Hip Joint]
            |               |
           Thigh           Thigh
            |               |
        [Knee Joint]   [Knee Joint]
            |               |
          Shin            Shin
            |               |
        [Ankle Joint]   [Ankle Joint]
            |               |
          Foot           Foot
```

#### Degrees of Freedom in Humanoid Robots

Humanoid robots typically have 20-50+ degrees of freedom (DOF):

**Upper Body (14-20 DOF):**
- Head: 3 DOF (pitch, yaw, roll)
- Left Arm: 7 DOF (shoulder: 3, elbow: 1, wrist: 3)
- Right Arm: 7 DOF (shoulder: 3, elbow: 1, wrist: 3)
- Torso: 1-3 DOF

**Lower Body (12-14 DOF):**
- Left Leg: 6-7 DOF (hip: 3, knee: 1, ankle: 2-3)
- Right Leg: 6-7 DOF (hip: 3, knee: 1, ankle: 2-3)

#### Forward Kinematics

Forward kinematics calculates the end-effector position from joint angles:

```python
# humanoid_kinematics.py
import numpy as np
from math import sin, cos, sqrt

class HumanoidKinematics:
    def __init__(self):
        # Define link lengths (example values for a humanoid)
        self.link_lengths = {
            'upper_arm': 0.3,    # meters
            'lower_arm': 0.25,   # meters
            'thigh': 0.4,        # meters
            'shin': 0.4,         # meters
            'torso': 0.6,        # meters
            'head_height': 0.1   # meters
        }

    def dh_transform(self, a, alpha, d, theta):
        """Denavit-Hartenberg transformation matrix"""
        return np.array([
            [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
            [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def arm_forward_kinematics(self, joint_angles, arm='left'):
        """Calculate forward kinematics for arm"""
        # Extract joint angles (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_pitch, wrist_yaw, wrist_roll)
        q1, q2, q3, q4, q5, q6, q7 = joint_angles

        # Calculate transformation matrices for each joint
        T1 = self.dh_transform(0, -np.pi/2, 0, q1)
        T2 = self.dh_transform(self.link_lengths['upper_arm'], 0, 0, q2)
        T3 = self.dh_transform(self.link_lengths['lower_arm'], 0, 0, q3)
        T4 = self.dh_transform(0, -np.pi/2, 0, q4)
        T5 = self.dh_transform(0, np.pi/2, 0, q5)
        T6 = self.dh_transform(0, -np.pi/2, 0, q6)
        T7 = self.dh_transform(0, 0, 0, q7)

        # Combine transformations
        T_total = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T7

        return T_total

    def leg_forward_kinematics(self, joint_angles, leg='left'):
        """Calculate forward kinematics for leg"""
        # Extract joint angles (hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll)
        q1, q2, q3, q4, q5, q6 = joint_angles

        # Calculate transformation matrices for each joint
        T1 = self.dh_transform(0, -np.pi/2, 0, q1)
        T2 = self.dh_transform(0, np.pi/2, 0, q2)
        T3 = self.dh_transform(self.link_lengths['thigh'], 0, 0, q3)
        T4 = self.dh_transform(self.link_lengths['shin'], 0, 0, q4)
        T5 = self.dh_transform(0, -np.pi/2, 0, q5)
        T6 = self.dh_transform(0, 0, 0, q6)

        # Combine transformations
        T_total = T1 @ T2 @ T3 @ T4 @ T5 @ T6

        return T_total

    def get_end_effector_position(self, T):
        """Extract end-effector position from transformation matrix"""
        return T[:3, 3]

# Example usage
kinematics = HumanoidKinematics()
arm_joints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # Example joint angles
T_arm = kinematics.arm_forward_kinematics(arm_joints)
end_pos = kinematics.get_end_effector_position(T_arm)
print(f"End-effector position: {end_pos}")
```

### Day 2: Inverse Kinematics for Humanoid Robots

#### Inverse Kinematics Challenges

Humanoid inverse kinematics is complex due to:
- Redundant DOF (multiple solutions possible)
- Joint limits and constraints
- Collision avoidance
- Balance maintenance

#### Analytical vs Numerical IK

**Analytical IK**: Closed-form solutions for specific kinematic chains
**Numerical IK**: Iterative methods for general solutions

```python
# numerical_ik.py
import numpy as np
from scipy.optimize import minimize

class HumanoidInverseKinematics:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.jacobian_epsilon = 1e-6

    def jacobian_transpose(self, joint_angles, target_pos, end_effector_idx):
        """Calculate Jacobian transpose for IK"""
        n_joints = len(joint_angles)
        jacobian = np.zeros((3, n_joints))

        # Calculate Jacobian using numerical differentiation
        for i in range(n_joints):
            # Perturb joint angle
            delta = 1e-6
            angles_plus = joint_angles.copy()
            angles_plus[i] += delta

            angles_minus = joint_angles.copy()
            angles_minus[i] -= delta

            # Calculate forward kinematics for both
            pos_plus = self.forward_kinematics(angles_plus)[end_effector_idx]
            pos_minus = self.forward_kinematics(angles_minus)[end_effector_idx]

            # Calculate partial derivative
            jacobian[:, i] = (pos_plus - pos_minus) / (2 * delta)

        return jacobian

    def inverse_kinematics_jacobian(self, current_angles, target_pos, max_iterations=100):
        """Solve IK using Jacobian transpose method"""
        angles = current_angles.copy()

        for iteration in range(max_iterations):
            # Calculate current end-effector position
            current_pos = self.forward_kinematics(angles)[-1]  # Assuming last link is end-effector

            # Calculate error
            error = target_pos - current_pos

            # Check if we're close enough
            if np.linalg.norm(error) < 1e-3:
                break

            # Calculate Jacobian
            J = self.jacobian_transpose(angles, target_pos, -1)

            # Calculate joint angle update using Jacobian transpose
            delta_angles = np.dot(J.T, error) * 0.01  # Learning rate

            # Update joint angles
            angles += delta_angles

            # Apply joint limits
            angles = self.apply_joint_limits(angles)

        return angles

    def forward_kinematics(self, joint_angles):
        """Forward kinematics implementation"""
        # This would typically use the robot's URDF model
        # For simplicity, returning a placeholder
        return np.array([0, 0, 0])  # Placeholder

    def apply_joint_limits(self, angles):
        """Apply joint limits to prevent damage"""
        # Example joint limits (in radians)
        min_limits = np.array([-np.pi, -np.pi/2, -np.pi, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi])
        max_limits = np.array([np.pi, np.pi/2, np.pi, np.pi/2, np.pi/2, np.pi/2, np.pi])

        return np.clip(angles, min_limits, max_limits)

# Alternative: Cyclic Coordinate Descent (CCD) for IK
class CCDInverseKinematics:
    def __init__(self):
        self.max_iterations = 100
        self.tolerance = 1e-3

    def inverse_kinematics_ccd(self, joint_positions, target_pos, joint_hierarchy):
        """
        Solve inverse kinematics using Cyclic Coordinate Descent
        joint_positions: list of joint positions in world coordinates
        target_pos: target position to reach
        joint_hierarchy: parent-child relationships
        """
        n_joints = len(joint_positions)

        for iteration in range(self.max_iterations):
            # Start from the end effector and work backwards
            for i in range(n_joints - 1, 0, -1):
                # Get the current joint and its parent
                current_pos = joint_positions[i]
                parent_pos = joint_positions[i - 1]

                # Calculate vectors
                target_vec = target_pos - parent_pos
                current_vec = current_pos - parent_pos

                # Calculate rotation to align current vector with target vector
                target_vec_norm = target_vec / np.linalg.norm(target_vec)
                current_vec_norm = current_vec / np.linalg.norm(current_vec)

                # Calculate rotation axis and angle
                rotation_axis = np.cross(current_vec_norm, target_vec_norm)
                rotation_axis_norm = np.linalg.norm(rotation_axis)

                if rotation_axis_norm > 1e-6:
                    rotation_axis = rotation_axis / rotation_axis_norm
                    rotation_angle = np.arccos(
                        np.clip(np.dot(current_vec_norm, target_vec_norm), -1.0, 1.0)
                    )

                    # Apply rotation to all child joints
                    self.rotate_chain(joint_positions, i - 1, rotation_axis, rotation_angle)

            # Check if we're close enough
            final_pos = joint_positions[-1]
            if np.linalg.norm(final_pos - target_pos) < self.tolerance:
                break

        return joint_positions

    def rotate_chain(self, joint_positions, pivot_idx, axis, angle):
        """Rotate a chain of joints around a pivot point"""
        pivot_pos = joint_positions[pivot_idx]

        # Create rotation matrix
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        one_minus_cos = 1 - cos_angle

        # Rotation matrix using axis-angle representation
        R = np.array([
            [cos_angle + axis[0]**2 * one_minus_cos,
             axis[0]*axis[1]*one_minus_cos - axis[2]*sin_angle,
             axis[0]*axis[2]*one_minus_cos + axis[1]*sin_angle],
            [axis[1]*axis[0]*one_minus_cos + axis[2]*sin_angle,
             cos_angle + axis[1]**2 * one_minus_cos,
             axis[1]*axis[2]*one_minus_cos - axis[0]*sin_angle],
            [axis[2]*axis[0]*one_minus_cos - axis[1]*sin_angle,
             axis[2]*axis[1]*one_minus_cos + axis[0]*sin_angle,
             cos_angle + axis[2]**2 * one_minus_cos]
        ])

        # Apply rotation to all joints after the pivot
        for i in range(pivot_idx + 1, len(joint_positions)):
            # Translate to origin
            rel_pos = joint_positions[i] - pivot_pos
            # Rotate
            rotated_pos = R @ rel_pos
            # Translate back
            joint_positions[i] = rotated_pos + pivot_pos
```

### Day 3: Bipedal Locomotion Fundamentals

#### Walking Gaits and Patterns

Humanoid robots use various walking patterns:

**Static Walking**: Stable at every step
- Center of Mass (CoM) always within support polygon
- Slow but very stable
- Uses large support polygon

**Dynamic Walking**: CoM can move outside support polygon
- More human-like
- Faster but requires active balance control
- Requires sophisticated control algorithms

#### Zero Moment Point (ZMP) Theory

ZMP is crucial for stable bipedal locomotion:

```python
# zmp_control.py
import numpy as np
from scipy import signal

class ZMPController:
    def __init__(self, robot_mass, gravity=9.81):
        self.mass = robot_mass
        self.gravity = gravity
        self.com_height = 0.8  # Center of mass height in meters

    def calculate_zmp(self, com_pos, com_vel, com_acc):
        """
        Calculate Zero Moment Point from CoM information
        ZMP_x = com_x - (com_height / gravity) * com_acc_x
        ZMP_y = com_y - (com_height / gravity) * com_acc_y
        """
        zmp_x = com_pos[0] - (self.com_height / self.gravity) * com_acc[0]
        zmp_y = com_pos[1] - (self.com_height / self.gravity) * com_acc[1]

        return np.array([zmp_x, zmp_y, 0.0])

    def zmp_stability_check(self, zmp_pos, support_polygon):
        """
        Check if ZMP is within support polygon
        support_polygon: list of (x, y) coordinates of support polygon vertices
        """
        # Simple point-in-polygon test (for quadrilateral support polygon)
        x, y = zmp_pos[0], zmp_pos[1]
        n = len(support_polygon)

        inside = False
        j = n - 1

        for i in range(n):
            xi, yi = support_polygon[i]
            xj, yj = support_polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

class WalkingPatternGenerator:
    def __init__(self, step_length=0.3, step_height=0.05, step_time=0.8):
        self.step_length = step_length
        self.step_height = step_height
        self.step_time = step_time
        self.half_step_time = step_time / 2.0

    def generate_foot_trajectory(self, start_pos, end_pos, step_time, step_height):
        """
        Generate smooth foot trajectory for walking
        """
        # Time vector
        t = np.linspace(0, step_time, int(step_time * 100))  # 100 Hz

        # Linear interpolation for x, y positions
        x = np.linspace(start_pos[0], end_pos[0], len(t))
        y = np.linspace(start_pos[1], end_pos[1], len(t))

        # Parabolic trajectory for z (step height)
        z = np.zeros_like(t)
        for i, time in enumerate(t):
            if time <= self.half_step_time:
                # Rising phase
                z[i] = start_pos[2] + step_height * (time / self.half_step_time)**2
            else:
                # Falling phase
                z[i] = start_pos[2] + step_height * (1 - (time - self.half_step_time) / self.half_step_time)**2

        return np.column_stack([x, y, z])

    def generate_com_trajectory(self, start_pos, target_pos, step_time):
        """
        Generate CoM trajectory following a 3rd order polynomial
        """
        # 3rd order polynomial: p(t) = a0 + a1*t + a2*t^2 + a3*t^3
        # Boundary conditions: start at start_pos, end at target_pos
        # with zero velocity at start and end

        t = np.linspace(0, step_time, int(step_time * 100))

        # Coefficients for 3rd order polynomial
        a0 = start_pos
        a1 = 0  # zero initial velocity
        a2 = 3 * (target_pos - start_pos) / step_time**2
        a3 = -2 * (target_pos - start_pos) / step_time**3

        # Calculate trajectory
        trajectory = a0 + a1*t + a2*t**2 + a3*t**3

        # Calculate velocity and acceleration
        velocity = a1 + 2*a2*t + 3*a3*t**2
        acceleration = 2*a2 + 6*a3*t

        return trajectory, velocity, acceleration
```

### Day 4: Balance Control Systems

#### Center of Mass (CoM) Control

Balance control is critical for humanoid robots:

```python
# balance_control.py
import numpy as np
from scipy import signal

class BalanceController:
    def __init__(self, robot_mass=60.0, com_height=0.8, control_freq=100):
        self.mass = robot_mass
        self.com_height = com_height
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq

        # PID controller gains
        self.kp = 100.0  # Proportional gain
        self.ki = 10.0   # Integral gain
        self.kd = 20.0   # Derivative gain

        # Error accumulation for integral term
        self.error_integral = np.zeros(2)
        self.prev_error = np.zeros(2)

        # Low-pass filter for sensor data
        self.filter_coeff = 0.1

    def update_balance(self, current_com, desired_com, current_com_vel, zmp_ref):
        """
        Update balance control based on CoM and ZMP feedback
        """
        # Calculate error
        error = desired_com[:2] - current_com[:2]  # Only x, y components

        # Update integral (with anti-windup)
        self.error_integral += error * self.dt
        # Limit integral to prevent windup
        self.error_integral = np.clip(self.error_integral, -1.0, 1.0)

        # Calculate derivative
        error_derivative = (error - self.prev_error) / self.dt

        # PID control
        control_output = (self.kp * error +
                         self.ki * self.error_integral +
                         self.kd * error_derivative)

        # Update previous error
        self.prev_error = error

        # Convert to joint torques or foot forces based on robot model
        joint_torques = self.map_to_joint_space(control_output)

        return joint_torques

    def map_to_joint_space(self, com_control):
        """
        Map CoM control commands to joint torques
        This is a simplified mapping - real implementation would use full robot model
        """
        # This would typically involve the robot's Jacobian and inverse dynamics
        # For now, return a simplified mapping
        n_joints = 28  # Example for humanoid
        joint_torques = np.zeros(n_joints)

        # Distribute control effort to relevant joints
        # Legs for balance, torso for posture
        leg_joints = [6, 7, 8, 9, 10, 11, 19, 20, 21, 22, 23, 24]  # Example leg joint indices

        for i, joint_idx in enumerate(leg_joints):
            if i < len(com_control):
                joint_torques[joint_idx] = com_control[i % 2] * 0.1  # Distribute x, y control

        return joint_torques

class CapturePointController:
    """
    Capture Point based balance control
    The capture point is where the robot would need to step to stop its motion
    """
    def __init__(self, com_height=0.8, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)

    def calculate_capture_point(self, com_pos, com_vel):
        """
        Calculate capture point where robot needs to step to stop
        Capture Point = CoM + CoM_vel / omega
        """
        capture_point = com_pos + com_vel / self.omega
        return capture_point

    def calculate_desired_foot_position(self, current_foot_pos, capture_point, step_margin=0.1):
        """
        Calculate where to place the foot based on capture point
        """
        # Calculate desired foot position
        desired_foot_pos = capture_point

        # Ensure foot is placed with appropriate margins
        # (prevent stepping too close to CoM)
        step_vec = desired_foot_pos - current_foot_pos
        step_dist = np.linalg.norm(step_vec)

        if step_dist < step_margin:
            # Step in the direction of the capture point
            if step_dist > 1e-6:  # Avoid division by zero
                step_dir = step_vec / step_dist
                desired_foot_pos = current_foot_pos + step_dir * step_margin

        return desired_foot_pos
```

### Day 5: Hands-on Exercise - Basic Humanoid Control

#### Exercise: Implement Simple Balance Controller

1. Create a basic humanoid model with simplified kinematics
2. Implement a PID-based balance controller
3. Test the controller with simulated disturbances
4. Visualize the results

## Week 12: Manipulation and Human-Robot Interaction

### Day 6: Humanoid Hand Design and Grasping

#### Humanoid Hand Kinematics

Humanoid hands typically have 10-22 DOF with complex kinematics:

```python
# hand_kinematics.py
import numpy as np

class HumanoidHand:
    def __init__(self, hand_type='anthropomorphic'):
        self.hand_type = hand_type
        self.fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        self.joints_per_finger = 3  # MCP, PIP, DIP joints

        # Finger lengths (example values)
        self.finger_lengths = {
            'thumb': [0.03, 0.025, 0.02],      # [MCP, PIP, DIP]
            'index': [0.04, 0.03, 0.025],
            'middle': [0.045, 0.035, 0.03],
            'ring': [0.04, 0.03, 0.025],
            'pinky': [0.035, 0.025, 0.02]
        }

    def finger_forward_kinematics(self, finger_name, joint_angles):
        """
        Calculate forward kinematics for a single finger
        joint_angles: [MCP_angle, PIP_angle, DIP_angle] in radians
        """
        if finger_name not in self.finger_lengths:
            raise ValueError(f"Unknown finger: {finger_name}")

        lengths = self.finger_lengths[finger_name]
        q_mcp, q_pip, q_dip = joint_angles

        # Calculate positions of each joint
        # Simplified 2D model (x, y) for finger flexion
        mcp_pos = np.array([0, 0])
        pip_pos = mcp_pos + np.array([
            lengths[0] * np.cos(q_mcp),
            lengths[0] * np.sin(q_mcp)
        ])
        dip_pos = pip_pos + np.array([
            lengths[1] * np.cos(q_mcp + q_pip),
            lengths[1] * np.sin(q_mcp + q_pip)
        ])
        tip_pos = dip_pos + np.array([
            lengths[2] * np.cos(q_mcp + q_pip + q_dip),
            lengths[2] * np.sin(q_mcp + q_pip + q_dip)
        ])

        return {
            'mcp': mcp_pos,
            'pip': pip_pos,
            'dip': dip_pos,
            'tip': tip_pos
        }

    def hand_grasp_analysis(self, finger_angles):
        """
        Analyze grasp configuration for all fingers
        finger_angles: dict with finger names as keys and joint angles as values
        """
        grasp_analysis = {}

        for finger_name in self.fingers:
            if finger_name in finger_angles:
                positions = self.finger_forward_kinematics(finger_name, finger_angles[finger_name])
                grasp_analysis[finger_name] = positions

        return grasp_analysis

class GraspController:
    def __init__(self):
        self.hand = HumanoidHand()
        self.grasp_types = ['cylindrical', 'spherical', 'lateral', 'tip']

    def plan_grasp(self, object_shape, object_size, grasp_type='cylindrical'):
        """
        Plan finger positions for grasping an object
        """
        if grasp_type == 'cylindrical':
            return self.plan_cylindrical_grasp(object_size)
        elif grasp_type == 'spherical':
            return self.plan_spherical_grasp(object_size)
        elif grasp_type == 'lateral':
            return self.plan_lateral_grasp(object_size)
        elif grasp_type == 'tip':
            return self.plan_tip_grasp(object_size)
        else:
            raise ValueError(f"Unknown grasp type: {grasp_type}")

    def plan_cylindrical_grasp(self, object_diameter):
        """
        Plan grasp for cylindrical objects (cups, bottles, etc.)
        """
        # For cylindrical grasp, fingers wrap around the object
        # Calculate joint angles based on object diameter
        finger_angle = np.arcsin(object_diameter / 0.1)  # Simplified calculation

        # Example finger configuration for cylindrical grasp
        grasp_config = {
            'thumb': [0.5, 0.3, 0.2],  # [MCP, PIP, DIP]
            'index': [0.8, 0.6, 0.4],
            'middle': [0.8, 0.6, 0.4],
            'ring': [0.7, 0.5, 0.3],
            'pinky': [0.6, 0.4, 0.2]
        }

        return grasp_config

    def evaluate_grasp_stability(self, grasp_config, object_properties):
        """
        Evaluate the stability of a grasp configuration
        """
        # Calculate grasp quality metrics
        # This is a simplified evaluation
        grasp_analysis = self.hand.hand_grasp_analysis(grasp_config)

        # Calculate contact points and forces
        contact_points = []
        for finger, positions in grasp_analysis.items():
            contact_points.append(positions['tip'])

        # Calculate grasp quality (simplified)
        contact_points = np.array(contact_points)
        if len(contact_points) >= 3:
            # Calculate area of contact polygon as a basic stability measure
            # This is a very simplified approach
            centroid = np.mean(contact_points, axis=0)
            stability = np.mean([np.linalg.norm(point - centroid) for point in contact_points])
        else:
            stability = 0

        return {
            'stability': stability,
            'contact_points': contact_points,
            'grasp_analysis': grasp_analysis
        }
```

### Day 7: Manipulation Planning for Humanoids

#### Task Space Control

Humanoid manipulation requires coordination of multiple limbs:

```python
# manipulation_planning.py
import numpy as np
from scipy.spatial.transform import Rotation as R

class HumanoidManipulationPlanner:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.ik_solver = self.initialize_ik_solver()

    def initialize_ik_solver(self):
        """Initialize inverse kinematics solver for manipulation"""
        # This would typically integrate with a full IK library
        # For this example, we'll use a simple implementation
        return HumanoidInverseKinematics(self.robot_model)

    def plan_reach_motion(self, start_pose, target_pose, n_waypoints=10):
        """
        Plan reaching motion from start to target pose
        """
        # Linear interpolation in Cartesian space
        positions = np.linspace(start_pose[:3], target_pose[:3], n_waypoints)

        # Interpolate orientations using SLERP
        start_rot = R.from_quat(start_pose[3:])
        target_rot = R.from_quat(target_pose[3:])

        t_values = np.linspace(0, 1, n_waypoints)
        orientations = []

        for t in t_values:
            # Spherical linear interpolation for rotations
            interp_rot = R.slerp(start_rot, target_rot, t)
            orientations.append(interp_rot.as_quat())

        # Combine positions and orientations
        waypoints = []
        for pos, quat in zip(positions, orientations):
            waypoint = np.concatenate([pos, quat])
            waypoints.append(waypoint)

        return np.array(waypoints)

    def dual_arm_manipulation(self, left_target, right_target):
        """
        Plan coordinated motion for both arms
        """
        # This requires considering both arms simultaneously
        # and avoiding self-collisions
        left_joints = self.ik_solver.inverse_kinematics_jacobian(
            self.get_current_left_arm_joints(),
            left_target[:3]  # Position only
        )

        right_joints = self.ik_solver.inverse_kinematics_jacobian(
            self.get_current_right_arm_joints(),
            right_target[:3]  # Position only
        )

        # Check for self-collision between arms
        if self.check_arm_collision(left_joints, right_joints):
            # Adjust one arm to avoid collision
            right_joints = self.avoid_collision(left_joints, right_joints)

        return left_joints, right_joints

    def get_current_left_arm_joints(self):
        """Get current joint angles for left arm"""
        # This would interface with the actual robot
        return np.zeros(7)  # Placeholder

    def get_current_right_arm_joints(self):
        """Get current joint angles for right arm"""
        # This would interface with the actual robot
        return np.zeros(7)  # Placeholder

    def check_arm_collision(self, left_joints, right_joints):
        """Check for collision between arms"""
        # Simplified collision check
        # In reality, this would use full collision checking
        return False  # Placeholder

    def avoid_collision(self, left_joints, right_joints):
        """Adjust right arm to avoid collision with left arm"""
        # Collision avoidance algorithm
        return right_joints  # Placeholder

class ObjectManipulationController:
    def __init__(self):
        self.manipulation_planner = HumanoidManipulationPlanner(None)
        self.grasp_controller = GraspController()

    def pick_and_place(self, object_pose, target_pose):
        """
        Execute pick and place operation
        """
        # 1. Approach the object
        approach_pose = self.calculate_approach_pose(object_pose)
        self.move_to_pose(approach_pose)

        # 2. Grasp the object
        grasp_config = self.grasp_controller.plan_grasp(
            object_shape='cylindrical',
            object_size=0.05  # 5cm diameter
        )
        self.execute_grasp(grasp_config)

        # 3. Lift the object
        lift_pose = self.calculate_lift_pose(object_pose)
        self.move_to_pose(lift_pose)

        # 4. Transport to target location
        self.move_to_pose(target_pose)

        # 5. Place the object
        self.execute_release()

        # 6. Retract
        self.move_to_safe_position()

    def calculate_approach_pose(self, object_pose):
        """Calculate approach pose for grasping"""
        # Approach from above or side depending on object orientation
        approach_offset = np.array([0.0, 0.0, 0.1])  # 10cm above object
        approach_pose = object_pose.copy()
        approach_pose[:3] += approach_offset
        return approach_pose

    def calculate_lift_pose(self, object_pose):
        """Calculate lift pose after grasping"""
        lift_offset = np.array([0.0, 0.0, 0.2])  # Lift 20cm
        lift_pose = object_pose.copy()
        lift_pose[:3] += lift_offset
        return lift_pose

    def move_to_pose(self, target_pose):
        """Move end-effector to target pose"""
        # This would call the actual motion controller
        pass

    def execute_grasp(self, grasp_config):
        """Execute grasp with hand configuration"""
        # Control hand joints to achieve grasp configuration
        pass

    def execute_release(self):
        """Release the grasped object"""
        # Open hand to release object
        pass

    def move_to_safe_position(self):
        """Move arms to safe position"""
        # Move arms to neutral position
        pass
```

### Day 8: Natural Human-Robot Interaction

#### Social Robotics Principles

Humanoid robots need to interact naturally with humans:

```python
# human_robot_interaction.py
import numpy as np
import time
from enum import Enum

class InteractionMode(Enum):
    GREETING = "greeting"
    TASK_ASSISTANCE = "task_assistance"
    CONVERSATION = "conversation"
    LEAVING = "leaving"

class HumanoidInteractionController:
    def __init__(self):
        self.current_mode = InteractionMode.GREETING
        self.human_tracking = True
        self.gaze_control = True
        self.gesture_system = GestureSystem()
        self.speech_synthesizer = SpeechSynthesizer()

    def detect_human_presence(self):
        """
        Detect humans using sensors (camera, LiDAR, etc.)
        """
        # This would interface with actual sensors
        # Return list of detected humans with positions
        return [{'position': np.array([1.0, 0.0, 0.0]), 'confidence': 0.9}]

    def track_human(self, human_position):
        """
        Track human and orient robot accordingly
        """
        # Calculate direction to human
        robot_pos = np.array([0.0, 0.0, 0.0])  # Robot position
        direction_to_human = human_position - robot_pos
        direction_to_human = direction_to_human / np.linalg.norm(direction_to_human)

        # Orient robot torso and head toward human
        self.orient_towards(direction_to_human)

        # Control gaze to look at human
        if self.gaze_control:
            self.control_gaze(human_position)

    def orient_towards(self, direction):
        """
        Orient robot torso and head towards a direction
        """
        # Calculate yaw angle to face direction
        yaw = np.arctan2(direction[1], direction[0])

        # Set head and torso orientation
        head_joints = [0.0, yaw, 0.0]  # pitch, yaw, roll
        torso_joints = [0.0, yaw, 0.0]

        # Move to orientation (this would interface with actual controllers)
        self.move_head(head_joints)
        self.move_torso(torso_joints)

    def control_gaze(self, target_position):
        """
        Control eye and head movements to look at target
        """
        # Calculate gaze direction
        eye_pos = self.get_eye_position()  # Robot's eye position
        gaze_direction = target_position - eye_pos
        gaze_direction = gaze_direction / np.linalg.norm(gaze_direction)

        # Convert to head joint angles
        pitch = np.arcsin(gaze_direction[2])  # Up/down
        yaw = np.arctan2(gaze_direction[1], gaze_direction[0])  # Left/right

        # Move eyes and head
        self.move_eyes(yaw, pitch)
        self.move_head([pitch, yaw, 0.0])

    def initiate_greeting(self, human_info):
        """
        Initiate greeting sequence
        """
        # Play greeting gesture
        self.gesture_system.play_greeting()

        # Speak greeting
        greeting_text = self.generate_greeting(human_info)
        self.speech_synthesizer.speak(greeting_text)

        # Maintain eye contact
        self.maintain_eye_contact(human_info['position'])

    def generate_greeting(self, human_info):
        """
        Generate appropriate greeting based on context
        """
        # Time-based greeting
        current_hour = time.localtime().tm_hour
        if current_hour < 12:
            time_greeting = "Good morning"
        elif current_hour < 18:
            time_greeting = "Good afternoon"
        else:
            time_greeting = "Good evening"

        return f"{time_greeting}, I am your humanoid assistant. How can I help you today?"

    def maintain_eye_contact(self, human_position, duration=2.0):
        """
        Maintain eye contact for specified duration
        """
        start_time = time.time()
        while time.time() - start_time < duration:
            self.control_gaze(human_position)
            time.sleep(0.1)  # Update every 100ms

    def move_head(self, joint_angles):
        """Move head joints"""
        # Interface with actual head controller
        pass

    def move_eyes(self, horizontal, vertical):
        """Move eye joints"""
        # Interface with actual eye controller
        pass

    def move_torso(self, joint_angles):
        """Move torso joints"""
        # Interface with actual torso controller
        pass

    def get_eye_position(self):
        """Get current eye position"""
        # Return robot's eye position in world coordinates
        return np.array([0.0, 0.0, 1.5])  # Placeholder

class GestureSystem:
    def __init__(self):
        self.gesture_library = self.load_gesture_library()

    def load_gesture_library(self):
        """Load predefined gestures"""
        return {
            'greeting': self.greeting_gesture,
            'acknowledgment': self.acknowledgment_gesture,
            'pointing': self.pointing_gesture,
            'waving': self.waving_gesture
        }

    def greeting_gesture(self):
        """Wave with right arm"""
        # Define joint trajectory for greeting wave
        wave_trajectory = [
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # Arm raised
            [0.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # Wave position 1
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # Wave position 2
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # Return to neutral
        ]

        # Execute trajectory
        for joint_state in wave_trajectory:
            self.move_arm_to_state(joint_state)
            time.sleep(0.5)

    def acknowledgment_gesture(self):
        """Nod head to acknowledge"""
        # Head nod trajectory
        nod_trajectory = [
            [0.0, 0.0, 0.0],  # Neutral
            [0.1, 0.0, 0.0],  # Nod down
            [0.0, 0.0, 0.0],  # Return to neutral
            [0.1, 0.0, 0.0],  # Nod down again
            [0.0, 0.0, 0.0]   # Return to neutral
        ]

        for joint_state in nod_trajectory:
            self.move_head_to_state(joint_state)
            time.sleep(0.3)

    def pointing_gesture(self, target_direction):
        """Point to a target direction"""
        # Calculate arm angles to point in target direction
        # This would involve IK to point to specific location
        pass

    def waving_gesture(self):
        """Continuous waving motion"""
        # Similar to greeting but repeated
        pass

    def move_arm_to_state(self, joint_angles):
        """Move arm to specific joint configuration"""
        pass

    def move_head_to_state(self, joint_angles):
        """Move head to specific joint configuration"""
        pass

class SpeechSynthesizer:
    def __init__(self):
        self.voice_params = {
            'pitch': 1.0,
            'speed': 1.0,
            'volume': 0.8
        }

    def speak(self, text):
        """Synthesize and play speech"""
        # This would interface with actual TTS system
        print(f"Robot says: {text}")

    def set_voice_parameters(self, pitch=None, speed=None, volume=None):
        """Adjust voice characteristics"""
        if pitch is not None:
            self.voice_params['pitch'] = pitch
        if speed is not None:
            self.voice_params['speed'] = speed
        if volume is not None:
            self.voice_params['volume'] = volume
```

### Day 9: Advanced Interaction Scenarios

#### Multi-Modal Interaction

Combine multiple interaction modalities for natural communication:

```python
# multi_modal_interaction.py
import numpy as np
import threading
import queue
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class InteractionEvent:
    """Represents an interaction event"""
    event_type: str  # 'speech', 'gesture', 'gaze', 'touch', 'proximity'
    timestamp: float
    data: Dict
    confidence: float = 1.0

class MultiModalInteractionManager:
    def __init__(self):
        self.event_queue = queue.Queue()
        self.event_handlers = {
            'speech': self.handle_speech_event,
            'gesture': self.handle_gesture_event,
            'gaze': self.handle_gaze_event,
            'proximity': self.handle_proximity_event
        }

        self.interaction_context = {
            'current_speaker': None,
            'attention_focus': None,
            'interaction_history': [],
            'social_context': {}
        }

        self.speech_recognizer = SpeechRecognizer()
        self.gesture_detector = GestureDetector()
        self.gaze_tracker = GazeTracker()

    def start_interaction_loop(self):
        """Start the main interaction processing loop"""
        # Start sensor processing threads
        threading.Thread(target=self.process_speech, daemon=True).start()
        threading.Thread(target=self.process_gestures, daemon=True).start()
        threading.Thread(target=self.process_gaze, daemon=True).start()
        threading.Thread(target=self.process_proximity, daemon=True).start()

        # Main event processing loop
        while True:
            try:
                event = self.event_queue.get(timeout=1.0)
                self.process_interaction_event(event)
            except queue.Empty:
                continue

    def process_interaction_event(self, event: InteractionEvent):
        """Process an interaction event based on its type"""
        if event.event_type in self.event_handlers:
            handler = self.event_handlers[event.event_type]
            handler(event)
        else:
            print(f"Unknown event type: {event.event_type}")

    def handle_speech_event(self, event: InteractionEvent):
        """Handle speech input event"""
        speech_data = event.data
        text = speech_data.get('text', '')
        speaker_id = speech_data.get('speaker_id')

        # Update interaction context
        self.interaction_context['current_speaker'] = speaker_id
        self.interaction_context['attention_focus'] = speaker_id

        # Process the speech command
        self.process_speech_command(text, speaker_id)

        # Add to interaction history
        self.interaction_context['interaction_history'].append({
            'type': 'speech',
            'text': text,
            'speaker': speaker_id,
            'timestamp': event.timestamp
        })

    def handle_gesture_event(self, event: InteractionEvent):
        """Handle gesture input event"""
        gesture_data = event.data
        gesture_type = gesture_data.get('gesture_type')
        performer_id = gesture_data.get('performer_id')

        # Process the gesture
        self.process_gesture(gesture_type, performer_id)

        # Add to interaction history
        self.interaction_context['interaction_history'].append({
            'type': 'gesture',
            'gesture': gesture_type,
            'performer': performer_id,
            'timestamp': event.timestamp
        })

    def handle_gaze_event(self, event: InteractionEvent):
        """Handle gaze tracking event"""
        gaze_data = event.data
        target = gaze_data.get('target')
        duration = gaze_data.get('duration', 0)

        # Update attention focus
        if duration > 0.5:  # Only update for sustained gaze
            self.interaction_context['attention_focus'] = target

    def handle_proximity_event(self, event: InteractionEvent):
        """Handle proximity detection event"""
        proximity_data = event.data
        person_id = proximity_data.get('person_id')
        distance = proximity_data.get('distance')

        if distance < 2.0:  # Within 2 meters
            # Person is close enough for interaction
            self.initiate_interaction(person_id)
        elif distance > 3.0:  # Moved away
            # Person is too far for interaction
            self.terminate_interaction(person_id)

    def process_speech_command(self, text: str, speaker_id: str):
        """Process natural language commands"""
        # Simple command parsing (in reality, this would use NLP)
        text_lower = text.lower()

        if 'hello' in text_lower or 'hi' in text_lower:
            self.respond_greeting(speaker_id)
        elif 'help' in text_lower or 'assist' in text_lower:
            self.offer_assistance(speaker_id)
        elif 'move' in text_lower or 'walk' in text_lower:
            self.process_navigation_command(text_lower, speaker_id)
        elif 'pick' in text_lower or 'grasp' in text_lower:
            self.process_manipulation_command(text_lower, speaker_id)
        else:
            self.process_general_command(text_lower, speaker_id)

    def process_gesture(self, gesture_type: str, performer_id: str):
        """Process gesture commands"""
        if gesture_type == 'pointing':
            # Robot should look in the direction pointed
            self.orient_to_pointing_direction()
        elif gesture_type == 'beckoning':
            # Person is calling the robot
            self.move_towards_performer(performer_id)
        elif gesture_type == 'waving':
            # Wave back as acknowledgment
            self.wave_back(performer_id)

    def respond_greeting(self, speaker_id: str):
        """Respond to greeting with appropriate social behavior"""
        controller = HumanoidInteractionController()
        controller.initiate_greeting({'position': self.get_person_position(speaker_id)})

    def offer_assistance(self, speaker_id: str):
        """Offer assistance to the person"""
        # Move closer if not already near
        person_pos = self.get_person_position(speaker_id)
        if self.distance_to_robot(person_pos) > 1.0:
            self.move_to_position(person_pos)

        # Offer help with speech and gesture
        controller = HumanoidInteractionController()
        controller.speech_synthesizer.speak(
            "I'm here to help. What would you like me to do?"
        )
        controller.gesture_system.acknowledgment_gesture()

    def process_navigation_command(self, text: str, speaker_id: str):
        """Process navigation-related commands"""
        # Extract destination from text
        if 'kitchen' in text:
            destination = 'kitchen'
        elif 'living room' in text:
            destination = 'living_room'
        elif 'bedroom' in text:
            destination = 'bedroom'
        else:
            destination = None

        if destination:
            self.navigate_to_location(destination)

    def process_manipulation_command(self, text: str, speaker_id: str):
        """Process manipulation-related commands"""
        # Extract object and action from text
        if 'cup' in text or 'bottle' in text:
            object_type = 'cylindrical'
            action = 'pick' if 'pick' in text else 'grasp'
        else:
            object_type = None
            action = None

        if object_type and action:
            self.execute_manipulation_task(object_type, action)

    def process_general_command(self, text: str, speaker_id: str):
        """Process other commands"""
        # Use more sophisticated NLP in real implementation
        self.respond_general_query(text)

    def initiate_interaction(self, person_id: str):
        """Initiate interaction when person comes close"""
        # Start tracking the person
        self.start_tracking_person(person_id)

        # If this is the first time seeing the person, greet them
        if person_id not in self.interaction_context['social_context']:
            self.interaction_context['social_context'][person_id] = {
                'first_seen': time.time(),
                'interaction_count': 0
            }
            self.respond_greeting(person_id)

    def terminate_interaction(self, person_id: str):
        """Terminate interaction when person moves away"""
        # Stop tracking the person
        self.stop_tracking_person(person_id)

    def process_speech(self):
        """Continuously process speech input"""
        while True:
            speech_result = self.speech_recognizer.listen()
            if speech_result and speech_result.confidence > 0.7:
                event = InteractionEvent(
                    event_type='speech',
                    timestamp=time.time(),
                    data=speech_result.data,
                    confidence=speech_result.confidence
                )
                self.event_queue.put(event)

    def process_gestures(self):
        """Continuously process gesture input"""
        while True:
            gesture_result = self.gesture_detector.detect()
            if gesture_result:
                event = InteractionEvent(
                    event_type='gesture',
                    timestamp=time.time(),
                    data=gesture_result,
                    confidence=0.9  # Gesture detection confidence
                )
                self.event_queue.put(event)

    def process_gaze(self):
        """Continuously process gaze tracking"""
        while True:
            gaze_result = self.gaze_tracker.track()
            if gaze_result:
                event = InteractionEvent(
                    event_type='gaze',
                    timestamp=time.time(),
                    data=gaze_result,
                    confidence=0.8
                )
                self.event_queue.put(event)

    def process_proximity(self):
        """Continuously process proximity detection"""
        while True:
            proximity_result = self.detect_proximity()
            if proximity_result:
                event = InteractionEvent(
                    event_type='proximity',
                    timestamp=time.time(),
                    data=proximity_result,
                    confidence=0.95
                )
                self.event_queue.put(event)

    def get_person_position(self, person_id: str):
        """Get the current position of a person"""
        # This would interface with person tracking system
        return np.array([1.0, 0.0, 0.0])  # Placeholder

    def distance_to_robot(self, person_pos):
        """Calculate distance from person to robot"""
        robot_pos = np.array([0.0, 0.0, 0.0])
        return np.linalg.norm(person_pos - robot_pos)

    def move_to_position(self, target_pos):
        """Move robot to target position"""
        # This would interface with navigation system
        pass

    def navigate_to_location(self, location: str):
        """Navigate to predefined location"""
        # This would use navigation system to go to location
        pass

    def execute_manipulation_task(self, object_type: str, action: str):
        """Execute manipulation task"""
        controller = ObjectManipulationController()
        # This would execute the specific manipulation task
        pass

    def respond_general_query(self, text: str):
        """Respond to general queries"""
        response = f"I heard you say: {text}. How can I assist you further?"
        controller = HumanoidInteractionController()
        controller.speech_synthesizer.speak(response)

class SpeechRecognizer:
    def listen(self):
        """Listen for speech and return recognition result"""
        # Placeholder for actual speech recognition
        # In reality, this would interface with speech recognition API
        return None

class GestureDetector:
    def detect(self):
        """Detect gestures from camera input"""
        # Placeholder for actual gesture detection
        return None

class GazeTracker:
    def track(self):
        """Track gaze direction"""
        # Placeholder for actual gaze tracking
        return None

# Usage example
def run_multi_modal_interaction():
    """Run the multi-modal interaction system"""
    interaction_manager = MultiModalInteractionManager()

    print("Starting multi-modal interaction system...")
    print("The system will respond to speech, gestures, gaze, and proximity")

    # Start the interaction loop (this would run continuously)
    # interaction_manager.start_interaction_loop()
```

### Day 10: Capstone Integration - Humanoid Robot Control

#### Exercise: Complete Humanoid System Integration

Integrate all humanoid components into a working system:

```python
# humanoid_system_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
import numpy as np

class HumanoidSystemIntegrator(Node):
    def __init__(self):
        super().__init__('humanoid_system_integrator')

        # Initialize all subsystems
        self.kinematics = HumanoidKinematics()
        self.ik_solver = HumanoidInverseKinematics(None)
        self.balance_controller = BalanceController()
        self.zmp_controller = ZMPController(robot_mass=60.0)
        self.interaction_controller = HumanoidInteractionController()
        self.manipulation_planner = HumanoidManipulationPlanner(None)

        # ROS publishers and subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.voice_cmd_sub = self.create_subscription(
            String, '/voice_command', self.voice_command_callback, 10)

        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)

        # Robot state
        self.current_joint_states = None
        self.imu_data = None
        self.scan_data = None
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta

        # Control loop timer
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100Hz

        self.get_logger().info("Humanoid System Integrator initialized")

    def joint_state_callback(self, msg):
        """Handle joint state updates"""
        self.current_joint_states = msg

    def imu_callback(self, msg):
        """Handle IMU data for balance control"""
        self.imu_data = msg

    def scan_callback(self, msg):
        """Handle laser scan data for navigation"""
        self.scan_data = msg

    def voice_command_callback(self, msg):
        """Handle voice commands"""
        self.process_voice_command(msg.data)

    def control_loop(self):
        """Main control loop running at 100Hz"""
        if self.current_joint_states is None:
            return

        # Get current robot state
        current_com = self.calculate_current_com()
        current_com_vel = self.calculate_com_velocity()
        current_com_acc = self.calculate_com_acceleration()

        # Balance control
        balance_torques = self.balance_controller.update_balance(
            current_com,
            self.get_desired_com(),
            current_com_vel,
            self.get_desired_zmp()
        )

        # ZMP control
        current_zmp = self.zmp_controller.calculate_zmp(
            current_com,
            current_com_vel,
            current_com_acc
        )

        # Check stability
        support_polygon = self.get_support_polygon()
        is_stable = self.zmp_controller.zmp_stability_check(current_zmp, support_polygon)

        # If unstable, adjust balance
        if not is_stable:
            self.get_logger().warn("Robot is unstable, adjusting balance")
            # Apply emergency balance control
            self.apply_emergency_balance_control()

        # Send joint commands
        self.send_joint_commands(balance_torques)

    def calculate_current_com(self):
        """Calculate current center of mass"""
        # This would use forward kinematics and mass properties
        # For now, return a placeholder
        return np.array([0.0, 0.0, 0.8])

    def calculate_com_velocity(self):
        """Calculate CoM velocity"""
        # Would calculate from joint velocities
        return np.array([0.0, 0.0, 0.0])

    def calculate_com_acceleration(self):
        """Calculate CoM acceleration"""
        # Would calculate from joint accelerations
        return np.array([0.0, 0.0, 0.0])

    def get_desired_com(self):
        """Get desired center of mass position"""
        # This would come from walking pattern generator
        return np.array([0.0, 0.0, 0.8])

    def get_desired_zmp(self):
        """Get desired Zero Moment Point"""
        # This would come from walking pattern
        return np.array([0.0, 0.0, 0.0])

    def get_support_polygon(self):
        """Get current support polygon vertices"""
        # This would calculate from foot positions
        # For bipedal stance, return rectangle between feet
        foot_separation = 0.2  # 20cm between feet
        return [
            (-0.1, -foot_separation/2),  # front left
            (0.1, -foot_separation/2),   # front right
            (0.1, foot_separation/2),    # back right
            (-0.1, foot_separation/2)    # back left
        ]

    def apply_emergency_balance_control(self):
        """Apply emergency balance control"""
        # Emergency balance strategy
        # This might involve widening stance, crouching, etc.
        pass

    def send_joint_commands(self, torques):
        """Send joint commands to robot"""
        if self.current_joint_states is None:
            return

        # Create joint command message
        cmd_msg = JointState()
        cmd_msg.name = self.current_joint_states.name
        cmd_msg.position = self.current_joint_states.position  # Keep current positions
        cmd_msg.velocity = [0.0] * len(self.current_joint_states.name)  # Zero velocity
        cmd_msg.effort = torques  # Apply calculated torques

        self.joint_cmd_pub.publish(cmd_msg)

    def process_voice_command(self, command):
        """Process voice commands and trigger appropriate behaviors"""
        command_lower = command.lower()

        if 'walk' in command_lower or 'move' in command_lower:
            self.execute_locomotion_command(command_lower)
        elif 'grasp' in command_lower or 'pick' in command_lower:
            self.execute_manipulation_command(command_lower)
        elif 'dance' in command_lower or 'move' in command_lower:
            self.execute_behavior_command(command_lower)
        else:
            self.interaction_controller.speech_synthesizer.speak(
                f"I heard {command}, but I'm not sure how to respond. Can you rephrase?"
            )

    def execute_locomotion_command(self, command):
        """Execute locomotion commands"""
        # Parse destination or direction
        if 'forward' in command or 'straight' in command:
            self.move_forward()
        elif 'backward' in command or 'back' in command:
            self.move_backward()
        elif 'left' in command:
            self.turn_left()
        elif 'right' in command:
            self.turn_right()
        else:
            # Try to extract destination
            self.navigate_to_destination(command)

    def execute_manipulation_command(self, command):
        """Execute manipulation commands"""
        # Parse object and action
        if 'cup' in command or 'bottle' in command:
            self.grasp_object('cylindrical')
        elif 'book' in command or 'box' in command:
            self.grasp_object('rectangular')
        else:
            self.interaction_controller.speech_synthesizer.speak(
                "I'm not sure what object you want me to grasp."
            )

    def execute_behavior_command(self, command):
        """Execute behavior commands"""
        if 'wave' in command:
            self.interaction_controller.gesture_system.waving_gesture()
        elif 'nod' in command:
            self.interaction_controller.gesture_system.acknowledgment_gesture()
        elif 'dance' in command:
            self.perform_dance_sequence()

    def move_forward(self):
        """Move robot forward"""
        cmd = Twist()
        cmd.linear.x = 0.2  # 20 cm/s
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    def move_backward(self):
        """Move robot backward"""
        cmd = Twist()
        cmd.linear.x = -0.2  # -20 cm/s
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    def turn_left(self):
        """Turn robot left"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.5  # 0.5 rad/s
        self.cmd_vel_pub.publish(cmd)

    def turn_right(self):
        """Turn robot right"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = -0.5  # -0.5 rad/s
        self.cmd_vel_pub.publish(cmd)

    def navigate_to_destination(self, command):
        """Navigate to specified destination"""
        # This would use navigation stack
        # For now, just acknowledge
        self.interaction_controller.speech_synthesizer.speak(
            "I will navigate to the specified location."
        )

    def grasp_object(self, object_type):
        """Grasp an object of specified type"""
        controller = ObjectManipulationController()
        controller.pick_and_place(
            object_pose=np.array([0.5, 0.0, 0.1, 0, 0, 0, 1]),  # Position and orientation
            target_pose=np.array([0.2, 0.3, 0.1, 0, 0, 0, 1])
        )

    def perform_dance_sequence(self):
        """Perform a simple dance sequence"""
        self.interaction_controller.speech_synthesizer.speak("Here's a little dance for you!")
        # Execute a simple dance pattern using joint commands

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidSystemIntegrator()

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

### Week 11 Assessment
1. **Technical Skills**: Implement inverse kinematics for a humanoid arm
2. **Understanding**: Explain the difference between static and dynamic walking
3. **Application**: Create a balance controller using ZMP theory

### Week 12 Assessment
1. **Integration**: Combine manipulation and interaction systems
2. **Problem Solving**: Plan a dual-arm manipulation task
3. **Analysis**: Evaluate the stability of a humanoid grasp

## Resources and Further Reading

### Required Reading
- "Humanoid Robotics: A Reference" by Alimoto et al.
- "Introduction to Humanoid Robotics" by Hashimoto et al.
- "Robotics: Modelling, Planning and Control" by Siciliano et al.

### Recommended Resources
- ROS 2 Control documentation: https://control.ros.org/
- MoveIt! Motion Planning Framework: https://moveit.ros.org/
- Humanoid Path Planner (HPP): https://humanoid-path-planner.github.io/hpp-doc/

## Next Steps

After completing Weeks 11-12, you'll have mastered humanoid robot development including kinematics, locomotion, manipulation, and interaction. In the final module (Week 13), we'll focus on conversational robotics, integrating GPT models and multi-modal interaction for natural human-robot communication.

The next module will cover:
- Integrating GPT models for conversational AI in robots
- Speech recognition and natural language understanding
- Multi-modal interaction: speech, gesture, vision
- Advanced conversational robotics applications

Continue to the [Week 13: Conversational Robotics](/docs/weekly-breakdown/week-13-conversational-robotics) module to complete your journey in Physical AI & Humanoid Robotics.