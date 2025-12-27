---
title: "Hardware Requirements"
description: "Detailed hardware requirements for the Physical AI & Humanoid Robotics course including workstation, edge computing, and robot platforms"
keywords: ["hardware", "requirements", "robotics", "nvidia", "isaac", "jetson", "workstation"]
sidebar_position: 7
---

# Hardware Requirements

This document outlines the hardware requirements for the Physical AI & Humanoid Robotics course. The course sits at the intersection of three computationally intensive domains: Physics Simulation (Isaac Sim/Gazebo), Visual Perception (SLAM/Computer Vision), and Generative AI (LLMs/VLA). As such, specific hardware requirements must be met to ensure a smooth learning experience.

## Overview

The hardware requirements are divided into three main categories:

1. **The "Digital Twin" Workstation** - Required for each student to run simulations and AI models
2. **The "Physical AI" Edge Kit** - For deploying code to edge devices and learning real-world constraints
3. **The Robot Lab** - Options for hands-on experience with physical robots

## 1. The "Digital Twin" Workstation (Required per Student)

This is the most critical component for the course. NVIDIA Isaac Sim is an Omniverse application that requires "RTX" (Ray Tracing) capabilities. Standard laptops (MacBooks or non-RTX Windows machines) will not work effectively.

### Minimum Specifications

- **GPU**: NVIDIA RTX 4070 Ti (12GB VRAM) or higher
  - **Why**: You need high VRAM to load the USD (Universal Scene Description) assets for the robot and environment, plus run the VLA (Vision-Language-Action) models simultaneously
  - **Ideal**: RTX 3090 or 4090 (24GB VRAM) allows for smoother "Sim-to-Real" training

- **CPU**: Intel Core i7 (13th Gen+) or AMD Ryzen 9
  - **Why**: Physics calculations (Rigid Body Dynamics) in Gazebo/Isaac are CPU-intensive

- **RAM**: 64 GB DDR5 (32 GB is the absolute minimum, but will crash during complex scene rendering)

- **OS**: Ubuntu 22.04 LTS
  - **Note**: While Isaac Sim runs on Windows, ROS 2 (Humble/Iron) is native to Linux. Dual-booting or dedicated Linux machines are mandatory for a friction-free experience

- **Storage**: 2TB NVMe SSD
  - **Why**: Isaac Sim assets, simulation environments, and training data require significant storage

### Recommended Specifications

For optimal performance, we recommend:

- **GPU**: NVIDIA RTX 4090 (24GB VRAM) or RTX 6000 Ada Generation (48GB VRAM)
- **CPU**: Intel i9-13900K or AMD Ryzen 9 7950X
- **RAM**: 128 GB DDR5
- **Storage**: 4TB+ NVMe SSD array
- **Additional**: 10GbE network for large asset streaming

### Alternative: Cloud-Based Workstations

For institutions or students unable to invest in high-end hardware:

- **AWS**: g5.2xlarge (A10G GPU, 24GB VRAM) or g6e.xlarge
- **Software**: NVIDIA Isaac Sim on Omniverse Cloud
- **Cost**: ~$1.50/hour for spot instances
- **Considerations**: Latency may impact real-time control tasks

## 2. The "Physical AI" Edge Kit

Since a full humanoid robot is expensive, students learn "Physical AI" by setting up the nervous system on a desk before deploying it to a robot. This kit covers Module 3 (Isaac ROS) and Module 4 (VLA).

### The Brain

- **NVIDIA Jetson Orin Nano (8GB) or Orin NX (16GB)**
  - **Role**: This is the industry standard for embodied AI. Students will deploy their ROS 2 nodes here to understand resource constraints vs. their powerful workstations
  - **Cost**: ~$400 (Nano) to ~$600 (NX)

### The Eyes (Vision)

- **Intel RealSense D435i or D455**
  - **Role**: Provides RGB (Color) and Depth (Distance) data. Essential for the VSLAM and Perception modules
  - **Why D435i**: Includes IMU (essential for SLAM) - Do NOT buy the D435 (non-i)
  - **Cost**: ~$350

### The Inner Ear (Balance)

- **Generic USB IMU (BNO055)** (Often built into the RealSense D435i or Jetson boards, but a separate module helps teach IMU calibration)
- **Cost**: ~$15-30

### Voice Interface

- **ReSpeaker USB Mic Array v2.0** for the "Voice-to-Action" Whisper integration
  - **Why**: Far-field microphone for voice commands
  - **Cost**: ~$70

### Complete Edge Kit Cost

- **Basic Kit**: ~$850 (Orin Nano + RealSense + ReSpeaker)
- **Premium Kit**: ~$1,200 (Orin NX + RealSense + ReSpeaker + additional sensors)

## 3. The Robot Lab Options

For the "Physical" part of the course, you have three tiers of options depending on budget.

### Option A: The "Proxy" Approach (Recommended for Budget)

Use a quadruped (dog) or a robotic arm as a proxy. The software principles (ROS 2, VSLAM, Isaac Sim) transfer 90% effectively to humanoids.

- **Robot**: Unitree Go2 Edu (~$1,800 - $3,000)
  - **Pros**: Highly durable, excellent ROS 2 support, affordable enough to have multiple units
  - **Cons**: Not a biped (humanoid)

### Option B: The "Miniature Humanoid" Approach

Small, table-top humanoids.

- **Robot**: Unitree G1 (~$16,000) or Robotis OP3 (older, but stable, ~$12,000)
  - **Budget Alternative**: Hiwonder TonyPi Pro (~$600)
  - **Warning**: The cheap kits (Hiwonder) usually run on Raspberry Pi, which cannot run NVIDIA Isaac ROS efficiently. You would use these only for kinematics (walking) and use the Jetson kits for AI.

### Option C: The "Premium" Lab (Sim-to-Real specific)

If the goal is to actually deploy the Capstone to a real humanoid:

- **Robot**: Unitree G1 Humanoid
  - **Why**: It is one of the few commercially available humanoids that can actually walk dynamically and has an SDK open enough for students to inject their own ROS 2 controllers
  - **Cost**: ~$20,000

## Architecture Summary

Your lab infrastructure should look like this:

| Component | Hardware | Function |
|-----------|----------|----------|
| Sim Rig | PC with RTX 4080 + Ubuntu 22.04 | Runs Isaac Sim, Gazebo, Unity, and trains LLM/VLA models |
| Edge Brain | Jetson Orin Nano | Runs the "Inference" stack. Students deploy their code here |
| Sensors | RealSense Camera + IMU | Connected to the Jetson to feed real-world data to the AI |
| Actuator | Unitree Go2 or G1 (Shared) | Receives motor commands from the Jetson |

## Tiered Implementation Strategy

### Basic Setup (Minimum Viable Lab)

For institutions starting out:

1. **1-2 High-End Workstations**: Shared among students for simulation
2. **3-5 Jetson Kits**: Each with RealSense and ReSpeaker
3. **2-3 Go2 Robots**: Shared for hands-on experience

**Total Cost**: ~$20,000-30,000

### Full Implementation (Individual Student Setup)

For comprehensive learning:

1. **1 Workstation per 2-3 students**: High-end simulation rigs
2. **1 Jetson Kit per student**: Individual edge deployment experience
3. **1 Robot per 4-6 students**: Hands-on experience

**Total Cost**: ~$5,000-8,000 per student capacity

### Premium Implementation (Advanced Research)

For research-focused programs:

1. **1 Workstation per student**: High-end individual simulation
2. **1 Jetson Kit per student**: Individual edge deployment
3. **Humanoid robots**: Unitree G1 or similar for each group

**Total Cost**: ~$25,000-30,000 per student capacity

## Cost Considerations

### On-Premise vs Cloud

**Option 1: High CapEx: The "Physical" Lab (On-Premise)**
- Advantages: No ongoing costs, predictable pricing, full control
- Disadvantages: High upfront investment, maintenance responsibility

**Option 2: High OpEx: The "Ether" Lab (Cloud-Native)**
- Advantages: Rapid deployment, no hardware maintenance, scalable
- Disadvantages: Ongoing costs, potential latency issues, network dependency

### The Latency Trap (Hidden Cost)

Simulating in the cloud works well, but controlling a real robot from a cloud instance is dangerous due to latency.

**Solution**: Students train in the Cloud, download the model (weights), and flash it to the local Jetson kit.

## Student Kit Recommendations

### The Economy Jetson Student Kit

Best for: Learning ROS 2, Basic Computer Vision, and Sim-to-Real control.

| Component | Model | Price (Approx.) | Notes |
|-----------|-------|----------------|-------|
| The Brain | NVIDIA Jetson Orin Nano Super Dev Kit (8GB) | $249 | New official MSRP (Price dropped from ~$499). Capable of 40 TOPS. |
| The Eyes | Intel RealSense D435i | $349 | Includes IMU (essential for SLAM). Do not buy the D435 (non-i). |
| The Ears | ReSpeaker USB Mic Array v2.0 | $69 | Far-field microphone for voice commands (Module 4). |
| Wi-Fi | (Included in Dev Kit) | $0 | The new "Super" kit includes the Wi-Fi module pre-installed. |
| Power/Misc | SD Card (128GB) + Jumper Wires | $30 | High-endurance microSD card required for the OS. |
| **TOTAL** | | **~$700 per kit** | |

## Procurement Recommendations

### Vendor Relationships

Consider establishing relationships with:
- **NVIDIA**: For educational discounts on GPUs and Jetson kits
- **Unitree**: For educational pricing on robots
- **Intel**: For RealSense camera bundles
- **Local System Integrators**: For custom workstation builds

### Phased Rollout

1. **Phase 1**: Deploy 2-3 workstations and 5 Jetson kits for pilot program
2. **Phase 2**: Expand to 1 robot platform for hands-on experience
3. **Phase 3**: Scale to full classroom implementation based on usage patterns

## Maintenance and Support

### Hardware Support Plan

- **Warranty**: Ensure 3-5 year warranty on major components
- **Spare Parts**: Maintain 10% spare inventory of critical components
- **Technical Support**: Establish support contracts for complex hardware (workstations, robots)
- **Student Training**: Train students in basic troubleshooting and maintenance

### Upgrades and Refresh Cycles

- **Workstations**: Plan for refresh every 4-5 years
- **GPUs**: Monitor for new RTX generations (typically 2-year cycles)
- **Robots**: Consider that robot platforms have 5-7 year lifecycles
- **Edge Kits**: Refresh every 3-4 years as AI demands increase

## Conclusion

The Physical AI & Humanoid Robotics course requires significant hardware investment, but this investment enables students to work with cutting-edge technology at the intersection of AI and robotics. The modular approach allows institutions to scale from basic to premium implementations based on budget and learning objectives.

Remember: The goal is not just to teach students about robotics, but to prepare them for careers in the rapidly evolving field of Physical AI, where simulation and real-world deployment go hand in hand.