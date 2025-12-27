# Feature Specification: Book Master Plan for Physical AI & Humanoid Robotics Course

**Feature Branch**: `1-book-master-plan`
**Created**: 2025-12-27
**Status**: Draft
**Input**: User description: "book-master-plan
**GOAL: Generate Initial Book Plan, Docusaurus Setup, and Design Specifications**

**Phase 1: Book Plan & Structure**

1. **Book Structure:** Develop a detailed **Book Plan**. Outline the main **Parts**, **Chapters**, and key **Sections** for the entire documentation. Ensure a logical flow from foundational concepts to advanced application, strictly following the attached 'Course Outline' as the primary source of truth for content sequence.

2. **Content Goals:** Define the **target audience**, the **primary learning objective** of the book, and the **overall tone** (e.g., tutorial, reference, deep dive).

3. **Draft TOC:** Based on the outline, generate a preliminary **Table of Contents (TOC)** structure.

---

**Phase 2: Docusaurus Project Setup & Design**

4. **Project Setup:** Detail the requirements for the initial **Docusaurus project setup**. Include specific requirements for:

    * Core dependencies (if any specific ones are needed).

    * Configuration file (`docusaurus.config.js`) structure (e.g., title, tagline, URL).

    * Initial file paths and folder structure for docs.

5. **Book Layout & Design:** Define the initial **book layout** and **design principles**. Specify requirements for:

    * **Sidebar structure** (must reflect the Draft TOC).

    * **Navigation tabs** (Home, Docs, Blog, etc.).

    * **Color Palette** and basic **CSS/Theme** requirements (e.g., dark mode compatibility).

    * Requirements for any necessary custom **React components** (like callouts, specific code block styles, etc.).

---

**CONTEXT: Attached Course Outline**

The Course Details

Physical AI & Humanoid Robotics

Focus and Theme: AI Systems in the Physical World. Embodied Intelligence.

Goal: Bridging the gap between the digital brain and the physical body. Students apply their AI knowledge to control Humanoid Robots in simulated and real-world environments.

Quarter Overview

The future of AI extends beyond digital spaces into the physical world. This capstone quarter introduces Physical AI—AI systems that function in reality and comprehend physical laws. Students learn to design, simulate, and deploy humanoid robots capable of natural human interactions using ROS 2, Gazebo, and NVIDIA Isaac.

Module 1: The Robotic Nervous System (ROS 2)

Focus: Middleware for robot control.

ROS 2 Nodes, Topics, and Services.

Bridging Python Agents to ROS controllers using rclpy.

Understanding URDF (Unified Robot Description Format) for humanoids.

Module 2: The Digital Twin (Gazebo & Unity)

Focus: Physics simulation and environment building.

Simulating physics, gravity, and collisions in Gazebo.

High-fidelity rendering and human-robot interaction in Unity.

Simulating sensors: LiDAR, Depth Cameras, and IMUs.

Module 3: The AI-Robot Brain (NVIDIA Isaac™)

Focus: Advanced perception and training.

NVIDIA Isaac Sim: Photorealistic simulation and synthetic data generation.

Isaac ROS: Hardware-accelerated VSLAM (Visual SLAM) and navigation.

Nav2: Path planning for bipedal humanoid movement.

Module 4: Vision-Language-Action (VLA)

Focus: The convergence of LLMs and Robotics.

Voice-to-Action: Using OpenAI Whisper for voice commands.

Cognitive Planning: Using LLMs to translate natural language ("Clean the room") into a sequence of ROS 2 actions.

Capstone Project: The Autonomous Humanoid. A final project where a simulated robot receives a voice command, plans a path, navigates obstacles, identifies an object using computer vision, and manipulates it.

Why Physical AI Matters

Humanoid robots are poised to excel in our human-centered world because they share our physical form and can be trained with abundant data from interacting in human environments. This represents a significant transition from AI models confined to digital environments to embodied intelligence that operates in physical space.

Learning Outcomes

Understand Physical AI principles and embodied intelligence

Master ROS 2 (Robot Operating System) for robotic control

Simulate robots with Gazebo and Unity

Develop with NVIDIA Isaac AI robot platform

Design humanoid robots for natural interactions

Integrate GPT models for conversational robotics

Weekly Breakdown

Weeks 1-2: Introduction to Physical AI

Foundations of Physical AI and embodied intelligence

From digital AI to robots that understand physical laws

Overview of humanoid robotics landscape

Sensor systems: LIDAR, cameras, IMUs, force/torque sensors

Weeks 3-5: ROS 2 Fundamentals

ROS 2 architecture and core concepts

Nodes, topics, services, and actions

Building ROS 2 packages with Python

Launch files and parameter management

Weeks 6-7: Robot Simulation with Gazebo

Gazebo simulation environment setup

URDF and SDF robot description formats

Physics simulation and sensor simulation

Introduction to Unity for robot visualization

Weeks 8-10: NVIDIA Isaac Platform

NVIDIA Isaac SDK and Isaac Sim

AI-powered perception and manipulation

Reinforcement learning for robot control

Sim-to-real transfer techniques

Weeks 11-12: Humanoid Robot Development

Humanoid robot kinematics and dynamics

Bipedal locomotion and balance control

Manipulation and grasping with humanoid hands

Natural human-robot interaction design

Week 13: Conversational Robotics

Integrating GPT models for conversational AI in robots

Speech recognition and natural language understanding

Multi-modal interaction: speech, gesture, vision

Assessments

ROS 2 package development project

Gazebo simulation implementation

Isaac-based perception pipeline

Capstone: Simulated humanoid robot with conversational AI

Hardware Requirements

This course is technically demanding. It sits at the intersection of three heavy computational loads: Physics Simulation (Isaac Sim/Gazebo), Visual Perception (SLAM/Computer Vision), and Generative AI (LLMs/VLA).

Because the capstone involves a "Simulated Humanoid," the primary investment must be in High-Performance Workstations. However, to fulfill the "Physical AI" promise, you also need Edge Computing Kits (brains without bodies) or specific robot hardware.

1. The "Digital Twin" Workstation (Required per Student)

This is the most critical component. NVIDIA Isaac Sim is an Omniverse application that requires "RTX" (Ray Tracing) capabilities. Standard laptops (MacBooks or non-RTX Windows machines) will not work.

GPU (The Bottleneck): NVIDIA RTX 4070 Ti (12GB VRAM) or higher.

Why: You need high VRAM to load the USD (Universal Scene Description) assets for the robot and environment, plus run the VLA (Vision-Language-Action) models simultaneously.

Ideal: RTX 3090 or 4090 (24GB VRAM) allows for smoother "Sim-to-Real" training.

CPU: Intel Core i7 (13th Gen+) or AMD Ryzen 9.

Why: Physics calculations (Rigid Body Dynamics) in Gazebo/Isaac are CPU-intensive.

RAM: 64 GB DDR5 (32 GB is the absolute minimum, but will crash during complex scene rendering).

OS: Ubuntu 22.04 LTS.

Note: While Isaac Sim runs on Windows, ROS 2 (Humble/Iron) is native to Linux. Dual-booting or dedicated Linux machines are mandatory for a friction-free experience.

2. The "Physical AI" Edge Kit

Since a full humanoid robot is expensive, students learn "Physical AI" by setting up the nervous system on a desk before deploying it to a robot. This kit covers Module 3 (Isaac ROS) and Module 4 (VLA).

The Brain: NVIDIA Jetson Orin Nano (8GB) or Orin NX (16GB).

Role: This is the industry standard for embodied AI. Students will deploy their ROS 2 nodes here to understand resource constraints vs. their powerful workstations.

The Eyes (Vision): Intel RealSense D435i or D455.

Role: Provides RGB (Color) and Depth (Distance) data. Essential for the VSLAM and Perception modules.

The Inner Ear (Balance): Generic USB IMU (BNO055) (Often built into the RealSense D435i or Jetson boards, but a separate module helps teach IMU calibration).

Voice Interface: A simple USB Microphone/Speaker array (e.g., ReSpeaker) for the "Voice-to-Action" Whisper integration.

3. The Robot Lab

For the "Physical" part of the course, you have three tiers of options depending on budget.

Option A: The "Proxy" Approach (Recommended for Budget)

Use a quadruped (dog) or a robotic arm as a proxy. The software principles (ROS 2, VSLAM, Isaac Sim) transfer 90% effectively to humanoids.

Robot: Unitree Go2 Edu (~$1,800 - $3,000).

Pros: Highly durable, excellent ROS 2 support, affordable enough to have multiple units.

Cons: Not a biped (humanoid).

Option B: The "Miniature Humanoid" Approach

Small, table-top humanoids.

Robot: Unitree H1 is too expensive ($90k+), so look at Unitree G1 (~$16k) or Robotis OP3 (older, but stable, ~$12k).

Budget Alternative: Hiwonder TonyPi Pro (~$600).

Warning: The cheap kits (Hiwonder) usually run on Raspberry Pi, which cannot run NVIDIA Isaac ROS efficiently. You would use these only for kinematics (walking) and use the Jetson kits for AI.

Option C: The "Premium" Lab (Sim-to-Real specific)

If the goal is to actually deploy the Capstone to a real humanoid:

Robot: Unitree G1 Humanoid.

Why: It is one of the few commercially available humanoids that can actually walk dynamically and has an SDK open enough for students to inject their own ROS 2 controllers.

4. Summary of Architecture

To teach this successfully, your lab infrastructure should look like this:

Component

Hardware

Function

Sim Rig

PC with RTX 4080 + Ubuntu 22.04

Runs Isaac Sim, Gazebo, Unity, and trains LLM/VLA models.

Edge Brain

Jetson Orin Nano

Runs the "Inference" stack. Students deploy their code here.

Sensors

RealSense Camera + Lidar

Connected to the Jetson to feed real-world data to the AI.

Actuator

Unitree Go2 or G1 (Shared)

Receives motor commands from the Jetson.

If you do not have access to RTX-enabled workstations, we must restructure the course to rely entirely on cloud-based instances (like AWS RoboMaker or NVIDIA's cloud delivery for Omniverse), though this introduces significant latency and cost complexity.

Building a "Physical AI" lab is a significant investment. You will have to choose between building a physical On-Premise Lab at Home (High CapEx) versus running a Cloud-Native Lab (High OpEx).

Option 2 High OpEx: The "Ether" Lab (Cloud-Native)

Best for: Rapid deployment, or students with weak laptops.

1. Cloud Workstations (AWS/Azure) Instead of buying PCs, you rent instances.

Instance Type: AWS g5.2xlarge (A10G GPU, 24GB VRAM) or g6e.xlarge.

Software: NVIDIA Isaac Sim on Omniverse Cloud (requires specific AMI).

Cost Calculation:

Instance cost: ~$1.50/hour (spot/on-demand mix).

Usage: 10 hours/week × 12 weeks = 120 hours.

Storage (EBS volumes for saving environments): ~$25/quarter.

Total Cloud Bill: ~$205 per quarter.

2. Local "Bridge" Hardware You cannot eliminate hardware entirely for "Physical AI." You still need the edge devices to deploy the code physically.

Edge AI Kits: You still need the Jetson Kit for the physical deployment phase.

Cost: $700 (One-time purchase).

Robot: You still need one physical robot for the final demo.

Cost: $3,000 (Unitree Go2 Standard).

The Economy Jetson Student Kit

Best for: Learning ROS 2, Basic Computer Vision, and Sim-to-Real control.

Component

Model

Price (Approx.)

Notes

The Brain

NVIDIA Jetson Orin Nano Super Dev Kit (8GB)

$249

New official MSRP (Price dropped from ~$499). Capable of 40 TOPS.

The Eyes

Intel RealSense D435i

$349

Includes IMU (essential for SLAM). Do not buy the D435 (non-i).

The Ears

ReSpeaker USB Mic Array v2.0

$69

Far-field microphone for voice commands (Module 4).

Wi-Fi

(Included in Dev Kit)

$0

The new "Super" kit includes the Wi-Fi module pre-installed.

Power/Misc

SD Card (128GB) + Jumper Wires

$30

High-endurance microSD card required for the OS.

TOTAL

~$700 per kit

3. The Latency Trap (Hidden Cost)

Simulating in the cloud works well, but controlling a real robot from a cloud instance is dangerous due to latency.

Solution: Students train in the Cloud, download the model (weights), and flash it to the local Jetson kit.

"

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Course Content Access (Priority: P1)

A student enrolled in the Physical AI & Humanoid Robotics course needs to access structured learning materials organized by modules (ROS 2, Gazebo/Unity, NVIDIA Isaac, VLA) to follow the curriculum systematically from foundational concepts to advanced applications. The student should be able to navigate through the content in a logical sequence that matches the weekly breakdown provided in the course outline.

**Why this priority**: This is the core functionality that enables the entire learning experience. Without organized, accessible content, the course cannot function effectively.

**Independent Test**: The student can access the first module (Introduction to Physical AI) and progress through all content sections in the correct sequence, completing all learning objectives by the end of the course.

**Acceptance Scenarios**:

1. **Given** a student has enrolled in the Physical AI course, **When** they access the book content, **Then** they see a clear table of contents organized by modules and weeks with logical progression from basic to advanced topics
2. **Given** a student is at the beginning of Module 1, **When** they navigate to the next section, **Then** they are guided to the appropriate next topic in the learning sequence

---

### User Story 2 - Docusaurus Documentation Navigation (Priority: P1)

An instructor or student needs to efficiently navigate through the course documentation using a well-structured sidebar that reflects the course's table of contents. They should be able to quickly find specific topics, jump between related sections, and access supplementary materials like hardware requirements and lab setup guides.

**Why this priority**: Effective navigation is critical for the usability of the course materials, enabling both quick reference and comprehensive study.

**Independent Test**: The user can use the sidebar navigation to access any section of the course content within 3 clicks and can easily move between related topics.

**Acceptance Scenarios**:

1. **Given** a user is viewing any section of the course, **When** they use the sidebar navigation, **Then** they can access all major sections of the course in a logical hierarchy
2. **Given** a user wants to find hardware requirements, **When** they navigate through the sidebar, **Then** they can quickly locate the hardware section and related setup guides

---

### User Story 3 - Technical Setup Guidance (Priority: P2)

A student or instructor needs clear, step-by-step instructions for setting up the required hardware and software environment for the course, including workstation specifications, ROS 2 installation, Gazebo setup, and NVIDIA Isaac platform configuration.

**Why this priority**: Without proper setup, students cannot engage with the course content, making this a critical prerequisite for course success.

**Independent Test**: A user can follow the setup instructions and successfully configure their environment to run the basic course examples.

**Acceptance Scenarios**:

1. **Given** a student has the required hardware, **When** they follow the setup instructions, **Then** they can successfully install and configure ROS 2, Gazebo, and NVIDIA Isaac
2. **Given** a student encounters a setup issue, **When** they refer to the troubleshooting section, **Then** they can find solutions for common configuration problems

---

### User Story 4 - Assessment and Capstone Project Guidance (Priority: P2)

A student needs clear guidance on assessments and the capstone project requirements, including evaluation criteria, submission guidelines, and examples of successful implementations.

**Why this priority**: Assessment guidance ensures students understand expectations and can successfully complete the course requirements.

**Independent Test**: A student can understand what is required for each assessment and the capstone project, and complete these successfully.

**Acceptance Scenarios**:

1. **Given** a student wants to understand capstone project requirements, **When** they read the project section, **Then** they understand the deliverables and evaluation criteria
2. **Given** a student is working on an assessment, **When** they refer to the guidelines, **Then** they can complete the assessment according to specified requirements

---

### Edge Cases

- What happens when a student accesses the course materials with limited internet connectivity and needs to download large simulation files?
- How does the system handle students with different technical backgrounds and varying levels of robotics experience?
- What occurs when students use different hardware configurations than those specified in the requirements?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a comprehensive table of contents that follows the course outline structure with modules, weeks, and sections in logical sequence
- **FR-002**: System MUST organize content by the four main modules: ROS 2, Gazebo/Unity, NVIDIA Isaac, and Vision-Language-Action
- **FR-003**: Users MUST be able to navigate between course sections using a structured sidebar that reflects the content hierarchy
- **FR-004**: System MUST provide detailed hardware requirements and setup instructions for workstations, edge computing kits, and robot options
- **FR-005**: System MUST include weekly breakdown content that aligns with the 13-week course schedule
- **FR-006**: System MUST provide assessment guidelines and capstone project requirements with clear evaluation criteria
- **FR-007**: System MUST include troubleshooting guides for common setup and configuration issues
- **FR-008**: System MUST provide learning outcomes for each module and the overall course
- **FR-009**: System MUST offer both on-premise and cloud-based lab setup options with cost and performance trade-offs
- **FR-010**: System MUST include detailed installation guides for ROS 2, Gazebo, Unity, and NVIDIA Isaac platforms

### Key Entities

- **Course Content**: Structured learning materials organized by modules, weeks, and topics following the Physical AI curriculum
- **Hardware Requirements**: Specifications for workstations, edge computing kits, sensors, and robot platforms needed for the course
- **Assessment Materials**: Guidelines, rubrics, and examples for course assessments and the capstone project
- **Setup Guides**: Step-by-step instructions for software installation, hardware configuration, and environment setup

## Clarifications

### Session 2025-12-27

- Q: What Docusaurus configuration should be used? → A: Standard Docusaurus 3.x with default theme and basic documentation features
- Q: What are the performance requirements? → A: Optimize both loading times and content quality for fast performance with high-quality, comprehensive content
- Q: What search and accessibility features are needed? → A: The best option possible - Intelligent search with recommendations and full accessibility compliance
- Q: Where should the site be deployed? → A: GitHub Pages, using the same repository
- Q: What security/authentication is required? → A: Public access initially - No authentication required, with plans to add authentication later

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a comprehensive table of contents that follows the course outline structure with modules, weeks, and sections in logical sequence
- **FR-002**: System MUST organize content by the four main modules: ROS 2, Gazebo/Unity, NVIDIA Isaac, and Vision-Language-Action
- **FR-003**: Users MUST be able to navigate between course sections using a structured sidebar that reflects the content hierarchy
- **FR-004**: System MUST provide detailed hardware requirements and setup instructions for workstations, edge computing kits, and robot options
- **FR-005**: System MUST include weekly breakdown content that aligns with the 13-week course schedule
- **FR-006**: System MUST provide assessment guidelines and capstone project requirements with clear evaluation criteria
- **FR-007**: System MUST include troubleshooting guides for common setup and configuration issues
- **FR-008**: System MUST provide learning outcomes for each module and the overall course
- **FR-009**: System MUST offer both on-premise and cloud-based lab setup options with cost and performance trade-offs
- **FR-010**: System MUST include detailed installation guides for ROS 2, Gazebo, Unity, and NVIDIA Isaac platforms
- **FR-011**: System MUST implement Standard Docusaurus 3.x with default theme and basic documentation features
- **FR-012**: System MUST optimize for fast loading times while maintaining high-quality, comprehensive content
- **FR-013**: System MUST provide intelligent search with recommendations and full accessibility compliance
- **FR-014**: System MUST be deployable to GitHub Pages
- **FR-015**: System MUST support public access initially with plans for future authentication integration

### Key Entities

- **Course Content**: Structured learning materials organized by modules, weeks, and topics following the Physical AI curriculum
- **Hardware Requirements**: Specifications for workstations, edge computing kits, sensors, and robot platforms needed for the course
- **Assessment Materials**: Guidelines, rubrics, and examples for course assessments and the capstone project
- **Setup Guides**: Step-by-step instructions for software installation, hardware configuration, and environment setup
- **Docusaurus Documentation Site**: The web-based platform hosting the course materials with navigation, search, and accessibility features

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Students can navigate to any section of the course content within 3 clicks using the sidebar navigation system
- **SC-002**: 90% of students successfully complete the hardware/software setup within the first week of the course
- **SC-003**: Students can understand and follow the 13-week course progression from foundational Physical AI concepts to the capstone project
- **SC-004**: 85% of students complete the capstone project successfully with the provided documentation and guidelines
- **SC-005**: Students can access all four main modules (ROS 2, Gazebo/Unity, NVIDIA Isaac, VLA) and understand the progression between them
- **SC-006**: Users can find hardware requirements and setup instructions within 2 minutes of accessing the documentation
- **SC-007**: Students report course documentation as "clear and helpful" in at least 80% of feedback surveys
- **SC-008**: Pages load quickly with optimized assets while maintaining high-quality content display
- **SC-009**: Intelligent search functionality allows users to find relevant content with recommendations
- **SC-010**: Site meets full accessibility compliance standards for inclusive learning
- **SC-011**: Documentation is successfully deployed and accessible via GitHub Pages
- **SC-012**: Content is publicly accessible with plans for future authentication system integration