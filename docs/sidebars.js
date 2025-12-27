/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

module.exports = {
  // Manual sidebar configuration for the Physical AI & Humanoid Robotics course
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: 'Home'
    },
    {
      type: 'category',
      label: 'Course Modules',
      items: [
        {
          type: 'category',
          label: 'Module 1: ROS 2 Fundamentals',
          items: [
            'modules/ros2/index',
            'modules/ros2/fundamentals',
            'modules/ros2/nodes-topics-services',
            'modules/ros2/urdf-humanoids'
          ]
        },
        {
          type: 'category',
          label: 'Module 2: Gazebo & Unity Simulation',
          items: [
            'modules/gazebo-unity/index',
            'modules/gazebo-unity/simulation-setup',
            'modules/gazebo-unity/physics-collision',
            'modules/gazebo-unity/sensors-simulation'
          ]
        },
        {
          type: 'category',
          label: 'Module 3: NVIDIA Isaac Platform',
          items: [
            'modules/nvidia-isaac/index',
            'modules/nvidia-isaac/isaac-sim',
            'modules/nvidia-isaac/vsalm-navigation',
            'modules/nvidia-isaac/nav2-path-planning'
          ]
        },
        {
          type: 'category',
          label: 'Module 4: Vision-Language-Action (VLA)',
          items: [
            'modules/vla/index',
            'modules/vla/voice-to-action',
            'modules/vla/cognitive-planning',
            'modules/vla/capstone-project'
          ]
        }
      ]
    },
    {
      type: 'category',
      label: 'Weekly Breakdown',
      items: [
        'weekly-breakdown/weeks-1-2-intro-physical-ai',
        'weekly-breakdown/weeks-3-5-ros2-fundamentals',
        'weekly-breakdown/weeks-6-7-gazebo-simulation',
        'weekly-breakdown/weeks-8-10-nvidia-isaac',
        'weekly-breakdown/weeks-11-12-humanoid-development',
        'weekly-breakdown/week-13-conversational-robotics'
      ]
    },
    {
      type: 'doc',
      id: 'hardware-requirements',
      label: 'Hardware Requirements'
    },
    {
      type: 'category',
      label: 'Reference Materials',
      items: [
        'reference/glossary',
        'reference/notation'
      ]
    },
    {
      type: 'doc',
      id: 'assessments/index',
      label: 'Assessment Guidelines'
    }
  ]
};
