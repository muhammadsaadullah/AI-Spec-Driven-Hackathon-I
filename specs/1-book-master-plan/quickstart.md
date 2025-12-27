# Quickstart Guide: Physical AI & Humanoid Robotics Course Documentation

## Overview
This guide will help you set up and run the Physical AI & Humanoid Robotics course documentation site locally. The site is built with Docusaurus 3.x and deployed on GitHub Pages.

## Prerequisites
- Node.js version 18 or higher
- npm or yarn package manager
- Git

## Local Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/[your-repo]/physical-ai-course.git
cd physical-ai-course
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Run the Development Server
```bash
npm run start
```
This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### 4. Build for Production
```bash
npm run build
```
This command generates static content into the `build` directory and can be served using any static contents hosting service.

### 5. Deploy to GitHub Pages
```bash
npm run deploy
```
This command builds your site and deploys it to GitHub Pages using the `gh-pages` branch.

## Docusaurus Configuration

### Site Configuration
The site configuration is located in `docusaurus.config.js` and includes:
- Site metadata (title, tagline, URL)
- Theme configuration
- Navigation bar settings
- Footer configuration
- Algolia search settings (if applicable)

### Sidebar Configuration
Sidebars are defined in `sidebars.js` and organized by:
- Course modules (ROS 2, Gazebo/Unity, NVIDIA Isaac, VLA)
- Weekly breakdown (Weeks 1-13)
- Hardware requirements
- Assessments
- Reference materials

### Content Structure
Content files are located in the `docs` directory and organized by:
- Modules: `/docs/modules/[module-name]/`
- Weekly breakdown: `/docs/weekly-breakdown/weeks-[range]-[topic]/`
- Hardware requirements: `/docs/hardware-requirements/`
- Assessments: `/docs/assessments/`
- Reference: `/docs/reference/`

## Adding New Content

### 1. Create a New Document
Create a new Markdown file in the appropriate directory:
```bash
# For example, adding a new ROS 2 tutorial
docs/modules/ros2/new-tutorial.md
```

### 2. Add Frontmatter Metadata
Include the required metadata at the top of your document:
```markdown
---
title: "Descriptive Title with Primary Keywords"
description: "Concise summary (150-160 chars) with secondary keywords"
keywords: ["primary-keyword", "ros2", "humanoid", "specific-concept"]
sidebar_position: 10
---
```

### 3. Update Sidebar Navigation
Add your document to the appropriate sidebar in `sidebars.js`:
```javascript
moduleSidebar: [
  // ... other items
  {
    type: 'doc',
    id: 'modules/ros2/new-tutorial', // Path to your document
    label: 'New Tutorial', // Display name in sidebar
  },
],
```

### 4. Add Internal Links
Use relative paths for internal links:
```markdown
[ROS 2 Fundamentals](./fundamentals.md)
[Week 1-2: Introduction to Physical AI](../weekly-breakdown/weeks-1-2-intro-physical-ai.md)
```

## Content Guidelines

### Document Structure
Each document should follow this structure:
1. Metadata (frontmatter)
2. H1 heading (matches title in frontmatter)
3. Brief introduction
4. Main content with appropriate headings
5. Summary or next steps
6. Related links (optional)

### Code Examples
All code blocks should specify the language:
```python
# Requires: numpy>=1.24.0, rclpy>=0.14.0
# Tested with: ros-humble-desktop (2024.1), python==3.10
import rclpy
from rclpy.node import Node

class ExampleNode(Node):
    def __init__(self):
        super().__init__('example_node')
        self.get_logger().info('Example node initialized')
```

### Image Usage
- Store images in `static/img/[category]/`
- Use descriptive filenames
- Include alt text for accessibility
```markdown
![Description of image](/img/modules/ros2/node-communication-diagram.svg)
```

## Building and Deployment

### Local Testing
Before committing changes, test the build:
```bash
npm run build
```

### GitHub Pages Deployment
The site is automatically deployed to GitHub Pages when changes are pushed to the `main` branch, or you can manually deploy with:
```bash
npm run deploy
```

### Continuous Integration
The repository should include a GitHub Actions workflow for:
- Build validation
- Link checking
- Spell checking
- Performance auditing

## Troubleshooting

### Common Issues

**Error: Node version mismatch**
- Solution: Use Node.js version 18 or higher

**Error: Missing dependencies**
- Solution: Run `npm install` to install all dependencies

**Error: Build fails**
- Solution: Check for syntax errors in Markdown files and configuration files

### Useful Commands
- `npm run serve`: Serve the built website locally
- `npm run swizzle`: Customize Docusaurus components (use carefully)
- `npm run clear`: Clear Docusaurus cache

## Next Steps
1. Review the [Docusaurus Documentation](https://docusaurus.io/docs) for advanced features
2. Explore the [Course Content Structure](/docs/intro) to understand how the documentation is organized
3. Start adding content by following the patterns established in existing documents
4. Update the sidebar configuration to include your new content
5. Test your changes locally before pushing to GitHub