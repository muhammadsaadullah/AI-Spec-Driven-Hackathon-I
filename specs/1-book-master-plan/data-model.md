# Data Model: Book Master Plan for Physical AI & Humanoid Robotics Course

## Entity: Course Content
**Description**: Structured learning materials organized by modules, weeks, and topics following the Physical AI curriculum
**Fields**:
- id: Unique identifier for the content piece
- title: Title of the content section
- module: Module category (ROS 2, Gazebo/Unity, NVIDIA Isaac, VLA)
- week: Week number in the 13-week schedule (if applicable)
- content_type: Type of content (lesson, tutorial, reference, assessment)
- prerequisites: List of required knowledge/skills
- learning_objectives: List of measurable learning outcomes
- duration: Estimated time to complete (in minutes)
- difficulty: Difficulty level (beginner, intermediate, advanced)
- content: Main content in Markdown format
- metadata: Additional metadata (tags, keywords, author, date)

**Relationships**:
- Contains multiple Sections
- Links to related Hardware Requirements
- Links to associated Code Examples
- Links to Assessment Materials

## Entity: Hardware Requirements
**Description**: Specifications for workstations, edge computing kits, sensors, and robot platforms needed for the course
**Fields**:
- id: Unique identifier for the hardware component
- component_type: Type of component (workstation, edge_kit, sensor, robot)
- name: Name of the hardware component
- model: Specific model number
- specifications: Technical specifications
- price: Approximate price
- notes: Additional notes about the component
- required_for: Which modules/weeks require this component
- alternatives: Alternative components that can be used

**Relationships**:
- Referenced by multiple Course Content sections
- Used in Setup Guides

## Entity: Assessment Materials
**Description**: Guidelines, rubrics, and examples for course assessments and the capstone project
**Fields**:
- id: Unique identifier for the assessment
- title: Title of the assessment
- type: Type of assessment (project, quiz, practical, capstone)
- module: Module the assessment belongs to
- week: Week the assessment is due
- objectives: Learning objectives being assessed
- requirements: Specific requirements for the assessment
- evaluation_criteria: Rubric for grading
- submission_guidelines: How to submit the assessment
- examples: Examples of successful implementations

**Relationships**:
- Links to specific Course Content
- Associated with specific Modules and Weeks

## Entity: Setup Guides
**Description**: Step-by-step instructions for software installation, hardware configuration, and environment setup
**Fields**:
- id: Unique identifier for the setup guide
- title: Title of the setup guide
- target: Target audience (student, instructor)
- prerequisites: Prerequisites for following the guide
- steps: List of step-by-step instructions
- platform: Target platform (Ubuntu 22.04, Windows, etc.)
- software_versions: Specific software versions required
- troubleshooting: Common issues and solutions
- estimated_time: Time needed to complete the setup

**Relationships**:
- References specific Hardware Requirements
- Links to relevant Course Content

## Entity: Docusaurus Documentation Site
**Description**: The web-based platform hosting the course materials with navigation, search, and accessibility features
**Fields**:
- id: Unique identifier for the site configuration
- title: Site title
- tagline: Site tagline
- url: Site URL
- base_url: Base URL for the site
- favicon: Path to favicon
- theme_config: Configuration for the theme
- navbar: Navigation bar configuration
- footer: Footer configuration
- algolia: Search configuration
- markdown_extensions: Additional Markdown extensions enabled

**Relationships**:
- Contains multiple Course Content pages
- Uses Hardware Requirements information for display
- Includes Assessment Materials
- Contains Setup Guides

## State Transitions

### Course Content States
- draft → review (when content is ready for review)
- review → published (when content passes review)
- published → archived (when content is outdated)

### Assessment States
- proposed → active (when assessment is ready for use)
- active → completed (when assessment period is over)
- completed → archived (when no longer needed)

## Validation Rules

### Course Content Validation
- Title must not be empty
- Content must be in valid Markdown format
- At least one learning objective must be specified
- Duration must be a positive number
- Week number must be between 1-13 for weekly content

### Hardware Requirements Validation
- Name must not be empty
- Price must be a positive number or marked as "TBD"
- Specifications must be provided
- Component type must be one of the defined types

### Assessment Materials Validation
- Title must not be empty
- At least one evaluation criterion must be specified
- Type must be one of the defined types
- Module must be specified

### Setup Guides Validation
- Title must not be empty
- Steps must not be empty
- Platform must be specified
- Estimated time must be a positive number