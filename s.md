# Physical AI & Humanoid Robotics Textbook Constitution

**Version:** 2.0.0 (Merged & Enhanced)  
**Ratified:** 2025-12-07  
**Authority:** Governs all content, review, and publishing practices for the textbook and supporting materials

> **Note:** This constitution represents the synthesis of foundational and enhanced governance frameworks, combining operational clarity with safety-critical rigor for Physical AI and Humanoid Robotics education.

---

## Table of Contents

1. [Core Principles](#core-principles)
2. [Content Development Workflow](#content-development-workflow)
3. [Quality Gates & Review Process](#quality-gates--review-process)
4. [Governance](#governance)

---

## Core Principles

### I. Content Accuracy & Technical Rigor

All content involving robotics, AI, simulation, hardware, physics, control theory, SLAM, human–robot interaction, and real-world deployment MUST be precise, validated, and grounded in established scientific principles. This textbook integrates multiple high-risk domains—therefore accuracy is not merely a quality metric; it is a safety requirement.

#### Rules

**Verification & Cross-Checking**

- All technical claims (robot dynamics, Isaac Sim specifications, ROS 2 architecture, SLAM algorithms, locomotion models, etc.) MUST be validated using authoritative robotics sources.

- Every technical statement MUST be cross-verified against authoritative robotics references (e.g., ROS 2 REP standards, IEEE papers, Isaac Sim documentation, textbooks on dynamics and control, sensor datasheets).

**Mathematical Content**

- Mathematical equations (kinematics, dynamics, optimization, control laws) MUST be checked for correctness and reviewed by a domain expert.

- Mathematical content (kinematics, dynamics, optimization, control stability conditions, sampling laws) MUST be correct, consistent across chapters, and reviewed by a domain-knowledgeable peer.

**Code & Runtime Environment**

- Code examples MUST be fully functional and runnable in the defined software stack:

  - Ubuntu 22.04 LTS
  - ROS 2 Humble or Iron
  - rclpy Python API
  - Gazebo (Garden/Fortress)
  - Unity HDRP (version-pinned)
  - NVIDIA Isaac Sim (Omniverse) specific version listed
  - Python 3.10+

- All example code MUST execute as written under a reproducible environment defined in a centralized configuration.

**Simulation & Physical Parameters**

- Simulation parameters (mass, inertia, friction coefficients, control gains, noise models, inertia tensors, physical material properties, sensor noise, locomotion controller gains) MUST NOT be fabricated and MUST be physically plausible and derived from cited or measured sources.

- Hardware specifications (Jetson boards, RealSense cameras, Unitree robots, sensors) MUST match manufacturer documentation.

- Hardware details (Jetson SKUs, depth sensors, IMUs, servo torque ratings, battery specs) MUST reflect the manufacturer's current documentation.

**Research Claims & Citations**

- No speculative claims about humanoid capabilities unless the text explicitly labels them as projections or research directions.

- All external research claims (VSLAM, LLM planning, imitation learning, locomotion control, computer vision, reinforcement learning, VLA models, and motion planning) MUST include academic citation.

**Stability & Performance**

- A derivation, proof, or experimental justification MUST be provided for any claim involving:

  - Bipedal stability
  - Center of Mass (CoM) modeling
  - Whole-body control
  - Pose estimation accuracy
  - SLAM drift rates
  - Latency constraints

- Stability or performance-critical concepts MUST be supported by derivations, empirical results, or diagrams explaining the intuition.

**Dependency Tracking & APIs**

- Software dependency versions MUST be explicitly stated with version constraints:

  ```python
  # Requires: numpy>=1.24.0, torch==2.2.0, isaacsim==2023.1.1
  # Tested with:
  # ros-humble-desktop (2024.1)
  # isaac-sim==2023.1.1
  # pytorch==2.2.1
  ```

- All APIs used MUST be documented with:

  - Module/library name and version
  - Function/class signatures
  - Usage context and why chosen
  - Alternative APIs rejected and why

- API deprecations MUST be tracked:

  - Old API → New API migration path
  - Timeline for deprecation
  - Backward compatibility notes

- Software dependencies MUST follow:
  - Prefer standard, widely-adopted packages (ROS 2, NumPy, PyTorch, scikit-learn) over obscure/niche libraries
  - Minimize external dependencies; justify each addition
  - Lock versions for reproducibility
  - Include both runtime AND development dependencies

**Logical Reasoning Foundation**

- All concepts MUST follow principles of either deductive or inductive logical reasoning:
  - **Deductive:** General principles → Specific applications (e.g., Newton's laws → robot dynamics)
  - **Inductive:** Specific observations → General patterns (e.g., sensor measurements → perception models)
  - Instructional choice: Use whichever is more effective for teaching each problem/topic
  - Explicitly state which reasoning approach is used and justify the pedagogical choice

#### Rationale

Physical AI integrates AI, physics, and robotics—errors propagate across domains and may lead to:

- Unstable simulations and control policies
- Incorrect robot behavior
- Invalid locomotion models
- Unsafe hardware interaction
- Simulation/real-world divergence
- Broken perception pipelines

Accuracy directly affects both learning quality and real-robot safety. Accuracy is non-negotiable in an embodied intelligence textbook.

---

### II. Educational Clarity & Accessibility

This textbook MUST distill advanced embodied AI concepts into material approachable by learners from multiple backgrounds while maintaining professional rigor. The textbook MUST teach complex robotic systems in a clear, structured, and accessible way.

#### Rules

**Pedagogical Structure**

- Each chapter MUST list explicit:

  - Prerequisites ( for eg, physics, Python, math etc )
  - Target audience (undergrad/grad/practitioners - specify per section if mixed)
  - Learning objectives MUST be measurable and stated at chapter start
  - Key terms introduced with context
  - Progressive conceptual buildup

- Every chapter MUST be pedagogically structured with explicit prerequisites, learning objectives, key terms introduced with context, and progressive conceptual buildup.

**Conceptual Development & Layering**

- The structure for each major concept must follow:

  1. Motivation (Why this matters in Physical AI)
  2. Simple intuitive explanation
  3. Formal definition (math, algorithm, architecture)
  4. Real system example (ROS 2, Isaac, Gazebo)

- Each new idea MUST be explained across three layers:
  - **Intuition** → **Formalism** → **Real System Implementation**

**Examples & Visualization**

- Worked examples required per major concept, esp for:

  - ROS 2 messaging and TF transforms
  - URDF modeling
  - Isaac Sim reinforcement learning
  - VLA planning pipelines
  - SLAM and perception systems

- Wherever possible, chapters should include:

  - Conceptual diagrams, flowcharts
  - Simplified analogies for intuition
  - Deep dives for advanced readers
  - Partial steps toward mastery-level understanding

- Physical concepts (force, inertia, stability, friction, error propagation) MUST include visual examples from simulation.

- Diagrams REQUIRED for:
  - Humanoid kinematics
  - ROS 2 graph structures
  - Sensor configurations
  - Unity/Gazebo simulation scenes
  - Isaac Sim perception pipelines

**Bridging Theory to Implementation**

- "Bridging explanations" MUST be used to connect:

  - Theory to simulation
  - Simulation to code
  - Code to real robot behavior

- All examples MUST include context explaining why they matter in humanoid robotics or Physical AI.

**Accessibility & Navigation**

- Glossary terms must link on first use per chapter.

- Use consistent explanation style: **Theory → System → Implementation → Experiment**

#### Rationale

Humanoid robotics is high-dimensional. The domain is complex; clarity reduces cognitive overload, learner frustration, prevents misunderstandings, and creates a stable foundation for advanced reasoning and enables independent learning.

---

### III. Consistency & Standards (NON-NEGOTIABLE)

Consistency is essential for robotic systems, where units, notation, naming, and frames must align or the system fails. Uniform terminology, notation, and formatting across the entire textbook.

#### Rules

**Terminology & Glossary**

- Terminology MUST strictly follow a centralized glossary defining:

  - ROS 2 naming conventions
  - Robot frames
  - Control terminology
  - State representations

- Terminology consistency enforced via `docs/glossary.md`  
  (e.g., "base_link" not "base"; "IMU" not "imu"; "Center of Mass (CoM)" always capitalized on first use).

**Mathematical Notation**

- Mathematical notation MUST follow a system-wide schema defined in `docs/notation.md`:
  - Vectors = bold lowercase (**v**)
  - Matrices = bold uppercase (**M**)
  - Robot frames: {B}, {W}, {H}
  - Joints: q, ẋ, τ
  - CoM, ZMP, base_link, map, odom all standardized

**Code Standards**

- ROS 2 code must follow official Python style conventions (PEP 8 + ROS 2 Python style).

- All ROS 2 examples MUST follow:

  - Proper rclpy structure
  - Proper launch system organization
  - Consistent TF trees

- Code comments MUST explain why something is done, not only what is done.

**URDF & Hardware Description**

- URDF must follow:
  - SI units
  - Right-handed coordinate frames
  - Consistent mesh paths

**Chapter Template Structure**

- All chapters MUST follow a uniform structure with:

  1. Learning Objectives
  2. Prerequisites
  3. Content Sections (Introduction → Core Concepts → Examples → Applications)
  4. Summary
  5. Exercises
  6. References

- Or alternatively:
  - Objectives
  - Prerequisites
  - Conceptual development
  - Applied examples
  - Implementation
  - Exercises
  - References

**Voice & Tone**

- Voice standards:
  - Theory uses third person ("the robot computes…")
  - Tutorials use second person ("you will launch…")

#### Rationale

Robot systems are sensitive to inconsistencies—especially units, frames, and terminology. In robotics, inconsistency leads to misaligned frames, misinterpreted quantities, and unpredictable system behavior. Consistency prevents conceptual errors and preserves conceptual integrity.

---

### IV. Docusaurus Structure & Quality

All textbook content must build cleanly and remain navigable. The website must behave like a professional-grade technical documentation platform for a robotics curriculum.

#### Rules

**Page Composition & Metadata**

- One core concept per page (max 2000 words).

- All pages must contain metadata:

  ```yaml
  ---
  title: "Topic Name"
  description: "Short descriptive summary"
  keywords: ["physical ai", "ros2", "humanoid robotics"]
  sidebar_position: <number>
  ---
  ```

- Each page MUST include clear metadata for indexing, navigation, and search optimization.

**Content Scope & Organization**

- Chapters MUST be atomic, readable, and scoped to a single central concept.

- The sidebar MUST mirror the curriculum's learning progression from foundational material to advanced application.
- Sidebar organization: Hierarchical by complexity (Fundamentals → Intermediate → Advanced → Specialized Topics)

**Search Optimization**

- SEO keywords MUST appear in:

  - H1 heading (page title)
  - First paragraph of content
  - Metadata fields (title, description, keywords)
  - Alt text for diagrams

- Docusaurus metadata structure:

  ```yaml
  ---
  title: "Descriptive Title with Primary Keywords"
  description: "Concise summary (150-160 chars) with secondary keywords"
  keywords: ["primary-keyword", "ros2", "humanoid", "specific-concept"]
  sidebar_position: <number>
  ---
  ```

- Internal links MUST use anchor text containing keywords (not "click here")
- Each section should have descriptive subheadings containing searchable terms

- Sidebar organized by course progression:
  - Foundations of Physical AI
  - ROS 2 Nervous System
  - Simulation (Gazebo & Unity)
  - NVIDIA Isaac Perception & Control
  - Vision-Language-Action (VLA)
  - Humanoid Mechanics & Locomotion
  - Capstone

**Image & Asset Management**

- Images stored in `static/img/[chapter]/`

  - Naming must be descriptive:
    - `ros2-node-graph.svg`
    - `humanoid-balance-control.png`

- Alt text required:

  - "Diagram of a humanoid robot showing CoM and support polygon"

- Images MUST be:

  - Version-controlled
  - Optimized
  - Properly annotated
  - Stored in logically structured folders

- All diagrams MUST carry instructive captions and alt text.

**Navigation & Cross-Linking**

- Internal links MUST use relative paths.

- The layout MUST support seamless cross-navigation between theory, simulation, and code.

#### Rationale

Students need to find material quickly across multiple technical systems. A structured knowledge system ensures that students and professionals can quickly locate relevant content and follow the logical flow of the curriculum.

---

### V. Code Example Quality

All programming content MUST be runnable on the defined toolchain. In robotics, code is not supplementary—it is the operational embodiment of theory. Therefore, all code must reflect professional-grade engineering standards.

#### Rules

**Language & Framework Selection**

- Use:

  - Python + rclpy for ROS 2 nodes
  - Isaac Sim Python API
  - Gazebo Ignition Python/ROS interface
  - Unity C# scripts ONLY when necessary

- Prefer standard libraries and widely-adopted packages:
  - ROS 2 core libraries
  - NumPy, SciPy for numerical computing
  - PyTorch/TensorFlow for ML
  - OpenCV for vision
  - Justify any external dependencies

**Code Quality Standards**

- Code examples must:

  - Run without modification on Ubuntu 22.04
  - Include dependency versions
  - Follow PEP 8
  - Include WHY-based comments
  - Include warnings for real-robot usage

- All code MUST run without modification on the reference environment.

- Examples MUST be short, focused, and directly relevant to the concept being taught.

**Language Specification & Code Formatting**

- All fenced code blocks MUST specify language:

  ```python
  # Correct: language specified
  def compute_kinematics(q):
      return forward_kinematics(q)
  ```

  NOT:

  ```
  # Incorrect: no language specified
  ```

- Complete examples ONLY—no code fragments except when explicitly marked:

  ```python
  # CORRECT: Complete, runnable example
  import rclpy
  from rclpy.node import Node

  class ExampleNode(Node):
      def __init__(self):
          super().__init__('example_node')
  ```

  vs.

  ```python
  # Excerpt from /examples/humanoid/locomotion/main.py
  # (use when showing a portion of larger file)
  ```

**Comments & Documentation**

- Comments MUST explain WHY, not WHAT:

  ```python
  # CORRECT: Why
  # Use inverse kinematics instead of forward kinematics because
  # we need to solve for joint angles given end-effector position
  joint_angles = compute_ik(desired_pose)

  # INCORRECT: What (obvious from code)
  # Set joint_angles using inverse kinematics
  joint_angles = compute_ik(desired_pose)
  ```

- Assume reader understands Python/ROS syntax; focus on algorithmic/domain reasoning

**Dependency Declaration**

- List all dependencies with explicit versions:
  ```python
  # Requires: numpy>=1.24.0, robotics-toolbox-python==1.1.0, rclpy>=0.14.0
  # Tested with: ros-humble-desktop (2024.1), python==3.10
  ```

**Repository Structure for Examples**

- Organize examples hierarchically:

  ```
  /examples/
    [chapter-name]/
      [example-name]/
        main.py              # Primary script
        test_[example].py    # Validation/test script
        README.md            # Usage guide and concept explanation
        requirements.txt     # Dependencies
  ```

- Example directory structure:
  - **main.py** — Complete, runnable example
  - **test\_\*.py** — Validation script or unit tests
  - **README.md** — Purpose, prerequisites, usage instructions, expected output
  - **requirements.txt** — All dependencies with versions

**Test Coverage & Validation**

- Each code example MUST include:
  - Validation script demonstrating correct output
  - Simple unit test (even if basic)
  - Expected output documented
  - Instructions for running validation

**Safety Warnings**

- Hardware-interacting code MUST include prominent safety comments:

  ```python
  # ⚠️ SAFETY WARNING: This code controls a real robot
  # BEFORE RUNNING:
  # 1. Ensure e-stop is reachable and functional
  # 2. Verify robot is on flat, clear ground
  # 3. Confirm no personnel within 2m radius
  # 4. Power on only when ready to execute

  def send_joint_command(angles):
      # Send command to hardware
      pass
  ```

**ROS 2 Best Practices**

- ROS 2 nodes MUST follow best practices:
  - Parameterized
  - Lifecycle nodes when appropriate
  - Structured logging
  - Clear TF publication

**Simulation & Visualization Scripts**

- Isaac Sim and Gazebo scripts MUST illustrate:
  - Deterministic initialization
  - Sensor setup
  - Stable physics settings
  - Proper scene management

**Example Organization & Scope**

- Structure: `/examples/[module]/[topic]/`

- Required examples per chapter:
  - **ROS 2:** Node, Publisher/Subscriber, TF2 broadcaster, Launch system
  - **Gazebo:** URDF spawn, sensor simulation
  - **Isaac Sim:** RL training loop, camera pipeline
  - **VLA:** Whisper + LLM + ROS action pipeline

**High-Level Applications**

- VLA examples MUST demonstrate full pipelines (e.g., Whisper → LLM → ROS action server).

**Safety Critical Content**

- Real-robot code must include safety warnings:

  - "Ensure Unitree robot is on flat ground and e-stop is reachable."

- Real-robot examples MUST include explicit safety notes (e.g., e-stop readiness, power isolation, surroundings clear).

#### Rationale

Code is the connection between AI and the physical humanoid. Broken examples break the learning chain. Poor code undermines comprehension, introduces technical debt, and creates hazardous failure modes when used on hardware.

---

### VI. Deployment & Publishing Standards

All content must build cleanly and maintain performance. The textbook MUST publish as a stable, production-grade platform with professional reliability.

#### Rules

**Branch & Publishing Policy**

- `main` branch contains ONLY production-ready textbook content (deployed to GitHub Pages)

- Only fully reviewed, production-ready content is allowed on the main branch.

**Branch & Pull Request Workflow**

- Feature branches MUST follow naming convention:

  - Content: `chapter/[chapter-name]` (e.g., `chapter/ros2-basics`)
  - Bug fixes: `fix/[issue-description]` (e.g., `fix/broken-simulation-link`)
  - Examples: `example/[module]/[name]` (e.g., `example/locomotion/bipedal-gait`)
  - Documentation: `docs/[topic]` (e.g., `docs/api-reference`)

- Pull Requests MUST include:
  - Description of changes
  - Checklist of compliance items
  - Reference to related issues
  - Evidence of testing (screenshots, terminal output, etc.)

**CI/CD & Quality Checks**

- All PRs MUST pass:

  - Docusaurus build (no warnings or errors )
  - Broken links checker passes (internal and external links)
  - Spell checker (robotics terms whitelisted)
  - Image optimization (<500 KB, use svgo for SVGs)
  - Code syntax validation (Python, YAML, Markdown)
  - Language specification in all code blocks
  - API documentation completeness

- The CI pipeline MUST validate:
  - Build success
  - Spell checking
  - Link integrity
  - Page performance
  - Asset optimization
  - Code example executability (test runs)
  - Dependency version constraints
  - Safety warnings presence in hardware code

**Performance Requirements**

- Performance targets:

  - Initial load < 3 sec
  - LCP < 2.5s
  - CLS < 0.1

- Performance requirements:
  - Pages load quickly
  - Images optimized
  - Navigation stable

**SEO & Discoverability**

- SEO standards:
  - Open Graph tags for social sharing
  - Sitemap generated automatically
  - robots.txt configured

**Versioning & Maintenance**

- Versioning:

  - Major book updates → v2.0, v3.0, etc.
  - Redirects maintained for deprecated URLs.

- Versioning MUST follow semantic structure:

  - Major updates → v2.0 (significant curriculum changes)
  - Minor updates → v2.1 (new chapters, major revisions)
  - Patch updates → v2.0.1 (bug fixes, clarifications)

- Every deprecated URL MUST maintain functional redirects.

**URL Redirects & Docusaurus Configuration**

- Deprecated URLs MUST redirect to updated content:

  - Configure in `docusaurus.config.js` using redirect rules
  - Document all redirect mappings in `REDIRECTS.md`
  - Maintain 301 (permanent) redirects for SEO
  - Test redirects in CI/CD pipeline

- Redirect configuration example:

  ```javascript
  // docusaurus.config.js
  const redirects = [
    {
      from: "/docs/old-chapter-name",
      to: "/docs/new-chapter-name",
    },
    {
      from: "/docs/deprecated-section",
      to: "/docs/updated-location/new-section",
    },
  ];
  ```

- Redirect maintenance:
  - Review redirects quarterly for dead ends
  - Archive old content to `/archive/` if no longer referenced
  - Document rationale for each URL change
  - Communicate deprecated URLs in release notes

#### Rationale

GitHub Pages deployment is automated - build failures block publishing. This is a high-traffic robotics reference. Reliability is essential. As a long-term educational reference, stability and maintenance are not optional—they define trustworthiness.

---

## Content Development Workflow

### Spec-Driven Creation Process

All chapters MUST originate from a formal specification.

#### Process Phases

1. **Specification** (`/sp.specify`): Define chapter scope, learning objectives, prerequisites, key concepts

   - Learning objectives
   - Prerequisites
   - Concepts from the course outline
   - Required diagrams
   - Required examples
   - Foundational theories
   - Required simulation scenes
   - Required datasets

2. **Planning** (`/sp.plan`): Outline structure, identify diagrams needed, plan code examples, research sources

   - Section structure
   - Simulation assets
   - Isaac Sim scenes
   - RealSense datasets

3. **Tasks** (`/sp.tasks`): Decompose into writing tasks (intro, concept sections, examples, exercises)

   - Section-level writing tasks
   - Code tasks
   - Diagram tasks

4. **Implementation**: Write content following constitution principles
5. **Review** (technical + pedagogy): Technical accuracy review + peer review for clarity
6. **Publishing** (CI, PR, merge): Build validation → PR → Merge to main → Auto-deploy

#### Artifacts

- specs/[chapter-name]/spec.md - Chapter requirements and learning objectives
- specs/[chapter-name]/plan.md - Content structure and resource plan
- specs/[chapter-name]/tasks.md - Granular writing tasks
- history/prompts/[chapter-name]/ - AI collaboration records (PHRs)
- history/adr/ - Architectural decisions (e.g., framework choice, chapter organization)

#### Content Types:

- Theory Chapters: Mathematical foundations, algorithms, proofs
- Implementation Chapters: Code walkthroughs, system integration
- Application Chapters: Case studies, real-world examples
- Reference Chapters: API docs, hardware specs, datasets

---

### Architectural Decision Records (ADR)

#### Triggers for ADR Creation

Create an ADR when ALL are true:

- The decision impacts multiple modules (e.g., simulation frameworks).
- Alternatives exist (Gazebo vs Unity, Isaac RL vs Gym).
- The choice affects long-term maintainability.

Additionally, create ADRs for:

- Multi-module impact
- Toolchain changes (ROS 2 version, Isaac Sim version)
- Framework decisions (Unity vs Gazebo)
- Hardware platform standardization (Jetson vs x86)

#### ADR Content Requirements

Each ADR MUST include:

- Context
- Decision
- Alternatives considered
- Tradeoffs
- Consequences

#### Example ADR Decisions

- Selecting Isaac Sim versions
- ROS 2 distribution choice
- Standardizing on Jetson Orin vs Xavier
- Choosing Unity HDRP for visualization
- VLA model integrations (LLMs, Whisper)

Process: Suggest via "📋 Architectural decision detected: [brief]. Document reasoning? Run /sp.adr [title]" - wait for user consent.

---

## Quality Gates & Review Process

### Pre-Merge Gates (NON-NEGOTIABLE)

All of the following MUST pass before merging:

1. Build Validation: Docusaurus build succeeds without warnings
2. Link Check: No broken internal/external links
3. Technical Review: Domain expert validates accuracy (formulas, code, claims)
4. Peer Review: Another contributor checks clarity and consistency
5. Accessibility Check: Alt text present, heading hierarchy correct, contrast ratios meet WCAG AA
6. Performance Check: Lighthouse score ≥ 90 for performance, accessibility, SEO

### Content Review Standards

#### Technical Accuracy

- Kinematics/dynamics formulas validated
- Code tested
- Robot hardware specs verified
- Isaac Sim scene parameters correct
- Correctness of equations
- Accuracy of hardware parameters
- Simulation stability
- Correctness of coordinate frames
- Reproducibility of examples

#### Educational Quality

- Clear learning objectives
- Prerequisites listed
- Worked examples included
- Diagrams for spatial concepts

#### Consistency

- Terminology matches glossary
- Units and notation consistent
- Chapter template followed
- Code style consistent

#### Production

- Image optimization and have alt text
- Metadata complete
- Cross-references accurate
- No placeholder text (TODO, TBD, etc.)

### Required Checks (Comprehensive)

- No structural warnings
- No content placeholders
- Zero dead links
- Notation consistency
- Code executed & validated
- Diagrams clear and appropriately labeled
- All metadata filled

### Review Roles

- Technical Reviewer: Validates accuracy (robotics domain expert)
- Peer Reviewer: Checks clarity and educational value (target audience perspective)
- Editor: Enforces consistency and standards
- Maintainer: Final approval and merge authority

---

## Governance

### Constitution Authority

The Constitution has authority over all writing, code, diagrams, specifications, workflows, and publication processes.

- Overrides all other documents.
- PRs must show compliance with these principles.

### Amendment Process

1. Proposed change + rationale
2. Impact analysis
3. Discussion
4. Approval
5. Version bump
6. Update templates

Alternative process:

1. Proposal
2. Discussion
3. Impact evaluation
4. Acceptance or rejection
5. Version bump
6. Document update

### Compliance Verification

Compliance MUST be enforced during:

- PR checklist
- Quarterly audits for drift
- Deviations require documentation
- Updates propagated quickly
- PR review
- Periodic audits
- Major version updates

### Complexity Justification

Any deviation from:

- Notation
- ROS 2 conventions
- Isaac standards
- SI units

Must be justified with:

- Reason
- Alternatives rejected
- Impact

---

## Appendix: Key References

### Standard Technologies

- **Ubuntu:** 22.04 LTS
- **ROS 2:** Humble or Iron distributions
- **Isaac Sim:** Version-locked (currently tested with 2023.1.1)
- **Gazebo:** Garden/Fortress
- **Python:** 3.10+

### External Standards

- ROS 2 REP (Request for Enhancement) standards
- IEEE robotics research standards
- PEP 8 Python style guide
- SI (International System of Units)

### Key Repositories & Resources

- `docs/glossary.md` — Centralized terminology
- `docs/notation.md` — Mathematical notation schema
- `static/img/[chapter]/` — Image asset storage
- `/examples/[module]/[topic]/` — Code example location

---

Runtime Guidance: Use CLAUDE.md for AI assistant behavior and workflow execution. Constitution defines WHAT we build; CLAUDE.md defines HOW we collaborate with AI agents.

**Last Updated:** December 7, 2025  
**Maintainer:** Physical AI & Humanoid Robotics Textbook Team  
**Status:** Active | Version 2.0.0 (Merged & Enhanced)
