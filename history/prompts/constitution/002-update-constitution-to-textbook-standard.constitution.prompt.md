---
id: 002
title: Update Constitution to Textbook Standard
stage: constitution
date: 2025-12-07
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: User
command: /sp.constitution
labels: ["constitution", "update"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:

---

## Prompt

```text
# Textbook Constitution Generator Prompt (Ready-to-Use)

**Prompt:**

Generate a **Textbook Constitution** for a "Physical AI & Humanoid Robotics" textbook. The constitution should define governance, rules, standards, workflows, quality assurance principles, and publishing guidelines. The output must be **fully structured in Markdown**, professional, hierarchical, and include rules, rationale, and examples. The format should be ready to serve as a reference for contributors, reviewers, and maintainers.

---

## Required Sections

### 1. Metadata
- **Version:** 2.0.0
- **Ratification Date:** 2025-12-07
- **Authority:** Governs all content, review, and publishing practices for the textbook and supporting materials

### 2. Table of Contents
- Core Principles
- Content Development Workflow
- Quality Gates & Review Process
- Governance
- Appendix & References

### 3. Core Principles

#### I. Content Accuracy & Technical Rigor
- Validate all robotics, AI, simulation, control, SLAM, perception, and humanoid hardware content using authoritative sources.
- Cross-verify formulas, kinematics, dynamics, optimization, and control laws.
- Ensure all code runs in the defined environment: Ubuntu 22.04, ROS 2 Humble/Iron, Python 3.10+, Isaac Sim 2023.1.1, Gazebo Garden/Fortress, Unity HDRP.
- Simulation and hardware parameters must be physically plausible and documented.
- Cite all research claims; label speculative research clearly.
- Provide derivations or experimental justification for stability, CoM modeling, whole-body control, pose estimation, and SLAM drift.

#### II. Educational Clarity & Accessibility
- List prerequisites, target audience, and measurable learning objectives for each chapter.
- Use a layered explanation approach: **Motivation → Intuition → Formalism → Real System Example**.
- Include diagrams, flowcharts, analogies, and worked examples.
- Provide bridging explanations connecting theory → simulation → code → hardware.
- Ensure accessibility: glossary linking, alt text, consistent headings.

#### III. Consistency & Standards
- Enforce terminology, notation, and units according to centralized glossary and `docs/notation.md`.
- Maintain consistent code style (PEP 8 + ROS 2 conventions).
- Chapters follow a uniform template: objectives, prerequisites, content, examples, exercises, references.
- Use consistent voice: theory in third person, tutorials in second person.

#### IV. Content Platform & Navigation
- One core concept per page (≤2000 words)
- Include page metadata for SEO and indexing
- Organize sidebar by course progression: Foundations → ROS 2 → Simulation → Isaac Perception & Control → VLA → Humanoid Mechanics → Capstone
- Store images in `static/img/[chapter]/` with descriptive names and alt text
- Internal links must use keywords in anchor text

#### V. Code & Example Quality
- Use Python + rclpy for ROS 2, Isaac Sim Python API, Gazebo Python/ROS, Unity C# if necessary
- All examples must run without modification, include dependencies with versions, and explain **why** in comments
- Provide validation scripts, unit tests, and safety warnings for hardware code
- Structure examples: `/examples/[chapter]/[example]/` with `main.py`, `test_*.py`, `README.md`, `requirements.txt`

#### VI. Deployment & Publishing Standards
- Main branch contains only production-ready content
- Use feature branches with naming convention: `chapter/...`, `fix/...`, `example/...`, `docs/...`
- PRs must include description, checklist, related issues, and evidence of testing
- CI/CD checks: build validation, broken links, spellcheck, code execution, dependency version checks, safety warnings, image optimization
- Performance targets: initial load <3s, LCP <2.5s, CLS <0.1
- Versioning: semantic, with major, minor, and patch updates
- Maintain URL redirects for deprecated content

---

### 4. Content Development Workflow
- **Spec-Driven Creation Process:** Specification → Planning → Tasks → Implementation → Review → Publishing
- Artifacts: `specs/[chapter]/spec.md`, `plan.md`, `tasks.md`, ADRs, diagrams, examples
- Content Types: Theory, Implementation, Application, Reference

---

### 5. Architectural Decision Records (ADR)
- Create ADR when decisions affect multiple modules, frameworks, toolchains, or hardware platforms
- ADR must include: Context, Decision, Alternatives, Tradeoffs, Consequences
- Examples: Isaac Sim version choice, ROS 2 distribution, Jetson hardware standardization

---

### 6. Quality Gates & Review Process
- Pre-merge gates: Docusaurus build, link check, technical review, peer review, accessibility, performance
- Technical accuracy: kinematics, dynamics, hardware specs, code, simulations
- Educational quality: clear objectives, prerequisites, examples, diagrams
- Consistency: terminology, notation, chapter templates, code style
- Review roles: technical reviewer, peer reviewer, editor, maintainer

---

### 7. Governance
- Constitution overrides all other documents
- Amendment process: propose → impact analysis → discussion → approval → version bump → update templates
- Compliance verification through PR checklists and periodic audits
- Deviations from standards require justification with alternatives considered and impact

---

### 8. Appendix & References
- Standard technologies: Ubuntu 22.04, ROS 2 Humble/Iron, Isaac Sim 2023.1.1, Gazebo Garden/Fortress, Python 3.10+
- External standards: ROS 2 REP, IEEE robotics, PEP 8, SI units
- Key Repositories: `docs/glossary.md`, `docs/notation.md`, `/examples/[module]/[topic]/`, `static/img/[chapter]/`

---

### Output Requirements
- Use **Markdown headings and subheadings**
- Number rules, use bullets for clarity
- Provide rationale for rules where appropriate
- Include example templates or snippets
- Maintain professional and educational tone
- Fully structured and ready as a contributor reference
```

## Response snapshot

```text
The constitution has been successfully updated to version 2.0.0.

The following files require manual follow-up to ensure consistency:
- .specify/templates/plan-template.md
- .specify/templates/spec-template.md
- .specify/templates/tasks-template.md

Suggested commit message: `docs: amend constitution to v2.0.0 (initial textbook constitution)`
```

## Outcome

- ✅ Impact: The project constitution has been successfully updated with the detailed guidelines for the "Physical AI & Humanoid Robotics" textbook.
- 🧪 Tests: No new tests were executed as part of this process.
- 📁 Files: Updated .specify/memory/constitution.md. PHR created at history/prompts/constitution/002-update-constitution-to-textbook-standard.constitution.prompt.md
- 🔁 Next prompts: User to review updated constitution and proceed with further development tasks.
- 🧠 Reflection: Successfully generated and applied the detailed textbook constitution. Noted the pending updates for other templates to maintain consistency.

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): N/A