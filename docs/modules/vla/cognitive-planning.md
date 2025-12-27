---
title: "Cognitive Planning"
description: "Using LLMs to translate natural language into ROS 2 action sequences for humanoid robots"
keywords: ["cognitive planning", "llm", "ros2", "actions", "humanoid", "robotics", "vla"]
sidebar_position: 3
---

# Cognitive Planning

This module covers using Large Language Models (LLMs) to translate natural language commands into complex sequences of ROS 2 actions for humanoid robots. This represents the cognitive planning aspect of the Vision-Language-Action (VLA) framework.

## Learning Objectives

By the end of this module, you will be able to:
- Integrate LLMs with ROS 2 for cognitive planning
- Design action planning systems for humanoid robots
- Translate natural language to executable action sequences
- Implement context-aware planning and execution
- Handle planning errors and recovery

## Prerequisites

- ROS 2 fundamentals
- Understanding of action servers and clients
- Basic knowledge of LLMs and NLP
- Voice-to-action system knowledge

## Cognitive Planning Architecture

### High-Level Architecture
```python
# cognitive_planning_architecture.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from builtin_interfaces.msg import Time
import json
import time
from typing import List, Dict, Any

class CognitivePlanningNode(Node):
    def __init__(self):
        super().__init__('cognitive_planning_node')

        # Subscribers for natural language commands
        self.command_sub = self.create_subscription(
            String, 'natural_language_command', self.command_callback, 10)

        # Publishers for action plans
        self.plan_pub = self.create_publisher(String, 'action_plan', 10)
        self.status_pub = self.create_publisher(String, 'planning_status', 10)

        # LLM integration
        self.llm_interface = LLMInterface()

        # Action execution manager
        self.action_manager = ActionExecutionManager(self)

        # Context management
        self.context_manager = ContextManager()

        self.get_logger().info("Cognitive Planning Node initialized")

    def command_callback(self, msg):
        """
        Handle incoming natural language commands
        """
        command_text = msg.data
        self.get_logger().info(f"Received command: {command_text}")

        # Publish planning status
        status_msg = String()
        status_msg.data = f"Processing command: {command_text}"
        self.status_pub.publish(status_msg)

        # Plan and execute
        try:
            # Get current context
            context = self.context_manager.get_context()

            # Generate action plan using LLM
            action_plan = self.llm_interface.generate_action_plan(
                command_text, context)

            # Validate the plan
            if self.validate_plan(action_plan):
                # Publish the action plan
                plan_msg = String()
                plan_msg.data = json.dumps(action_plan)
                self.plan_pub.publish(plan_msg)

                # Execute the plan
                self.action_manager.execute_plan(action_plan)
            else:
                self.get_logger().error("Generated plan is invalid")
                self.publish_error("Invalid action plan generated")

        except Exception as e:
            self.get_logger().error(f"Error processing command: {e}")
            self.publish_error(f"Error processing command: {e}")

    def validate_plan(self, plan):
        """
        Validate that the generated plan is executable
        """
        if not isinstance(plan, list):
            return False

        for action in plan:
            if not isinstance(action, dict) or 'action' not in action:
                return False

            # Check if action is supported
            if action['action'] not in self.action_manager.supported_actions:
                return False

        return True

    def publish_error(self, error_msg):
        """
        Publish error status
        """
        error_status = String()
        error_status.data = f"ERROR: {error_msg}"
        self.status_pub.publish(error_status)
```

### LLM Interface
```python
import openai
import json
import requests
from typing import Dict, List, Any
import time

class LLMInterface:
    def __init__(self, model_name="gpt-3.5-turbo", api_key=None):
        if api_key:
            openai.api_key = api_key
        self.model_name = model_name
        self.local_model = None  # For local LLMs

    def generate_action_plan(self, natural_language, context):
        """
        Generate action plan from natural language using LLM
        """
        # Create detailed prompt for the LLM
        prompt = self.create_planning_prompt(natural_language, context)

        try:
            # Use OpenAI API (or local model in production)
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            # Extract and parse the action plan
            plan_text = response.choices[0].message.content
            action_plan = self.parse_action_plan(plan_text)

            return action_plan

        except Exception as e:
            print(f"Error with LLM processing: {e}")
            # Return fallback plan
            return self.create_fallback_plan(natural_language)

    def create_planning_prompt(self, command, context):
        """
        Create detailed prompt for action planning
        """
        return f"""
        You are a cognitive planning system for a humanoid robot. Your task is to translate natural language commands into sequences of specific actions that the robot can execute.

        Natural Language Command: "{command}"

        Current Context:
        {json.dumps(context, indent=2)}

        Available Actions:
        - move_to_pose: Move to a specific pose (x, y, z, roll, pitch, yaw)
        - move_forward: Move forward by a specified distance
        - turn: Turn by a specified angle
        - pick_object: Pick up an object at a specific location
        - place_object: Place an object at a specific location
        - speak: Speak a message
        - detect_object: Detect objects in the environment
        - grasp: Grasp an object with specified parameters
        - release: Release a grasped object

        Please generate a JSON array of actions in the following format:
        [
            {{
                "action": "action_name",
                "parameters": {{
                    "param1": value1,
                    "param2": value2
                }},
                "description": "Brief description of what this action does",
                "expected_outcome": "What should happen after this action"
            }}
        ]

        Consider the robot's current state, environment, and the goal when creating the plan.
        Be specific about coordinates, distances, and other parameters.
        """

    def get_system_prompt(self):
        """
        System prompt for the LLM
        """
        return """
        You are an expert roboticist and cognitive planner. You generate detailed, executable action plans for humanoid robots based on natural language commands. Your plans should be:
        1. Sequential and logical
        2. Detailed with specific parameters
        3. Safe and executable
        4. Context-aware
        5. Robust to potential failures

        Always return a valid JSON array of actions.
        """

    def parse_action_plan(self, plan_text):
        """
        Parse the LLM response into a structured action plan
        """
        try:
            # Try to find JSON in the response
            start_idx = plan_text.find('[')
            end_idx = plan_text.rfind(']') + 1

            if start_idx != -1 and end_idx != 0:
                json_text = plan_text[start_idx:end_idx]
                action_plan = json.loads(json_text)
                return action_plan
            else:
                # If no JSON found, create a simple plan
                return self.create_simple_plan(plan_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, create fallback plan
            return self.create_fallback_plan(plan_text)

    def create_fallback_plan(self, command):
        """
        Create a fallback plan if LLM fails
        """
        command_lower = command.lower()

        if "move" in command_lower or "go" in command_lower:
            return [{
                "action": "move_forward",
                "parameters": {"distance": 1.0},
                "description": "Move forward 1 meter",
                "expected_outcome": "Robot moves forward"
            }]
        elif "turn" in command_lower:
            return [{
                "action": "turn",
                "parameters": {"angle": 90.0},
                "description": "Turn 90 degrees",
                "expected_outcome": "Robot turns"
            }]
        elif "pick" in command_lower or "grasp" in command_lower:
            return [{
                "action": "grasp",
                "parameters": {},
                "description": "Attempt to grasp object",
                "expected_outcome": "Object grasped"
            }]
        else:
            return [{
                "action": "speak",
                "parameters": {"message": f"I don't know how to execute: {command}"},
                "description": "Speak error message",
                "expected_outcome": "Robot speaks error"
            }]

    def create_simple_plan(self, command):
        """
        Create a simple plan from command text
        """
        return self.create_fallback_plan(command)
```

## Context Management

### Context Manager for Cognitive Planning
```python
import json
from datetime import datetime
from typing import Dict, Any

class ContextManager:
    def __init__(self):
        self.robot_state = {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "battery_level": 100.0,
            "current_gait": "walk",
            "attached_object": None,
            "current_task": None
        }

        self.environment = {
            "known_objects": [],
            "navigation_map": {},
            "obstacles": [],
            "safe_zones": [],
            "forbidden_zones": []
        }

        self.task_history = []
        self.user_preferences = {}

    def get_context(self):
        """
        Get the current context for planning
        """
        context = {
            "timestamp": datetime.now().isoformat(),
            "robot_state": self.robot_state,
            "environment": self.environment,
            "task_history": self.task_history[-10:],  # Last 10 tasks
            "user_preferences": self.user_preferences
        }
        return context

    def update_robot_state(self, new_state):
        """
        Update robot state from sensors and feedback
        """
        self.robot_state.update(new_state)

    def update_environment(self, new_objects=None, new_obstacles=None):
        """
        Update environment information
        """
        if new_objects:
            self.environment["known_objects"].extend(new_objects)

        if new_obstacles:
            self.environment["obstacles"].extend(new_obstacles)

    def add_task_to_history(self, task_description, success=True):
        """
        Add task to history
        """
        task_entry = {
            "timestamp": datetime.now().isoformat(),
            "description": task_description,
            "success": success
        }
        self.task_history.append(task_entry)

    def get_object_location(self, object_name):
        """
        Get location of a known object
        """
        for obj in self.environment["known_objects"]:
            if obj.get("name", "").lower() == object_name.lower():
                return obj.get("position")
        return None

    def is_safe_to_execute(self, action_plan):
        """
        Check if it's safe to execute the action plan
        """
        # Check battery level
        if self.robot_state["battery_level"] < 20.0:
            return False, "Battery level too low"

        # Check for obstacles in path
        for action in action_plan:
            if action["action"] == "move_to_pose":
                target_pos = action["parameters"].get("position", {})
                if self.is_path_blocked(target_pos):
                    return False, f"Path to {target_pos} is blocked"

        return True, "Safe to execute"

    def is_path_blocked(self, target_position):
        """
        Check if path to target is blocked
        """
        # Simple check - in practice this would use path planning
        for obstacle in self.environment["obstacles"]:
            # Calculate distance to obstacle
            dist = self.calculate_distance(
                self.robot_state["position"],
                obstacle.get("position", {})
            )
            if dist < 0.5:  # Within 50cm
                return True
        return False

    def calculate_distance(self, pos1, pos2):
        """
        Calculate distance between two positions
        """
        dx = pos1.get("x", 0) - pos2.get("x", 0)
        dy = pos1.get("y", 0) - pos2.get("y", 0)
        dz = pos1.get("z", 0) - pos2.get("z", 0)
        return (dx*dx + dy*dy + dz*dz)**0.5
```

## Action Execution Management

### Action Execution Manager
```python
import asyncio
import time
from typing import List, Dict, Any
from enum import Enum

class ActionStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ActionExecutionManager:
    def __init__(self, node):
        self.node = node
        self.supported_actions = [
            "move_to_pose", "move_forward", "turn", "pick_object",
            "place_object", "speak", "detect_object", "grasp", "release"
        ]
        self.current_action = None
        self.action_status = ActionStatus.PENDING

    async def execute_plan(self, action_plan: List[Dict[str, Any]]):
        """
        Execute the action plan sequentially
        """
        self.node.get_logger().info(f"Executing action plan with {len(action_plan)} actions")

        for i, action in enumerate(action_plan):
            self.node.get_logger().info(f"Executing action {i+1}/{len(action_plan)}: {action['action']}")

            # Check if action is supported
            if action['action'] not in self.supported_actions:
                self.node.get_logger().error(f"Unsupported action: {action['action']}")
                continue

            # Execute the action
            success = await self.execute_single_action(action)

            if not success:
                self.node.get_logger().error(f"Action failed: {action['action']}")
                # Handle failure - continue, retry, or abort
                break

            # Small delay between actions
            await asyncio.sleep(0.5)

        self.node.get_logger().info("Action plan execution completed")

    async def execute_single_action(self, action: Dict[str, Any]) -> bool:
        """
        Execute a single action and return success status
        """
        action_name = action['action']
        parameters = action.get('parameters', {})

        self.current_action = action
        self.action_status = ActionStatus.EXECUTING

        try:
            if action_name == "move_to_pose":
                return await self.execute_move_to_pose(parameters)
            elif action_name == "move_forward":
                return await self.execute_move_forward(parameters)
            elif action_name == "turn":
                return await self.execute_turn(parameters)
            elif action_name == "pick_object":
                return await self.execute_pick_object(parameters)
            elif action_name == "place_object":
                return await self.execute_place_object(parameters)
            elif action_name == "speak":
                return await self.execute_speak(parameters)
            elif action_name == "detect_object":
                return await self.execute_detect_object(parameters)
            elif action_name == "grasp":
                return await self.execute_grasp(parameters)
            elif action_name == "release":
                return await self.execute_release(parameters)
            else:
                self.node.get_logger().error(f"Unknown action: {action_name}")
                return False

        except Exception as e:
            self.node.get_logger().error(f"Error executing action {action_name}: {e}")
            self.action_status = ActionStatus.FAILED
            return False

    async def execute_move_to_pose(self, params: Dict[str, Any]) -> bool:
        """
        Execute move to pose action
        """
        # This would call navigation services
        target_pose = params.get('pose', {})
        self.node.get_logger().info(f"Moving to pose: {target_pose}")

        # In a real implementation, this would:
        # 1. Send goal to navigation system
        # 2. Monitor progress
        # 3. Return success/failure

        # Simulate execution
        await asyncio.sleep(2.0)  # Simulate movement time

        # For simulation, assume success
        return True

    async def execute_move_forward(self, params: Dict[str, Any]) -> bool:
        """
        Execute move forward action
        """
        distance = params.get('distance', 1.0)
        self.node.get_logger().info(f"Moving forward {distance} meters")

        # Simulate execution
        await asyncio.sleep(1.0)

        return True

    async def execute_turn(self, params: Dict[str, Any]) -> bool:
        """
        Execute turn action
        """
        angle = params.get('angle', 90.0)  # degrees
        self.node.get_logger().info(f"Turning {angle} degrees")

        # Simulate execution
        await asyncio.sleep(1.0)

        return True

    async def execute_pick_object(self, params: Dict[str, Any]) -> bool:
        """
        Execute pick object action
        """
        object_name = params.get('object_name', '')
        position = params.get('position', {})

        self.node.get_logger().info(f"Picking object '{object_name}' at {position}")

        # Simulate execution
        await asyncio.sleep(2.0)

        return True

    async def execute_place_object(self, params: Dict[str, Any]) -> bool:
        """
        Execute place object action
        """
        position = params.get('position', {})

        self.node.get_logger().info(f"Placing object at {position}")

        # Simulate execution
        await asyncio.sleep(2.0)

        return True

    async def execute_speak(self, params: Dict[str, Any]) -> bool:
        """
        Execute speak action
        """
        message = params.get('message', '')

        self.node.get_logger().info(f"Speaking: {message}")

        # In a real implementation, this would use text-to-speech
        # For now, just log the message

        return True

    async def execute_detect_object(self, params: Dict[str, Any]) -> bool:
        """
        Execute detect object action
        """
        object_type = params.get('object_type', 'any')

        self.node.get_logger().info(f"Detecting {object_type} objects")

        # Simulate execution
        await asyncio.sleep(1.0)

        # Update context with detected objects
        # This would involve calling perception services

        return True

    async def execute_grasp(self, params: Dict[str, Any]) -> bool:
        """
        Execute grasp action
        """
        self.node.get_logger().info("Executing grasp action")

        # Simulate execution
        await asyncio.sleep(1.5)

        return True

    async def execute_release(self, params: Dict[str, Any]) -> bool:
        """
        Execute release action
        """
        self.node.get_logger().info("Executing release action")

        # Simulate execution
        await asyncio.sleep(1.0)

        return True
```

## Advanced Cognitive Planning

### Hierarchical Planning
```python
class HierarchicalPlanner:
    def __init__(self):
        self.high_level_planner = HighLevelPlanner()
        self.low_level_planner = LowLevelPlanner()
        self.plan_cache = {}

    def create_hierarchical_plan(self, high_level_goal, context):
        """
        Create a hierarchical plan with high-level and low-level components
        """
        # Check cache first
        cache_key = f"{high_level_goal}_{hash(str(context))}"
        if cache_key in self.plan_cache:
            return self.plan_cache[cache_key]

        # Generate high-level plan
        high_level_plan = self.high_level_planner.generate_plan(
            high_level_goal, context)

        # For each high-level action, generate low-level details
        detailed_plan = []
        for high_action in high_level_plan:
            low_level_actions = self.low_level_planner.generate_detailed_actions(
                high_action, context)
            detailed_plan.extend(low_level_actions)

        # Cache the result
        self.plan_cache[cache_key] = detailed_plan

        return detailed_plan

class HighLevelPlanner:
    def generate_plan(self, goal, context):
        """
        Generate high-level plan (e.g., "Go to kitchen, pick up cup, bring to user")
        """
        # This would use LLM to generate high-level steps
        # For example: "navigate -> detect -> grasp -> navigate -> place"
        return [
            {"action": "navigate", "description": "Go to kitchen"},
            {"action": "detect", "description": "Find the cup"},
            {"action": "grasp", "description": "Pick up the cup"},
            {"action": "navigate", "description": "Return to user"},
            {"action": "place", "description": "Give cup to user"}
        ]

class LowLevelPlanner:
    def generate_detailed_actions(self, high_level_action, context):
        """
        Generate detailed low-level actions for a high-level action
        """
        action_type = high_level_action["action"]

        if action_type == "navigate":
            return self.create_navigation_sequence(high_level_action, context)
        elif action_type == "detect":
            return self.create_detection_sequence(high_level_action, context)
        elif action_type == "grasp":
            return self.create_grasp_sequence(high_level_action, context)
        elif action_type == "place":
            return self.create_placement_sequence(high_level_action, context)
        else:
            return [high_level_action]  # Return as-is if no specific handling

    def create_navigation_sequence(self, action, context):
        """
        Create detailed navigation sequence
        """
        return [
            {
                "action": "localize",
                "parameters": {},
                "description": "Determine current position"
            },
            {
                "action": "plan_path",
                "parameters": {"target": action.get("target", "unknown")},
                "description": "Plan path to target"
            },
            {
                "action": "execute_path",
                "parameters": {"speed": 0.5},
                "description": "Follow planned path"
            },
            {
                "action": "verify_arrival",
                "parameters": {},
                "description": "Confirm arrival at destination"
            }
        ]
```

## Context-Aware Planning

### Context-Aware LLM Integration
```python
class ContextAwareLLM:
    def __init__(self, llm_interface):
        self.llm_interface = llm_interface
        self.context_history = []
        self.max_context_length = 10

    def generate_context_aware_plan(self, command, current_context):
        """
        Generate plan considering historical context
        """
        # Include recent context history
        recent_context = self.context_history[-3:] if self.context_history else []

        # Create enhanced prompt with context
        prompt = self.create_context_enhanced_prompt(
            command, current_context, recent_context)

        # Generate plan with LLM
        action_plan = self.llm_interface.generate_action_plan(
            prompt, current_context)

        # Store this interaction in context history
        self.context_history.append({
            "command": command,
            "context": current_context,
            "plan": action_plan,
            "timestamp": time.time()
        })

        # Keep context history to reasonable size
        if len(self.context_history) > self.max_context_length:
            self.context_history = self.context_history[-self.max_context_length:]

        return action_plan

    def create_context_enhanced_prompt(self, command, current_context, recent_context):
        """
        Create prompt that includes historical context
        """
        context_str = f"Current Context:\n{json.dumps(current_context, indent=2)}\n\n"

        if recent_context:
            context_str += "Recent Interactions:\n"
            for i, interaction in enumerate(recent_context):
                context_str += f"  {i+1}. Command: {interaction['command']}\n"
                context_str += f"     Result: {interaction['plan']}\n\n"

        return f"""
        {context_str}

        Natural Language Command: "{command}"

        Based on the current context and recent interactions, generate an appropriate action plan.
        Consider the robot's previous actions and the user's likely intent based on the conversation history.
        """
```

## Error Handling and Recovery

### Planning Error Handler
```python
class PlanningErrorHandler:
    def __init__(self, node):
        self.node = node
        self.error_history = []
        self.max_retries = 3

    def handle_planning_error(self, error, failed_action, current_plan):
        """
        Handle errors during plan execution
        """
        error_entry = {
            "error": str(error),
            "failed_action": failed_action,
            "timestamp": time.time(),
            "plan_state": current_plan
        }
        self.error_history.append(error_entry)

        # Determine appropriate recovery action
        recovery_action = self.determine_recovery_action(error, failed_action)

        if recovery_action:
            self.node.get_logger().info(f"Attempting recovery: {recovery_action}")
            return recovery_action
        else:
            self.node.get_logger().error(f"Cannot recover from error: {error}")
            return None

    def determine_recovery_action(self, error, failed_action):
        """
        Determine appropriate recovery action based on error type
        """
        error_str = str(error).lower()

        if "navigation" in error_str or "path" in error_str:
            return {
                "action": "find_alternative_path",
                "parameters": {"original_target": failed_action.get("parameters", {}).get("target")},
                "description": "Try alternative navigation route"
            }
        elif "grasp" in error_str or "object" in error_str:
            return {
                "action": "reposition_and_retry",
                "parameters": {"action_to_retry": failed_action},
                "description": "Adjust position and try grasp again"
            }
        elif "timeout" in error_str:
            return {
                "action": "increase_timeout_and_retry",
                "parameters": {"action_to_retry": failed_action, "multiplier": 2.0},
                "description": "Retry with longer timeout"
            }
        elif "collision" in error_str:
            return {
                "action": "clear_path_and_retry",
                "parameters": {"action_to_retry": failed_action},
                "description": "Clear obstacle and retry"
            }
        else:
            return {
                "action": "request_human_assistance",
                "parameters": {"error_description": str(error)},
                "description": "Request human intervention"
            }

    def can_retry_action(self, failed_action):
        """
        Determine if an action can be retried
        """
        # Check how many times this action has failed
        recent_failures = [
            err for err in self.error_history
            if err["failed_action"].get("action") == failed_action.get("action")
            and time.time() - err["timestamp"] < 300  # Last 5 minutes
        ]

        return len(recent_failures) < self.max_retries
```

## Learning and Adaptation

### Plan Learning System
```python
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class PlanLearningSystem:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()
        self.command_plan_pairs = []
        self.is_trained = False

    def record_successful_plan(self, command, plan, context=None):
        """
        Record successful command-plan pairs for learning
        """
        self.command_plan_pairs.append({
            "command": command,
            "plan": plan,
            "context": context or {},
            "success": True
        })

    def train_from_history(self):
        """
        Train the system from historical successful plans
        """
        if len(self.command_plan_pairs) < 10:  # Need minimum examples
            return

        commands = [pair["command"] for pair in self.command_plan_pairs]
        labels = [hash(str(pair["plan"])) for pair in self.command_plan_pairs]  # Simplified

        # Vectorize commands
        X = self.vectorizer.fit_transform(commands)

        # Train classifier
        self.classifier.fit(X, labels)
        self.is_trained = True

        print(f"Trained on {len(self.command_plan_pairs)} examples")

    def suggest_plan(self, command):
        """
        Suggest a plan based on learned patterns
        """
        if not self.is_trained:
            return None

        # Vectorize the command
        X = self.vectorizer.transform([command])

        # Predict most similar plan
        predicted_label = self.classifier.predict(X)[0]

        # Find the most similar plan from history
        for pair in self.command_plan_pairs:
            if hash(str(pair["plan"])) == predicted_label:
                return pair["plan"]

        return None

    def save_model(self, filepath):
        """
        Save the trained model
        """
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'command_plan_pairs': self.command_plan_pairs,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath):
        """
        Load a trained model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.command_plan_pairs = model_data['command_plan_pairs']
        self.is_trained = model_data['is_trained']
```

## Performance Optimization

### Optimized Planning Pipeline
```python
import asyncio
import concurrent.futures
from functools import lru_cache

class OptimizedPlanningPipeline:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.plan_cache = {}
        self.context_cache_ttl = 300  # 5 minutes

    @lru_cache(maxsize=128)
    def get_cached_plan(self, command_hash, context_hash):
        """
        Get cached plan if available
        """
        cache_key = f"{command_hash}_{context_hash}"
        return self.plan_cache.get(cache_key)

    def cache_plan(self, command, context, plan):
        """
        Cache a generated plan
        """
        command_hash = hash(command)
        context_hash = hash(str(context))
        cache_key = f"{command_hash}_{context_hash}"

        self.plan_cache[cache_key] = {
            "plan": plan,
            "timestamp": time.time(),
            "command": command
        }

    async def generate_optimized_plan(self, command, context):
        """
        Generate plan with optimization techniques
        """
        # Check cache first
        command_hash = hash(command)
        context_hash = hash(str(context))

        cached = self.get_cached_plan(command_hash, context_hash)
        if cached:
            print("Using cached plan")
            return cached

        # Use thread pool for LLM processing
        loop = asyncio.get_event_loop()
        plan = await loop.run_in_executor(
            self.executor,
            self.generate_plan_sync,
            command, context
        )

        # Cache the result
        self.cache_plan(command, context, plan)

        return plan

    def generate_plan_sync(self, command, context):
        """
        Synchronous plan generation (for threading)
        """
        # This would call the actual LLM interface
        llm_interface = LLMInterface()
        return llm_interface.generate_action_plan(command, context)
```

## Best Practices

### Planning Best Practices
1. **Context Awareness**: Always consider current state and environment
2. **Safety First**: Validate plans for safety before execution
3. **Modularity**: Break complex tasks into smaller, manageable actions
4. **Error Handling**: Plan for potential failures and recovery
5. **Learning**: Use past experiences to improve future planning

### Performance Considerations
- Cache frequently used plans
- Use hierarchical planning for complex tasks
- Implement parallel execution where safe
- Monitor and optimize planning time

### Safety Measures
- Validate all generated plans
- Implement safety zones and limits
- Use simulation before real execution
- Provide manual override capabilities

## Troubleshooting Common Issues

### Planning Problems
- **Overly Complex Plans**: Break down into simpler steps
- **Context Ignoring**: Ensure context is properly integrated
- **Safety Violations**: Add more safety checks
- **Performance Issues**: Optimize with caching and parallelization

### LLM-Specific Issues
- **Inconsistent Outputs**: Use structured prompting
- **Hallucinations**: Add validation steps
- **Latency**: Implement caching and local models

## Advanced Topics

### Multi-Agent Coordination
- Coordinate multiple robots for complex tasks
- Handle inter-robot communication
- Manage shared resources and conflicts

### Learning from Demonstration
- Learn new plans from human demonstrations
- Generalize from specific examples
- Adapt to new situations

### Adaptive Planning
- Adjust plans based on real-time feedback
- Handle unexpected situations
- Learn from plan execution outcomes

## Next Steps

After mastering cognitive planning, explore [Capstone Project: Autonomous Humanoid](/docs/modules/vla/capstone-project) to learn how to integrate all VLA components for a complete autonomous humanoid robot system.