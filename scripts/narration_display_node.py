#!/usr/bin/env python3
"""
Narration Display Node

A ROS2 node that listens to narration image and text topics and saves them
as images with text overlay to a specified directory.

Topics:
  - /narration_image (sensor_msgs/Image) - The image to display
  - /narration_text (std_msgs/String) - The text to overlay on the image
  - /vlm_answer (std_msgs/String) - The VLM's answer to the object identification query
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import time
from datetime import datetime
import threading
import tempfile
import base64
import io
import json
import math
import string
from PIL import Image as PILImage
from openai import OpenAI

class NarrationDisplayNode(Node):
    def __init__(self):
        super().__init__('narration_display_node')
        
        # Professional startup message
        self.get_logger().info("=" * 60)
        self.get_logger().info("NARRATION DISPLAY SYSTEM INITIALIZING")
        self.get_logger().info("=" * 60)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Synchronization
        self.lock = threading.Lock()
        self.image_counter = 0
        
        # Store latest messages with timestamps
        self.latest_image = None
        self.latest_image_time = None
        self.latest_text = None
        self.latest_text_time = None
        
        # Synchronization tolerance (seconds)
        self.sync_tolerance = 0.5  # 500ms tolerance for image/text sync
        
        # Create output directory
        self.output_dir = os.path.expanduser("~/narration_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # VLM API settings
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4o-mini"
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 
            '/narration_image',
            self.image_callback, 
            10
        )
        self.text_sub = self.create_subscription(
            String, 
            '/narration_text', 
            self.text_callback, 
            10
        )
        
        # Publisher for VLM answers
        self.vlm_answer_pub = self.create_publisher(
            String,
            '/vlm_answer',
            10
        )
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("NARRATION DISPLAY SYSTEM READY")
        self.get_logger().info("=" * 60)
        
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.get_logger().info(f"Subscribing to: /narration_image, /narration_text")
        self.get_logger().info(f"Publishing to: /vlm_answer")
        self.get_logger().info(f"Sync tolerance: {self.sync_tolerance}s")
        
        if self.api_key:
            self.get_logger().info("VLM API key found - VLM integration enabled")
        else:
            self.get_logger().warn("No OPENAI_API_KEY found - VLM integration disabled")

    def image_callback(self, msg):
        """Handle incoming image messages"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Use current time for both image and text to ensure same time base
            msg_timestamp = time.time()
            
            with self.lock:
                self.latest_image = cv_image.copy()
                self.latest_image_time = msg_timestamp
            
            self.get_logger().debug(f"Received image at {msg_timestamp:.3f}")
            
            # Try to save if we have matching text
            self.try_save_synchronized()
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def text_callback(self, msg):
        """Handle incoming text messages"""
        try:
            # Use current time for both image and text to ensure same time base
            msg_timestamp = time.time()
            
            with self.lock:
                self.latest_text = msg.data
                self.latest_text_time = msg_timestamp
            
            self.get_logger().debug(f"Received text: '{msg.data}' at {msg_timestamp:.3f}")
            
            # Try to save if we have matching image
            self.try_save_synchronized()
            
        except Exception as e:
            self.get_logger().error(f"Error processing text: {e}")

    def try_save_synchronized(self):
        """Try to save image with text if they are synchronized"""
        with self.lock:
            # Check if we have both image and text
            if (self.latest_image is None or 
                self.latest_text is None or 
                self.latest_image_time is None or 
                self.latest_text_time is None):
                return
            
            # Check if they are synchronized (within tolerance)
            time_diff = abs(self.latest_image_time - self.latest_text_time)
            
            if time_diff <= self.sync_tolerance:
                # They are synchronized - save the image with text
                self.save_image_with_text(self.latest_image, self.latest_text)
                
                # Clear the stored messages after saving
                self.latest_image = None
                self.latest_image_time = None
                self.latest_text = None
                self.latest_text_time = None
                
                self.get_logger().info(f"Saved synchronized image+text (time diff: {time_diff:.3f}s)")
            else:
                self.get_logger().debug(f"Image and text not synchronized (time diff: {time_diff:.3f}s)")

    def encode_image_for_api(self, cv_image, max_dim: int = 768) -> str:
        """Convert OpenCV image to base64 for API"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = PILImage.fromarray(rgb_image)
        
        # Resize while keeping aspect ratio
        pil_image.thumbnail((max_dim, max_dim))
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def query_vlm(self, image, narration_text):
        """Query VLM with double prompt mechanism: get objects then score them
        
        Returns:
            tuple: (most_probable_answer, score) or (error_message, 0.0) on failure
        """
        if not self.api_key:
            self.get_logger().warning("No API key available - skipping VLM query")
            return "VLM not available", 0.0
        
        try:
            client = OpenAI(api_key=self.api_key)
            image_base64 = self.encode_image_for_api(image)
            data_url = f"data:image/png;base64,{image_base64}"
            
            # STEP 1: Get 15 distinct objects from the image
            objects = self._get_object_list(client, data_url)
            if not objects or len(objects) == 0:
                return "VLM Error: No objects detected", 0.0
            
            # STEP 2: Score objects using logprobs
            base_prompt = "I am a drone, after 1s"
            full_narration = f"{base_prompt} {narration_text}"
            most_probable, score = self._score_objects_with_logprobs(client, data_url, objects, full_narration)
            
            self.get_logger().info(f"VLM answer: {most_probable} (score: {score:.4f})")
            return most_probable, score
            
        except Exception as e:
            self.get_logger().error(f"Error querying VLM: {e}")
            return f"VLM Error: {str(e)}", 0.0
    
    def _get_object_list(self, client, data_url, retries=3):
        """Get 15 distinct objects from image"""
        prompt = """You must output EXACTLY 15 distinct object types from the image.

RESPONSE FORMAT RULES:
- Output MUST be ONLY a JSON array.
- EXACTLY 15 strings.
- No markdown.
- No backticks.
- No explanation.
- No extra objects."""
        
        for attempt in range(retries):
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Follow the rules exactly."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                max_tokens=300,
            )
            
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.strip("`").replace("json", "", 1).strip()
            
            try:
                object_list = json.loads(raw)
                if isinstance(object_list, list) and len(object_list) >= 15:
                    return object_list[:15]
                elif isinstance(object_list, list) and len(object_list) > 0:
                    return object_list  # Return what we have
            except json.JSONDecodeError:
                continue
        
        return []
    
    def _score_objects_with_logprobs(self, client, data_url, objects, narration):
        """Score objects using logprobs and return most probable with score
        
        Returns:
            tuple: (most_probable_object, probability_score)
        """
        letters = string.ascii_uppercase[:len(objects)]
        options = {letter: obj for letter, obj in zip(letters, objects)}
        options_text = "\n".join([f"{letter}. {obj}" for letter, obj in options.items()])
        
        prompt = f"""{narration}

Looking at this image, which object most likely caused the drift?

ANSWER OPTIONS:
{options_text}

Output the Letter corresponding to the cause.

Formatting rules:
- Respond with EXACTLY one uppercase letter from A to {letters[-1]}.
- No spaces, punctuation, or newlines."""
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a drone flight dynamics expert. Analyze the image and choose the object that likely caused drift.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            max_tokens=1,
            temperature=2.0,
            logprobs=True,
            top_logprobs=max(20, len(options) + 5),
            stop=["\n"],
        )
        
        # Extract logprobs and compute scores
        logprob_entry = response.choices[0].logprobs.content[0]
        actual_token = logprob_entry.token
        actual_logprob = logprob_entry.logprob
        
        raw_scores = {}
        actual_clean = actual_token.strip().upper()
        if actual_clean in options:
            raw_scores[actual_clean] = math.exp(actual_logprob)
        
        for item in logprob_entry.top_logprobs:
            tok_clean = item.token.strip().upper()
            if tok_clean in options and tok_clean not in raw_scores:
                raw_scores[tok_clean] = math.exp(item.logprob)
        
        # Return most probable object with its score
        if raw_scores:
            most_likely_letter, score = max(raw_scores.items(), key=lambda x: x[1])
            return options[most_likely_letter], float(score)
        
        # Fallback to sampled answer with exp(logprob) as score
        fallback_score = math.exp(actual_logprob) if actual_clean in options else 0.0
        return options.get(actual_clean, objects[0]), float(fallback_score)

    def publish_vlm_answer(self, answer):
        """Publish VLM answer to ROS topic"""
        try:
            msg = String()
            msg.data = answer
            self.vlm_answer_pub.publish(msg)
            self.get_logger().info(f"Published VLM answer: {answer}")
        except Exception as e:
            self.get_logger().error(f"Error publishing VLM answer: {e}")

    def add_text_to_image(self, image, text):
        """Add text overlay to image"""
        if image is None:
            return None
        
        # Create a copy to avoid modifying the original
        display_image = image.copy()
        
        # Split text into lines (max 50 characters per line)
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= 50:
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Calculate text position
        height, width = display_image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        line_height = int(font_scale * 30)
        
        # Calculate total text height
        total_text_height = len(lines) * line_height
        start_y = max(20, height - total_text_height - 20)
        
        # Draw semi-transparent background for text
        overlay = display_image.copy()
        cv2.rectangle(overlay, (10, start_y - 10), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_image, 0.3, 0, display_image)
        
        # Draw text
        for i, line in enumerate(lines):
            y = start_y + i * line_height
            # Get text size for centering
            (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            x = max(20, (width - text_width) // 2)
            
            # Draw text with white color
            cv2.putText(display_image, line, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        return display_image

    def save_image_with_text(self, image, narration_text):
        """Save image with text overlay to file"""
        try:
            # Query VLM with the narration
            vlm_answer, vlm_score = self.query_vlm(image, narration_text)
            
            # Publish VLM answer with score (format: "answer|score")
            vlm_message = f"{vlm_answer}|{vlm_score:.6f}"
            self.publish_vlm_answer(vlm_message)
            
            # Create the full text to display
            base_prompt = "I am a drone, after 1s"
            full_narration = f"{base_prompt} {narration_text}"
            full_text = f"{full_narration}\n\nVLM Answer: {vlm_answer} (score: {vlm_score:.4f})"
            
            # Add text to image
            display_image = self.add_text_to_image(image, full_text)
            
            if display_image is None:
                self.get_logger().error("Failed to create display image")
                return
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"narration_{timestamp}_{self.image_counter:04d}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, display_image)
            
            self.image_counter += 1
            self.get_logger().info(f"Saved narration image with VLM answer: {filepath}")
            
        except Exception as e:
            self.get_logger().error(f"Error saving image: {e}")

def main():
    rclpy.init()
    node = NarrationDisplayNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main() 