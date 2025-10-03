"""
DINO Pipeline - GraspNet 6D Grasp Prediction Module
Implements 6D grasp pose estimation for RGB-D data
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import logging
import math

# Try to import GraspNet - fallback if not available
try:
    # This would be the actual GraspNet import
    # import graspnetAPI
    # from graspnet import GraspNet
    GRASPNET_AVAILABLE = False  # Set to False since GraspNet is complex to install
except ImportError:
    GRASPNET_AVAILABLE = False

class GraspNetPredictor:
    """
    6D Grasp prediction using GraspNet or similar approach
    Generates grasp poses for detected objects in RGB-D scenes
    """
    
    def __init__(self, device: str = "auto", num_grasps_per_object: int = 5):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.num_grasps_per_object = num_grasps_per_object
        self.model = None
        
        if GRASPNET_AVAILABLE:
            self._init_graspnet()
            self.model_type = "graspnet"
        else:
            self._init_fallback()
            self.model_type = "fallback"
            
        print(f"✅ GraspNet Predictor initialized using: {self.model_type}")
    
    def _init_graspnet(self):
        """Initialize actual GraspNet model"""
        # This would be the actual GraspNet initialization
        # self.model = GraspNet(checkpoint_path="path/to/graspnet/checkpoint")
        # self.model.to(self.device)
        # self.model.eval()
        pass
    
    def _init_fallback(self):
        """Initialize fallback grasp predictor"""
        print("⚠️ Using fallback grasp predictor (geometric grasps)")
        
        # Define grasp parameters
        self.gripper_width = 0.08  # 8cm gripper width
        self.grasp_depth_offset = 0.02  # 2cm from surface
        
        # Camera intrinsic parameters (typical values)
        self.fx = 525.0  # focal length x
        self.fy = 525.0  # focal length y
        self.cx = 320.0  # principal point x
        self.cy = 240.0  # principal point y
    
    def predict_grasps(self, rgb_image: np.ndarray, depth_image: np.ndarray, 
                      detections: List[Dict]) -> List[Dict]:
        """
        Predict 6D grasps for detected objects
        
        Args:
            rgb_image: RGB image (H, W, 3)
            depth_image: Depth image (H, W) in meters
            detections: List of object detections
            
        Returns:
            List of grasp predictions with 6D poses
        """
        if self.model_type == "graspnet":
            return self._predict_graspnet(rgb_image, depth_image, detections)
        else:
            return self._predict_fallback(rgb_image, depth_image, detections)
    
    def _predict_graspnet(self, rgb_image: np.ndarray, depth_image: np.ndarray, 
                         detections: List[Dict]) -> List[Dict]:
        """Predict grasps using actual GraspNet"""
        # This would be the actual GraspNet prediction
        # point_cloud = self._rgbd_to_pointcloud(rgb_image, depth_image)
        # grasp_group = self.model.predict(point_cloud)
        # return self._process_grasp_group(grasp_group, detections)
        pass
    
    def _predict_fallback(self, rgb_image: np.ndarray, depth_image: np.ndarray, 
                         detections: List[Dict]) -> List[Dict]:
        """Predict grasps using geometric fallback approach"""
        all_grasps = []
        
        # Convert depth to proper format if needed
        if depth_image.dtype == np.uint8:
            # If depth is stored as uint8, assume it's normalized
            depth_image = depth_image.astype(np.float32) / 255.0 * 2.0  # Assume 2m max depth
        
        for detection in detections:
            object_grasps = self._generate_object_grasps(
                rgb_image, depth_image, detection
            )
            all_grasps.extend(object_grasps)
        
        return all_grasps
    
    def _generate_object_grasps(self, rgb_image: np.ndarray, depth_image: np.ndarray, 
                               detection: Dict) -> List[Dict]:
        """Generate multiple grasp poses for a single object"""
        grasps = []
        
        # Get object region
        x, y, w, h = detection["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)  # Ensure integer coordinates
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Sample depth at object center (with some robustness)
        depth_region = depth_image[max(0, y):min(depth_image.shape[0], y+h),
                                  max(0, x):min(depth_image.shape[1], x+w)]
        
        if depth_region.size == 0:
            print(f"Warning: Empty depth region for {detection['label']}")
            return grasps
        
        # Get median depth (more robust than mean)
        valid_depths = depth_region[depth_region > 0]
        if len(valid_depths) == 0:
            print(f"Warning: No valid depth values for {detection['label']}")
            return grasps
        
        object_depth = np.median(valid_depths)
        
        # Convert to 3D point
        object_3d = self._pixel_to_3d(center_x, center_y, object_depth)
        
        # Generate multiple grasp orientations
        for i in range(self.num_grasps_per_object):
            # Vary grasp angle around object
            angle = (2 * math.pi * i) / self.num_grasps_per_object
            
            # Generate 6D pose
            grasp_pose = self._generate_6d_pose(object_3d, angle, detection)
            
            # Calculate grasp quality score
            quality_score = self._calculate_grasp_quality(
                rgb_image, depth_image, detection, grasp_pose
            )
            
            grasp = {
                "object_id": len(grasps),
                "object_label": detection["label"],
                "pose": grasp_pose,
                "quality_score": quality_score,
                "gripper_width": self.gripper_width,
                "approach_vector": grasp_pose["approach_vector"],
                "grasp_center": object_3d,
                "confidence": quality_score * detection["confidence"]
            }
            
            grasps.append(grasp)
        
        # Sort by quality score
        grasps.sort(key=lambda g: g["quality_score"], reverse=True)
        
        return grasps
    
    def _pixel_to_3d(self, u: int, v: int, depth: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates to 3D point"""
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return (x, y, z)
    
    def _generate_6d_pose(self, object_3d: Tuple[float, float, float], 
                         angle: float, detection: Dict) -> Dict:
        """Generate a 6D grasp pose"""
        x, y, z = object_3d
        
        # Approach vector (pointing towards object)
        approach_x = -math.sin(angle)
        approach_y = -math.cos(angle)  
        approach_z = -0.2  # Slight downward approach
        
        # Normalize approach vector
        approach_length = math.sqrt(approach_x**2 + approach_y**2 + approach_z**2)
        approach_vector = (
            approach_x / approach_length,
            approach_y / approach_length, 
            approach_z / approach_length
        )
        
        # Grasp position (offset from object center)
        grasp_offset = 0.05  # 5cm offset
        grasp_x = x + approach_vector[0] * grasp_offset
        grasp_y = y + approach_vector[1] * grasp_offset
        grasp_z = z + approach_vector[2] * grasp_offset
        
        # Calculate rotation matrix (simplified)
        # In a real implementation, this would be more sophisticated
        rotation_matrix = self._calculate_rotation_matrix(approach_vector, angle)
        
        pose = {
            "position": (grasp_x, grasp_y, grasp_z),
            "approach_vector": approach_vector,
            "rotation_matrix": rotation_matrix,
            "angle": angle,
            "quaternion": self._matrix_to_quaternion(rotation_matrix)
        }
        
        return pose
    
    def _calculate_rotation_matrix(self, approach_vector: Tuple[float, float, float], 
                                 angle: float) -> np.ndarray:
        """Calculate rotation matrix for grasp pose"""
        # Simplified rotation matrix calculation
        # In practice, this would be more sophisticated
        ax, ay, az = approach_vector
        
        # Create a simple rotation matrix
        # This is a simplified version - real implementation would be more complex
        R = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ])
        
        return R
    
    def _matrix_to_quaternion(self, rotation_matrix: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert rotation matrix to quaternion (w, x, y, z)"""
        # Simplified quaternion conversion
        # Real implementation would handle all cases properly
        R = rotation_matrix
        
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            # Simplified case
            w, x, y, z = 1.0, 0.0, 0.0, 0.0
        
        return (w, x, y, z)
    
    def _calculate_grasp_quality(self, rgb_image: np.ndarray, depth_image: np.ndarray,
                                detection: Dict, grasp_pose: Dict) -> float:
        """Calculate grasp quality score based on various factors"""
        quality = 0.5  # Base quality
        
        # Factor 1: Object size (larger objects might be easier to grasp)
        x, y, w, h = detection["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)  # Ensure integer coordinates
        object_area = w * h
        image_area = rgb_image.shape[0] * rgb_image.shape[1]
        size_factor = min(object_area / image_area * 10, 1.0)  # Normalize
        
        # Factor 2: Depth consistency (objects with consistent depth are better)
        depth_region = depth_image[max(0, y):min(depth_image.shape[0], y+h),
                                  max(0, x):min(depth_image.shape[1], x+w)]
        if depth_region.size > 0:
            valid_depths = depth_region[depth_region > 0]
            if len(valid_depths) > 0:
                depth_std = np.std(valid_depths)
                depth_consistency = max(0, 1.0 - depth_std * 5)  # Lower std = better
            else:
                depth_consistency = 0.1
        else:
            depth_consistency = 0.1
        
        # Factor 3: Detection confidence
        detection_confidence = detection["confidence"]
        
        # Factor 4: Grasp angle (some angles might be better)
        angle_factor = 0.8 + 0.2 * math.sin(grasp_pose["angle"] * 2)  # Vary between 0.6-1.0
        
        # Combine factors
        quality = (0.3 * size_factor + 
                  0.3 * depth_consistency + 
                  0.3 * detection_confidence + 
                  0.1 * angle_factor)
        
        return min(quality, 1.0)
    
    def visualize_grasps(self, rgb_image: np.ndarray, grasps: List[Dict], 
                        save_path: str = None) -> np.ndarray:
        """
        Visualize 6D grasps projected onto 2D image
        
        Args:
            rgb_image: Input RGB image
            grasps: List of grasp predictions
            save_path: Optional path to save visualization
            
        Returns:
            Annotated image with grasp visualizations
        """
        vis_image = rgb_image.copy()
        
        # Group grasps by object
        object_grasps = {}
        for grasp in grasps:
            obj_label = grasp["object_label"]
            if obj_label not in object_grasps:
                object_grasps[obj_label] = []
            object_grasps[obj_label].append(grasp)
        
        # Color map for different objects
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255)]
        
        for obj_idx, (obj_label, obj_grasps) in enumerate(object_grasps.items()):
            color = colors[obj_idx % len(colors)]
            
            # Show only top 3 grasps per object for clarity
            for grasp in obj_grasps[:3]:
                self._draw_grasp_2d(vis_image, grasp, color)
        
        # Add legend
        y_offset = 30
        for obj_idx, obj_label in enumerate(object_grasps.keys()):
            color = colors[obj_idx % len(colors)]
            cv2.putText(vis_image, f"{obj_label} grasps", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"✅ Grasp visualization saved to: {save_path}")
        
        return vis_image
    
    def _draw_grasp_2d(self, image: np.ndarray, grasp: Dict, color: Tuple[int, int, int]):
        """Draw a single grasp as 2D projection"""
        # Project 3D grasp center to 2D
        x_3d, y_3d, z_3d = grasp["grasp_center"]
        
        # Simple perspective projection
        if z_3d > 0:
            u = int(self.fx * x_3d / z_3d + self.cx)
            v = int(self.fy * y_3d / z_3d + self.cy)
            
            # Draw grasp center
            cv2.circle(image, (u, v), 5, color, -1)
            
            # Draw grasp orientation (simplified)
            angle = grasp["pose"]["angle"]
            length = 30
            end_u = int(u + length * math.cos(angle))
            end_v = int(v + length * math.sin(angle))
            
            cv2.line(image, (u, v), (end_u, end_v), color, 2)
            
            # Draw gripper jaws (simplified rectangle)
            jaw_width = 20
            jaw_height = 5
            
            # Calculate jaw corners
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            corners = [
                (u - jaw_width//2 * cos_a - jaw_height//2 * sin_a,
                 v - jaw_width//2 * sin_a + jaw_height//2 * cos_a),
                (u + jaw_width//2 * cos_a - jaw_height//2 * sin_a,
                 v + jaw_width//2 * sin_a + jaw_height//2 * cos_a),
                (u + jaw_width//2 * cos_a + jaw_height//2 * sin_a,
                 v + jaw_width//2 * sin_a - jaw_height//2 * cos_a),
                (u - jaw_width//2 * cos_a + jaw_height//2 * sin_a,
                 v - jaw_width//2 * sin_a - jaw_height//2 * cos_a)
            ]
            
            # Draw gripper rectangle
            pts = np.array([(int(x), int(y)) for x, y in corners], np.int32)
            cv2.polylines(image, [pts], True, color, 2)
            
            # Add quality score text
            quality_text = f"{grasp['quality_score']:.2f}"
            cv2.putText(image, quality_text, (int(u + 10), int(v - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def main():
    """Test the GraspNet predictor"""
    predictor = GraspNetPredictor()
    
    # Load test images
    rgb_path = "../src/img.PNG"
    depth_path = "../src/img-d.PNG"
    
    if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
        print("Test images not found!")
        return
    
    rgb_image = cv2.imread(rgb_path)
    depth_image = cv2.imread(depth_path, -1)
    
    # Convert depth if needed
    if len(depth_image.shape) == 3:
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    
    print(f"RGB shape: {rgb_image.shape}, Depth shape: {depth_image.shape}")
    
    # Create sample detections
    sample_detections = [
        {
            "label": "bottle",
            "confidence": 0.85,
            "bbox": [200, 100, 80, 150]
        },
        {
            "label": "laptop",
            "confidence": 0.90, 
            "bbox": [300, 50, 200, 120]
        }
    ]
    
    # Predict grasps
    grasps = predictor.predict_grasps(rgb_image, depth_image, sample_detections)
    
    print(f"Generated {len(grasps)} grasps:")
    for grasp in grasps:
        print(f"  - {grasp['object_label']}: quality={grasp['quality_score']:.2f}")
    
    # Visualize grasps
    vis_image = predictor.visualize_grasps(rgb_image, grasps, "../outputs/grasps_6d.jpg")
    
    print("✅ GraspNet prediction test completed!")


if __name__ == "__main__":
    import os
    main()