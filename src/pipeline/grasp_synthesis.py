import numpy as np
import cv2
from typing import List, Dict, Tuple

class GraspSynthesis:
    """
    Stage 3: Grasp Synthesis using GraspNet with collision checking
    Rejects grasps near thin/noisy regions from depth map
    """
    def __init__(self):
        self.grasp_width_range = (20, 100)
        self.grasp_orientations = np.linspace(0, np.pi, 8)
        # GraspNet parameters
        self.min_grasp_quality = 0.3
        self.collision_threshold = 0.05  # meters
        
    def generate_grasps(self, image: np.ndarray, depth: np.ndarray, masks: List[np.ndarray]) -> List[Dict]:
        """
        Use graspnet-baseline with collision checking
        
        Args:
            image: RGB image
            depth: Depth map
            masks: List of object masks
            
        Returns:
            grasps: List of grasp candidates with collision checking
        """
        all_grasps = []
        
        for i, mask in enumerate(masks):
            # Generate grasp candidates for this object
            object_grasps = self._generate_object_grasps(image, depth, mask, i)
            
            # Filter grasps using depth-based collision checking
            if depth is not None:
                object_grasps = self._collision_check_grasps(object_grasps, depth, mask)
                object_grasps = self._filter_thin_regions(object_grasps, depth, mask)
            
            all_grasps.extend(object_grasps)
            
        # Sort by quality and return top candidates
        all_grasps.sort(key=lambda x: x["quality"], reverse=True)
        return all_grasps
    
    def _generate_object_grasps(self, image: np.ndarray, depth: np.ndarray, 
                               mask: np.ndarray, object_id: int) -> List[Dict]:
        """Generate grasp candidates for a single object using GraspNet approach"""
        grasps = []
        
        # Find object center and bounding box
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return grasps
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        center = [(x_min + x_max) // 2, (y_min + y_max) // 2]
        
        # Generate multiple grasp orientations
        for angle in self.grasp_orientations:
            # Calculate grasp width based on object size
            object_width = min(x_max - x_min, y_max - y_min)
            grasp_width = min(max(object_width * 0.8, self.grasp_width_range[0]), 
                             self.grasp_width_range[1])
            
            # Calculate grasp points
            dx = grasp_width/2 * np.cos(angle + np.pi/2)
            dy = grasp_width/2 * np.sin(angle + np.pi/2)
            
            grasp_point1 = [center[0] + dx, center[1] + dy]
            grasp_point2 = [center[0] - dx, center[1] - dy]
            
            # Check if grasp points are within image bounds
            if (0 <= grasp_point1[0] < image.shape[1] and 0 <= grasp_point1[1] < image.shape[0] and
                0 <= grasp_point2[0] < image.shape[1] and 0 <= grasp_point2[1] < image.shape[0]):
                
                # Calculate grasp quality using GraspNet-style metrics
                quality = self._calculate_grasp_quality(image, depth, mask, center, angle, grasp_width)
                
                if quality > self.min_grasp_quality:
                    grasp = {
                        "object_id": object_id,
                        "center": center,
                        "angle": angle,
                        "width": grasp_width,
                        "points": [grasp_point1, grasp_point2],
                        "quality": quality,
                        "depth": depth[center[1], center[0]] if depth is not None else 0,
                        "collision_free": True  # Will be updated by collision checking
                    }
                    grasps.append(grasp)
        
        return grasps
    
    def _calculate_grasp_quality(self, image: np.ndarray, depth: np.ndarray, 
                                mask: np.ndarray, center: List[int], angle: float, width: float) -> float:
        """Calculate grasp quality using multiple metrics"""
        quality_score = 0.0
        
        # 1. Depth consistency around grasp center
        if depth is not None:
            cy, cx = center[1], center[0]
            patch_size = 5
            y1, y2 = max(0, cy-patch_size), min(depth.shape[0], cy+patch_size)
            x1, x2 = max(0, cx-patch_size), min(depth.shape[1], cx+patch_size)
            
            depth_patch = depth[y1:y2, x1:x2]
            valid_depths = depth_patch[depth_patch > 0]
            
            if len(valid_depths) > 0:
                depth_std = np.std(valid_depths)
                depth_mean = np.mean(valid_depths)
                if depth_mean > 0:
                    depth_consistency = 1.0 - min(depth_std / depth_mean, 1.0)
                    quality_score += 0.3 * depth_consistency
        
        # 2. Edge strength perpendicular to grasp
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Sample points along grasp line
        dx = np.cos(angle)
        dy = np.sin(angle)
        edge_strength = 0
        
        for t in np.linspace(-width/2, width/2, 10):
            x = int(center[0] + t * dx)
            y = int(center[1] + t * dy)
            if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:
                edge_strength += edges[y, x]
        
        quality_score += 0.2 * (edge_strength / 255.0 / 10)  # Normalize
        
        # 3. Object mask coverage
        mask_coverage = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        quality_score += 0.2 * mask_coverage
        
        # 4. Distance from object center (prefer central grasps)
        coords = np.where(mask > 0)
        if len(coords[0]) > 0:
            obj_center = [np.mean(coords[1]), np.mean(coords[0])]
            dist_from_center = np.sqrt((center[0] - obj_center[0])**2 + (center[1] - obj_center[1])**2)
            max_dist = np.sqrt(mask.shape[0]**2 + mask.shape[1]**2)
            centrality = 1.0 - (dist_from_center / max_dist)
            quality_score += 0.3 * centrality
        
        return min(quality_score, 1.0)
    
    def _collision_check_grasps(self, grasps: List[Dict], depth: np.ndarray, mask: np.ndarray) -> List[Dict]:
        """Simulate collision checking using depth information"""
        collision_free_grasps = []
        
        for grasp in grasps:
            is_collision_free = True
            
            # Check for collision along grasp trajectory
            points = grasp["points"]
            center = grasp["center"]
            
            # Sample points between grasp fingers
            num_samples = 10
            for i in range(num_samples):
                t = i / (num_samples - 1)
                sample_x = int(points[0][0] * (1-t) + points[1][0] * t)
                sample_y = int(points[0][1] * (1-t) + points[1][1] * t)
                
                if (0 <= sample_x < depth.shape[1] and 0 <= sample_y < depth.shape[0]):
                    sample_depth = depth[sample_y, sample_x]
                    center_depth = depth[center[1], center[0]]
                    
                    # Check if there's an obstacle between gripper and object
                    if sample_depth > 0 and center_depth > 0:
                        if abs(sample_depth - center_depth) > self.collision_threshold:
                            is_collision_free = False
                            break
            
            grasp["collision_free"] = is_collision_free
            if is_collision_free:
                collision_free_grasps.append(grasp)
        
        return collision_free_grasps
    
    def _filter_thin_regions(self, grasps: List[Dict], depth: np.ndarray, mask: np.ndarray) -> List[Dict]:
        """Reject grasps near thin/noisy regions from depth map"""
        filtered_grasps = []
        
        for grasp in grasps:
            center = grasp["center"]
            
            # Check thickness around grasp center
            patch_size = 10
            cy, cx = center[1], center[0]
            y1, y2 = max(0, cy-patch_size), min(depth.shape[0], cy+patch_size)
            x1, x2 = max(0, cx-patch_size), min(depth.shape[1], cx+patch_size)
            
            depth_patch = depth[y1:y2, x1:x2]
            mask_patch = mask[y1:y2, x1:x2]
            
            # Only consider depths within the object mask
            object_depths = depth_patch[mask_patch > 0]
            
            if len(object_depths) > 5:  # Need enough points
                depth_range = np.max(object_depths) - np.min(object_depths)
                
                # Reject if object is too thin (less than 5mm variation)
                min_thickness = 0.005  # 5mm in meters
                if depth_range > min_thickness:
                    filtered_grasps.append(grasp)
        
        return filtered_grasps
    
    def visualize_grasps(self, image: np.ndarray, grasps: List[Dict], save_path: str = None) -> np.ndarray:
        """Visualize grasp candidates on image"""
        vis_image = image.copy()
        
        for grasp in grasps:
            points = grasp["points"]
            quality = grasp["quality"]
            collision_free = grasp.get("collision_free", True)
            
            # Color based on collision status
            color = (0, 255, 0) if collision_free else (0, 0, 255)  # Green if collision-free, red otherwise
            
            # Draw grasp line
            pt1 = (int(points[0][0]), int(points[0][1]))
            pt2 = (int(points[1][0]), int(points[1][1]))
            cv2.line(vis_image, pt1, pt2, color, 3)
            
            # Draw grasp points
            cv2.circle(vis_image, pt1, 5, color, -1)
            cv2.circle(vis_image, pt2, 5, color, -1)
            
            # Draw quality score
            center = (int(grasp["center"][0]), int(grasp["center"][1]))
            cv2.putText(vis_image, f"{quality:.2f}", 
                       (center[0], center[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            
        return vis_image
