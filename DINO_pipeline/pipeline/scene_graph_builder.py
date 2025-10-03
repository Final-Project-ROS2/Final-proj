"""
DINO Pipeline - Scene Graph Construction Module
Builds structured scene representations with objects, relations, and affordances
"""

import cv2
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any
import json
from collections import defaultdict

class SceneGraphBuilder:
    """
    Constructs scene graphs from detected objects, semantic tags, and grasps
    Creates structured representations for robotic task planning
    """
    
    def __init__(self, image_width: int = 640, image_height: int = 480):
        self.image_width = image_width
        self.image_height = image_height
        
        # Spatial relation thresholds
        self.proximity_threshold = 0.3  # 30cm in 3D space
        self.overlap_threshold = 0.1   # 10% overlap for "touching"
        self.above_below_threshold = 0.05  # 5cm vertical difference
        
        # Relation types
        self.spatial_relations = [
            "left_of", "right_of", "above", "below", "near", "far_from",
            "in_front_of", "behind", "touching", "overlapping", "contains",
            "supported_by", "supports"
        ]
        
        print("✅ Scene Graph Builder initialized")
    
    def build_scene_graph(self, rgb_image: np.ndarray, depth_image: np.ndarray,
                         detections: List[Dict], grasps: List[Dict]) -> Dict[str, Any]:
        """
        Build complete scene graph from all pipeline components
        
        Args:
            rgb_image: RGB image
            depth_image: Depth image  
            detections: Enhanced detections with semantic tags
            grasps: 6D grasp predictions
            
        Returns:
            Complete scene graph dictionary
        """
        # Extract scene nodes (objects)
        nodes = self._create_scene_nodes(detections, grasps)
        
        # Extract spatial relations (edges)
        edges = self._extract_spatial_relations(nodes, rgb_image.shape)
        
        # Analyze scene properties
        scene_properties = self._analyze_scene_properties(nodes, edges)
        
        # Group objects by affordances and semantic categories
        semantic_groups = self._group_by_semantics(nodes)
        
        # Generate task-relevant insights
        task_insights = self._generate_task_insights(nodes, edges, semantic_groups)
        
        # Create complete scene graph
        scene_graph = {
            "metadata": {
                "timestamp": self._get_timestamp(),
                "image_dimensions": {"width": rgb_image.shape[1], "height": rgb_image.shape[0]},
                "num_objects": len(nodes),
                "num_relations": len(edges),
                "num_grasps": len(grasps)
            },
            "nodes": nodes,
            "edges": edges,
            "scene_properties": scene_properties,
            "semantic_groups": semantic_groups,
            "task_insights": task_insights,
            "grasp_summary": self._summarize_grasps(grasps)
        }
        
        return scene_graph
    
    def _create_scene_nodes(self, detections: List[Dict], grasps: List[Dict]) -> List[Dict]:
        """Create scene graph nodes from detected objects"""
        nodes = []
        
        # Group grasps by object
        object_grasps = defaultdict(list)
        for grasp in grasps:
            object_grasps[grasp["object_label"]].append(grasp)
        
        for i, detection in enumerate(detections):
            # Calculate 3D properties
            bbox_3d = self._calculate_3d_bbox(detection)
            
            # Get associated grasps
            obj_grasps = object_grasps.get(detection["label"], [])
            best_grasp = max(obj_grasps, key=lambda g: g["quality_score"]) if obj_grasps else None
            
            node = {
                "id": f"obj_{i}",
                "label": detection["label"],
                "semantic_category": detection.get("semantic_category", "unknown"),
                "confidence": detection["confidence"],
                "semantic_confidence": detection.get("semantic_confidence", 0.0),
                
                # Spatial properties
                "bbox_2d": detection["bbox"],  # [x, y, w, h]
                "bbox_3d": bbox_3d,
                "center_2d": [detection["bbox"][0] + detection["bbox"][2]//2,
                             detection["bbox"][1] + detection["bbox"][3]//2],
                "center_3d": bbox_3d["center"] if bbox_3d else None,
                
                # Semantic properties
                "affordances": detection.get("affordances", []),
                "description": detection.get("description", f"a {detection['label']}"),
                
                # Grasp properties
                "num_grasps": len(obj_grasps),
                "best_grasp_quality": best_grasp["quality_score"] if best_grasp else 0.0,
                "graspable": len(obj_grasps) > 0,
                "best_grasp": best_grasp,
                
                # Physical properties (estimated)
                "estimated_size": self._estimate_object_size(detection),
                "estimated_weight": self._estimate_object_weight(detection),
                "fragility": self._estimate_fragility(detection)
            }
            
            nodes.append(node)
        
        return nodes
    
    def _calculate_3d_bbox(self, detection: Dict) -> Optional[Dict]:
        """Calculate 3D bounding box from 2D detection (simplified)"""
        # This is a simplified version - real implementation would use depth data
        x, y, w, h = detection["bbox"]
        
        # Estimate depth (placeholder - would use actual depth data)
        estimated_depth = 1.0  # 1 meter default
        
        # Simple 3D bbox estimation
        bbox_3d = {
            "center": [x + w/2, y + h/2, estimated_depth],
            "dimensions": [w * 0.001, h * 0.001, 0.1],  # Convert to meters, assume 10cm depth
            "min_bounds": [x, y, estimated_depth - 0.05],
            "max_bounds": [x + w, y + h, estimated_depth + 0.05]
        }
        
        return bbox_3d
    
    def _extract_spatial_relations(self, nodes: List[Dict], image_shape: Tuple) -> List[Dict]:
        """Extract spatial relations between objects"""
        edges = []
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i >= j:  # Avoid duplicate pairs and self-relations
                    continue
                
                relations = self._compute_pairwise_relations(node1, node2, image_shape)
                
                for relation in relations:
                    edge = {
                        "id": f"rel_{len(edges)}",
                        "subject": {"id": node1["id"], "label": node1["label"]},
                        "object": {"id": node2["id"], "label": node2["label"]},
                        "relation": relation["type"],
                        "confidence": relation["confidence"],
                        "description": f"{node1['label']} {relation['type']} {node2['label']}"
                    }
                    edges.append(edge)
        
        return edges
    
    def _compute_pairwise_relations(self, node1: Dict, node2: Dict, 
                                  image_shape: Tuple) -> List[Dict]:
        """Compute spatial relations between two objects"""
        relations = []
        
        # Get 2D centers
        center1 = node1["center_2d"]
        center2 = node2["center_2d"]
        
        # Get bounding boxes
        bbox1 = node1["bbox_2d"]
        bbox2 = node2["bbox_2d"]
        
        # Calculate distances
        distance_2d = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Normalize distance by image diagonal
        image_diagonal = math.sqrt(image_shape[0]**2 + image_shape[1]**2)
        normalized_distance = distance_2d / image_diagonal
        
        # Left/Right relations
        if center1[0] < center2[0] - 20:  # 20 pixel threshold
            relations.append({"type": "left_of", "confidence": 0.8})
        elif center1[0] > center2[0] + 20:
            relations.append({"type": "right_of", "confidence": 0.8})
        
        # Above/Below relations
        if center1[1] < center2[1] - 20:
            relations.append({"type": "above", "confidence": 0.8})
        elif center1[1] > center2[1] + 20:
            relations.append({"type": "below", "confidence": 0.8})
        
        # Proximity relations
        if normalized_distance < 0.2:
            relations.append({"type": "near", "confidence": 0.9})
        elif normalized_distance > 0.5:
            relations.append({"type": "far_from", "confidence": 0.7})
        
        # Overlap/touching relations
        overlap = self._calculate_bbox_overlap(bbox1, bbox2)
        if overlap > 0.1:
            relations.append({"type": "overlapping", "confidence": min(overlap * 2, 1.0)})
        elif overlap > 0.05:
            relations.append({"type": "touching", "confidence": 0.6})
        
        return relations
    
    def _calculate_bbox_overlap(self, bbox1: List, bbox2: List) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            intersection = (right - left) * (bottom - top)
            union = w1 * h1 + w2 * h2 - intersection
            return intersection / union if union > 0 else 0
        
        return 0
    
    def _analyze_scene_properties(self, nodes: List[Dict], edges: List[Dict]) -> Dict:
        """Analyze overall scene properties"""
        # Object statistics
        object_counts = defaultdict(int)
        category_counts = defaultdict(int)
        affordance_counts = defaultdict(int)
        
        total_grasp_quality = 0
        graspable_objects = 0
        
        for node in nodes:
            object_counts[node["label"]] += 1
            category_counts[node["semantic_category"]] += 1
            
            for affordance in node["affordances"]:
                affordance_counts[affordance] += 1
            
            if node["graspable"]:
                graspable_objects += 1
                total_grasp_quality += node["best_grasp_quality"]
        
        # Relation statistics
        relation_counts = defaultdict(int)
        for edge in edges:
            relation_counts[edge["relation"]] += 1
        
        # Scene complexity metrics
        object_density = len(nodes) / (self.image_width * self.image_height / 10000)  # objects per 100x100 area
        relation_density = len(edges) / max(len(nodes), 1)
        
        return {
            "object_statistics": {
                "total_objects": len(nodes),
                "unique_labels": len(object_counts),
                "object_counts": dict(object_counts),
                "category_counts": dict(category_counts),
                "affordance_counts": dict(affordance_counts)
            },
            "grasp_statistics": {
                "graspable_objects": graspable_objects,
                "avg_grasp_quality": total_grasp_quality / max(graspable_objects, 1),
                "grasp_success_rate": graspable_objects / len(nodes) if nodes else 0
            },
            "relation_statistics": {
                "total_relations": len(edges),
                "relation_counts": dict(relation_counts),
                "relation_density": relation_density
            },
            "scene_complexity": {
                "object_density": object_density,
                "relation_density": relation_density,
                "complexity_score": (object_density + relation_density) / 2
            }
        }
    
    def _group_by_semantics(self, nodes: List[Dict]) -> Dict:
        """Group objects by semantic properties"""
        groups = {
            "by_category": defaultdict(list),
            "by_affordance": defaultdict(list),
            "by_graspability": {"graspable": [], "not_graspable": []},
            "by_size": {"small": [], "medium": [], "large": []}
        }
        
        for node in nodes:
            # Group by category
            groups["by_category"][node["semantic_category"]].append(node["id"])
            
            # Group by affordances
            for affordance in node["affordances"]:
                groups["by_affordance"][affordance].append(node["id"])
            
            # Group by graspability
            if node["graspable"]:
                groups["by_graspability"]["graspable"].append(node["id"])
            else:
                groups["by_graspability"]["not_graspable"].append(node["id"])
            
            # Group by estimated size
            size = node["estimated_size"]
            if size < 0.3:
                groups["by_size"]["small"].append(node["id"])
            elif size < 0.7:
                groups["by_size"]["medium"].append(node["id"])
            else:
                groups["by_size"]["large"].append(node["id"])
        
        # Convert defaultdicts to regular dicts
        return {
            "by_category": dict(groups["by_category"]),
            "by_affordance": dict(groups["by_affordance"]),
            "by_graspability": groups["by_graspability"],
            "by_size": groups["by_size"]
        }
    
    def _generate_task_insights(self, nodes: List[Dict], edges: List[Dict], 
                               semantic_groups: Dict) -> Dict:
        """Generate insights for robotic task planning"""
        insights = {
            "manipulation_candidates": [],
            "spatial_constraints": [],
            "task_recommendations": [],
            "potential_actions": []
        }
        
        # Find best manipulation candidates
        graspable_nodes = [n for n in nodes if n["graspable"]]
        graspable_nodes.sort(key=lambda n: n["best_grasp_quality"], reverse=True)
        
        for node in graspable_nodes[:3]:  # Top 3 candidates
            insights["manipulation_candidates"].append({
                "object_id": node["id"],
                "label": node["label"],
                "grasp_quality": node["best_grasp_quality"],
                "reason": f"High grasp quality ({node['best_grasp_quality']:.2f}) and good accessibility"
            })
        
        # Extract spatial constraints
        for edge in edges:
            if edge["relation"] in ["touching", "overlapping", "supports"]:
                insights["spatial_constraints"].append({
                    "constraint_type": edge["relation"],
                    "description": edge["description"],
                    "impact": "Movement of one object affects the other"
                })
        
        # Generate task recommendations
        if "container" in semantic_groups["by_category"]:
            insights["task_recommendations"].append({
                "task": "pouring",
                "objects": semantic_groups["by_category"]["container"],
                "feasibility": "high" if len(semantic_groups["by_category"]["container"]) >= 2 else "medium"
            })
        
        if "tool" in semantic_groups["by_category"]:
            insights["task_recommendations"].append({
                "task": "tool_use",
                "objects": semantic_groups["by_category"]["tool"], 
                "feasibility": "high"
            })
        
        # Generate potential actions
        for node in graspable_nodes:
            actions = ["pick", "move"]
            
            if "pourable" in node["affordances"]:
                actions.append("pour")
            if "openable" in node["affordances"]:
                actions.append("open")
            if "usable" in node["affordances"]:
                actions.append("use")
            
            insights["potential_actions"].append({
                "object_id": node["id"],
                "label": node["label"],
                "actions": actions
            })
        
        return insights
    
    def _estimate_object_size(self, detection: Dict) -> float:
        """Estimate object size (normalized 0-1)"""
        bbox = detection["bbox"]
        area = bbox[2] * bbox[3]
        # Normalize by image area (simplified)
        normalized_area = area / (self.image_width * self.image_height)
        return min(normalized_area * 10, 1.0)  # Scale up and cap at 1.0
    
    def _estimate_object_weight(self, detection: Dict) -> str:
        """Estimate object weight category"""
        label = detection["label"].lower()
        
        # Simple heuristic based on object type
        heavy_objects = ["laptop", "monitor", "tv", "pot", "pan", "book"]
        light_objects = ["cup", "mouse", "pen", "remote", "phone"]
        
        if any(obj in label for obj in heavy_objects):
            return "heavy"
        elif any(obj in label for obj in light_objects):
            return "light"
        else:
            return "medium"
    
    def _estimate_fragility(self, detection: Dict) -> str:
        """Estimate object fragility"""
        label = detection["label"].lower()
        
        fragile_objects = ["glass", "cup", "mug", "vase", "monitor", "laptop"]
        robust_objects = ["hammer", "wrench", "book", "remote"]
        
        if any(obj in label for obj in fragile_objects):
            return "fragile"
        elif any(obj in label for obj in robust_objects):
            return "robust"
        else:
            return "moderate"
    
    def _summarize_grasps(self, grasps: List[Dict]) -> Dict:
        """Create summary of grasp predictions"""
        if not grasps:
            return {"total_grasps": 0}
        
        quality_scores = [g["quality_score"] for g in grasps]
        
        return {
            "total_grasps": len(grasps),
            "avg_quality": np.mean(quality_scores),
            "max_quality": max(quality_scores),
            "min_quality": min(quality_scores),
            "high_quality_grasps": len([g for g in grasps if g["quality_score"] > 0.7]),
            "objects_with_grasps": len(set(g["object_label"] for g in grasps))
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def visualize_scene_graph(self, rgb_image: np.ndarray, scene_graph: Dict,
                             save_path: str = None) -> np.ndarray:
        """
        Visualize scene graph on image
        
        Args:
            rgb_image: Input RGB image
            scene_graph: Complete scene graph
            save_path: Optional path to save visualization
            
        Returns:
            Annotated image with scene graph visualization
        """
        vis_image = rgb_image.copy()
        
        # Draw nodes (objects)
        for node in scene_graph["nodes"]:
            x, y, w, h = node["bbox_2d"]
            x, y, w, h = int(x), int(y), int(w), int(h)  # Ensure integer coordinates
            center = node["center_2d"]
            center = (int(center[0]), int(center[1]))  # Ensure integer center
            
            # Color based on graspability
            color = (0, 255, 0) if node["graspable"] else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw node info
            info_text = f"{node['label']} ({node['semantic_category']})"
            cv2.putText(vis_image, info_text, (int(x), int(y - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw center point
            cv2.circle(vis_image, center, 5, color, -1)
        
        # Draw edges (relations)
        for edge in scene_graph["edges"]:
            # Find subject and object nodes
            subject_node = next(n for n in scene_graph["nodes"] if n["id"] == edge["subject"]["id"])
            object_node = next(n for n in scene_graph["nodes"] if n["id"] == edge["object"]["id"])
            
            # Draw relation line
            start = (int(subject_node["center_2d"][0]), int(subject_node["center_2d"][1]))
            end = (int(object_node["center_2d"][0]), int(object_node["center_2d"][1]))
            
            cv2.line(vis_image, start, end, (255, 0, 255), 1)
            
            # Draw relation label at midpoint
            mid_x = (start[0] + end[0]) // 2
            mid_y = (start[1] + end[1]) // 2
            
            cv2.putText(vis_image, edge["relation"], (int(mid_x), int(mid_y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
        
        # Add scene statistics
        stats = scene_graph["scene_properties"]
        stats_text = [
            f"Objects: {stats['object_statistics']['total_objects']}",
            f"Relations: {stats['relation_statistics']['total_relations']}",
            f"Graspable: {stats['grasp_statistics']['graspable_objects']}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(vis_image, text, (10, 30 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"✅ Scene graph visualization saved to: {save_path}")
        
        return vis_image
    
    def save_scene_graph(self, scene_graph: Dict, save_path: str):
        """Save scene graph to JSON file"""
        with open(save_path, 'w') as f:
            json.dump(scene_graph, f, indent=2, default=str)
        print(f"✅ Scene graph saved to: {save_path}")


def main():
    """Test the scene graph builder"""
    builder = SceneGraphBuilder()
    
    # Create sample data for testing
    sample_detections = [
        {
            "label": "bottle",
            "confidence": 0.85,
            "bbox": [100, 50, 80, 150],
            "semantic_category": "container",
            "affordances": ["graspable", "pourable", "moveable"],
            "description": "cylindrical container for liquids",
            "semantic_confidence": 0.9
        },
        {
            "label": "laptop",
            "confidence": 0.90,
            "bbox": [300, 100, 200, 120],
            "semantic_category": "electronics",
            "affordances": ["openable", "usable", "moveable"],
            "description": "portable computer device",
            "semantic_confidence": 0.95
        }
    ]
    
    sample_grasps = [
        {
            "object_label": "bottle",
            "quality_score": 0.85,
            "grasp_center": (0.2, 0.3, 1.0)
        },
        {
            "object_label": "laptop",
            "quality_score": 0.70,
            "grasp_center": (0.4, 0.4, 1.0)
        }
    ]
    
    # Create dummy images
    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_image = np.ones((480, 640), dtype=np.float32)
    
    # Build scene graph
    scene_graph = builder.build_scene_graph(rgb_image, depth_image, 
                                           sample_detections, sample_grasps)
    
    print("Scene Graph Summary:")
    print(f"  Objects: {scene_graph['metadata']['num_objects']}")
    print(f"  Relations: {scene_graph['metadata']['num_relations']}")
    print(f"  Grasps: {scene_graph['metadata']['num_grasps']}")
    
    # Save scene graph
    builder.save_scene_graph(scene_graph, "../outputs/scene_graph.json")
    
    print("✅ Scene graph construction test completed!")


if __name__ == "__main__":
    import os
    main()