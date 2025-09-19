import numpy as np
from typing import List, Dict, Tuple

class SceneUnderstandingVLM:
    """
    Stage 4: Scene Understanding using Vision Language Model
    Extract object-object spatial relations and generate human-readable summaries
    """
    def __init__(self, model_name: str = "placeholder"):
        self.model_name = model_name
        # Spatial relation templates
        self.spatial_relations = [
            "left_of", "right_of", "above", "below", "near", "far_from", 
            "in_front_of", "behind", "touching", "overlapping"
        ]
        
    def understand_scene(self, image: np.ndarray, labels: List[str], boxes: List[List[float]]) -> Dict:
        """
        Extract scene graph / relations rather than full text description
        
        Args:
            image: RGB image
            labels: List of object class labels
            boxes: List of bounding boxes [x, y, w, h]
            
        Returns:
            scene_description: Structured scene understanding with spatial relations
        """
        # Extract object-object spatial relations
        spatial_graph = self._extract_spatial_relations(labels, boxes)
        
        # Generate human-readable scene summary for research paper
        scene_summary = self._generate_research_summary(labels, boxes, spatial_graph)
        
        # Count objects by category
        object_counts = self._count_objects_by_category(labels)
        
        # Analyze scene complexity
        scene_complexity = self._analyze_scene_complexity(labels, boxes, spatial_graph)
        
        scene_understanding = {
            "spatial_graph": spatial_graph,
            "scene_summary": scene_summary,
            "object_counts": object_counts,
            "scene_complexity": scene_complexity,
            "total_objects": len(labels),
            "unique_classes": len(set(labels))
        }
        
        return scene_understanding
    
    def _extract_spatial_relations(self, labels: List[str], boxes: List[List[float]]) -> List[Dict]:
        """Extract structured spatial relations between objects"""
        relations = []
        
        for i, (label1, box1) in enumerate(zip(labels, boxes)):
            for j, (label2, box2) in enumerate(zip(labels, boxes)):
                if i != j:  # Don't compare object with itself
                    relation = self._determine_spatial_relation(box1, box2)
                    if relation:
                        relations.append({
                            "subject": {"id": i, "class": label1, "bbox": box1},
                            "object": {"id": j, "class": label2, "bbox": box2},
                            "relation": relation,
                            "confidence": self._calculate_relation_confidence(box1, box2, relation)
                        })
        
        # Filter out low-confidence relations
        high_confidence_relations = [r for r in relations if r["confidence"] > 0.7]
        
        return high_confidence_relations
    
    def _determine_spatial_relation(self, box1: List[float], box2: List[float]) -> str:
        """Determine spatial relation between two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate centers
        center1 = [x1 + w1/2, y1 + h1/2]
        center2 = [x2 + w2/2, y2 + h2/2]
        
        # Calculate distances
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Determine primary relation based on largest displacement
        if abs(dx) > abs(dy):
            if dx > 0:
                primary_relation = "right_of"
            else:
                primary_relation = "left_of"
        else:
            if dy > 0:
                primary_relation = "below"
            else:
                primary_relation = "above"
        
        # Check for proximity
        avg_size = (w1 + h1 + w2 + h2) / 4
        if distance < avg_size * 0.5:
            return "near"
        elif distance > avg_size * 2:
            return "far_from"
        else:
            return primary_relation
    
    def _calculate_relation_confidence(self, box1: List[float], box2: List[float], relation: str) -> float:
        """Calculate confidence score for a spatial relation"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        center1 = [x1 + w1/2, y1 + h1/2]
        center2 = [x2 + w2/2, y2 + h2/2]
        
        dx = abs(center2[0] - center1[0])
        dy = abs(center2[1] - center1[1])
        
        # Higher confidence for clearer spatial separations
        if relation in ["left_of", "right_of"]:
            confidence = min(dx / max(w1, w2), 1.0)
        elif relation in ["above", "below"]:
            confidence = min(dy / max(h1, h2), 1.0)
        elif relation == "near":
            avg_size = (w1 + h1 + w2 + h2) / 4
            distance = np.sqrt(dx**2 + dy**2)
            confidence = max(0, 1.0 - distance / avg_size)
        else:
            confidence = 0.5  # Default confidence
        
        return min(confidence, 1.0)
    
    def _generate_research_summary(self, labels: List[str], boxes: List[List[float]], 
                                 spatial_graph: List[Dict]) -> str:
        """Generate human-readable summary for research paper"""
        if not labels:
            return "Empty scene with no detected objects."
        
        # Count object types
        object_counts = self._count_objects_by_category(labels)
        
        # Create summary
        summary_parts = []
        summary_parts.append(f"Scene contains {len(labels)} objects:")
        
        for obj_class, count in object_counts.items():
            if count > 1:
                summary_parts.append(f"{count} {obj_class}s")
            else:
                summary_parts.append(f"{count} {obj_class}")
        
        # Add spatial relations summary
        if spatial_graph:
            relation_types = [r["relation"] for r in spatial_graph]
            unique_relations = set(relation_types)
            summary_parts.append(f"Spatial relations include: {', '.join(unique_relations)}")
        
        return ". ".join(summary_parts) + "."
    
    def _count_objects_by_category(self, labels: List[str]) -> Dict[str, int]:
        """Count objects by category"""
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return counts
    
    def _analyze_scene_complexity(self, labels: List[str], boxes: List[List[float]], 
                                spatial_graph: List[Dict]) -> Dict:
        """Analyze scene complexity metrics"""
        # Object density
        if boxes:
            total_area = sum(box[2] * box[3] for box in boxes)  # Sum of w*h
            # Assuming image size (this should be passed as parameter in real implementation)
            image_area = 640 * 480  # Default assumption
            object_density = total_area / image_area
        else:
            object_density = 0
        
        # Spatial complexity (number of relations per object)
        if labels:
            spatial_complexity = len(spatial_graph) / len(labels)
        else:
            spatial_complexity = 0
        
        # Class diversity (ratio of unique classes to total objects)
        if labels:
            class_diversity = len(set(labels)) / len(labels)
        else:
            class_diversity = 0
        
        return {
            "object_density": object_density,
            "spatial_complexity": spatial_complexity,
            "class_diversity": class_diversity,
            "total_relations": len(spatial_graph)
        }
