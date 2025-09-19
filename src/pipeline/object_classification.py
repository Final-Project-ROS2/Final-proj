import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple

class ObjectClassification:
    """
    Stage 2: Object Classification using CLIP embeddings
    Compare to curated tool embeddings for few-shot classification
    """
    def __init__(self, model_path: str = None):
        # Tool classes for robotic manipulation
        self.tool_classes = [
            "hammer", "screwdriver", "pliers", "wrench", "drill", 
            "saw", "chisel", "file", "clamp", "measuring_tape",
            "scissors", "knife", "allen_key", "socket", "unknown"
        ]
        
        # Curated tool embeddings (placeholder - would be pre-computed CLIP embeddings)
        self.tool_embeddings = self._load_tool_embeddings()
        
        # Initialize CLIP model (placeholder)
        self.clip_model = None  # Would load actual CLIP model here
        
    def _load_tool_embeddings(self) -> Dict[str, np.ndarray]:
        """Load pre-computed CLIP embeddings for tool classes"""
        # Placeholder - would load actual embeddings from file
        embeddings = {}
        for tool in self.tool_classes:
            # Random embeddings for demonstration
            embeddings[tool] = np.random.randn(512)  # CLIP embedding dimension
            embeddings[tool] = embeddings[tool] / np.linalg.norm(embeddings[tool])
        return embeddings
        
    def classify_objects(self, image: np.ndarray, masks: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        """
        CLIP embedding comparison for tool classification
        
        Args:
            image: RGB image
            masks: List of binary masks
            
        Returns:
            labels: List of predicted class labels
            confidences: List of confidence scores
        """
        labels = []
        confidences = []
        
        for mask in masks:
            # Extract object region using mask
            object_crop = self._extract_object_crop(image, mask)
            
            # Get CLIP embedding for the cropped object
            object_embedding = self._get_clip_embedding(object_crop)
            
            # Compare with tool embeddings
            best_match, confidence = self._find_best_match(object_embedding)
            
            labels.append(best_match)
            confidences.append(confidence)
            
        return labels, confidences
    
    def _extract_object_crop(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract object region using mask"""
        # Find bounding box of mask
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return image  # Return full image if mask is empty
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Add padding
        padding = 10
        y_min = max(0, y_min - padding)
        y_max = min(image.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        
        # Crop the object region
        crop = image[y_min:y_max, x_min:x_max]
        
        # Apply mask to focus on object
        mask_crop = mask[y_min:y_max, x_min:x_max]
        crop[mask_crop == 0] = [255, 255, 255]  # White background
        
        return crop
    
    def _get_clip_embedding(self, image_crop: np.ndarray) -> np.ndarray:
        """Get CLIP embedding for image crop (placeholder)"""
        # Placeholder - would use actual CLIP model
        # For now, return random normalized embedding
        embedding = np.random.randn(512)
        return embedding / np.linalg.norm(embedding)
    
    def _find_best_match(self, object_embedding: np.ndarray) -> Tuple[str, float]:
        """Find best matching tool class using cosine similarity"""
        best_similarity = -1
        best_class = "unknown"
        
        for tool_class, tool_embedding in self.tool_embeddings.items():
            # Cosine similarity
            similarity = np.dot(object_embedding, tool_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_class = tool_class
        
        # Convert similarity to confidence (0-1 range)
        confidence = (best_similarity + 1) / 2
        
        return best_class, confidence
