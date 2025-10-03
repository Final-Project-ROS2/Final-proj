"""
DINO Pipeline - CLIP Semantic Tagging Module
Enhanced semantic labeling using CLIP embeddings
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import logging

# Try to import CLIP
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

class CLIPSemanticTagger:
    """
    CLIP-based semantic tagging for detected objects
    Provides rich semantic labels and affordance understanding
    """
    
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.processor = None
        
        # Object categories for robotic manipulation
        self.object_categories = [
            # Tools and utensils
            "hammer", "screwdriver", "pliers", "wrench", "drill", "saw", "chisel",
            "scissors", "knife", "spoon", "fork", "spatula",
            
            # Kitchen items
            "bottle", "cup", "mug", "glass", "bowl", "plate", "pot", "pan",
            
            # Office supplies
            "laptop", "mouse", "keyboard", "monitor", "book", "pen", "pencil", "notebook",
            
            # Household items
            "remote", "phone", "clock", "lamp", "vase", "picture frame",
            
            # Containers
            "box", "bag", "basket", "jar", "can", "container",
            
            # Generic
            "object", "item", "tool", "device", "appliance"
        ]
        
        # Affordance categories
        self.affordances = [
            "graspable", "pickable", "moveable", "stackable", "pourable", 
            "cuttable", "holdable", "usable", "fragile", "heavy", "light",
            "openable", "closable", "pressable", "turnable"
        ]
        
        # Initialize CLIP model
        if CLIP_AVAILABLE:
            self._init_clip()
            self.model_type = "clip"
        else:
            self._init_fallback()
            self.model_type = "fallback"
            
        print(f"✅ CLIP Semantic Tagger initialized using: {self.model_type}")
    
    def _init_clip(self):
        """Initialize CLIP model"""
        print("Loading CLIP model...")
        try:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.eval()
            
            # Pre-compute text embeddings for efficiency
            self._precompute_embeddings()
            
        except Exception as e:
            print(f"Failed to load CLIP: {e}")
            self._init_fallback()
            self.model_type = "fallback"
    
    def _init_fallback(self):
        """Initialize fallback semantic tagger"""
        print("⚠️ Using fallback semantic tagger")
        
        # Create mapping from common object names to semantic info
        self.semantic_mapping = {
            "bottle": {
                "category": "container",
                "affordances": ["graspable", "holdable", "pourable", "moveable"],
                "description": "cylindrical container for liquids"
            },
            "cup": {
                "category": "container",
                "affordances": ["graspable", "holdable", "pourable", "fragile"],
                "description": "small drinking vessel"
            },
            "laptop": {
                "category": "electronics",
                "affordances": ["openable", "closable", "usable", "moveable"],
                "description": "portable computer device"
            },
            "mouse": {
                "category": "electronics",
                "affordances": ["graspable", "moveable", "clickable", "usable"],
                "description": "computer pointing device"
            },
            "keyboard": {
                "category": "electronics", 
                "affordances": ["usable", "pressable", "moveable"],
                "description": "computer input device"
            },
            "book": {
                "category": "object",
                "affordances": ["graspable", "openable", "readable", "moveable"],
                "description": "bound collection of pages"
            }
        }
    
    def _precompute_embeddings(self):
        """Pre-compute text embeddings for object categories and affordances"""
        if self.model_type != "clip":
            return
            
        # Prepare text descriptions
        object_texts = [f"a photo of a {obj}" for obj in self.object_categories]
        affordance_texts = [f"an object that is {aff}" for aff in self.affordances]
        
        all_texts = object_texts + affordance_texts
        
        # Compute embeddings
        with torch.no_grad():
            inputs = self.processor(text=all_texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            text_embeddings = self.model.get_text_features(**inputs)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            # Split embeddings
            self.object_embeddings = text_embeddings[:len(object_texts)]
            self.affordance_embeddings = text_embeddings[len(object_texts):]
    
    def tag_objects(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Add semantic tags to detected objects
        
        Args:
            image: RGB image
            detections: List of detection dictionaries from DINO
            
        Returns:
            Enhanced detections with semantic tags
        """
        enhanced_detections = []
        
        for detection in detections:
            # Extract object crop
            object_crop = self._extract_object_crop(image, detection)
            
            # Get semantic information
            semantic_info = self._analyze_object_semantics(object_crop, detection["label"])
            
            # Enhance detection with semantic info
            enhanced_detection = detection.copy()
            enhanced_detection.update({
                "semantic_category": semantic_info["category"],
                "affordances": semantic_info["affordances"],
                "description": semantic_info["description"],
                "semantic_confidence": semantic_info["confidence"],
                "clip_scores": semantic_info.get("clip_scores", {})
            })
            
            enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections
    
    def _extract_object_crop(self, image: np.ndarray, detection: Dict) -> np.ndarray:
        """Extract object region from image"""
        x, y, w, h = detection["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)  # Ensure integer coordinates
        
        # Add some padding
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        crop = image[y_start:y_end, x_start:x_end]
        
        # Ensure minimum size
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            crop = cv2.resize(crop, (64, 64))
        
        return crop
    
    def _analyze_object_semantics(self, object_crop: np.ndarray, detected_label: str) -> Dict:
        """Analyze object semantics using CLIP or fallback"""
        if self.model_type == "clip":
            return self._analyze_with_clip(object_crop, detected_label)
        else:
            return self._analyze_with_fallback(detected_label)
    
    def _analyze_with_clip(self, object_crop: np.ndarray, detected_label: str) -> Dict:
        """Analyze semantics using CLIP"""
        # Convert BGR to RGB if needed
        if len(object_crop.shape) == 3:
            object_crop = cv2.cvtColor(object_crop, cv2.COLOR_BGR2RGB)
        
        # Process image
        with torch.no_grad():
            inputs = self.processor(images=object_crop, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarities with object categories
            object_similarities = torch.matmul(image_features, self.object_embeddings.T)
            object_scores = torch.softmax(object_similarities * 100, dim=-1)
            
            # Compute similarities with affordances
            affordance_similarities = torch.matmul(image_features, self.affordance_embeddings.T)
            affordance_scores = torch.sigmoid(affordance_similarities * 100)
            
            # Get top predictions
            top_object_idx = torch.argmax(object_scores, dim=-1).item()
            best_category = self.object_categories[top_object_idx]
            best_confidence = object_scores[0, top_object_idx].item()
            
            # Get top affordances
            affordance_threshold = 0.3
            top_affordances = []
            for i, score in enumerate(affordance_scores[0]):
                if score > affordance_threshold:
                    top_affordances.append(self.affordances[i])
            
            # Create clip scores dictionary
            clip_scores = {}
            for i, category in enumerate(self.object_categories):
                clip_scores[category] = object_scores[0, i].item()
        
        return {
            "category": best_category,
            "confidence": best_confidence,
            "affordances": top_affordances[:5],  # Top 5 affordances
            "description": f"{best_category} with {len(top_affordances)} affordances",
            "clip_scores": clip_scores
        }
    
    def _analyze_with_fallback(self, detected_label: str) -> Dict:
        """Analyze semantics using fallback mapping"""
        # Look up in semantic mapping
        if detected_label in self.semantic_mapping:
            semantic_info = self.semantic_mapping[detected_label].copy()
            semantic_info["confidence"] = 0.8
        else:
            # Generic fallback
            semantic_info = {
                "category": "object",
                "affordances": ["graspable", "moveable"],
                "description": f"generic {detected_label}",
                "confidence": 0.5
            }
        
        return semantic_info
    
    def visualize_semantic_tags(self, image: np.ndarray, enhanced_detections: List[Dict],
                               save_path: str = None) -> np.ndarray:
        """
        Visualize semantic tags on image
        
        Args:
            image: Input image
            enhanced_detections: List of semantically enhanced detections
            save_path: Optional path to save visualization
            
        Returns:
            Annotated image
        """
        vis_image = image.copy()
        
        for detection in enhanced_detections:
            x, y, w, h = detection["bbox"]
            x, y, w, h = int(x), int(y), int(w), int(h)  # Ensure integer coordinates
            label = detection["label"]
            category = detection["semantic_category"]
            affordances = detection["affordances"]
            confidence = detection["semantic_confidence"]
            
            # Draw bounding box with different color for semantic info
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Prepare text
            main_text = f"{label} → {category}"
            conf_text = f"conf: {confidence:.2f}"
            affordance_text = f"affordances: {', '.join(affordances[:3])}"
            
            # Draw text background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            texts = [main_text, conf_text, affordance_text]
            text_heights = []
            text_widths = []
            
            for text in texts:
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                text_heights.append(text_h + baseline)
                text_widths.append(text_w)
            
            max_width = max(text_widths)
            total_height = sum(text_heights) + 10
            
            # Draw background rectangle
            cv2.rectangle(vis_image, (int(x), int(y - total_height - 5)), 
                         (int(x + max_width + 10), int(y)), (0, 0, 255), -1)
            
            # Draw texts
            current_y = int(y - total_height + text_heights[0])
            for i, text in enumerate(texts):
                cv2.putText(vis_image, text, (int(x + 5), int(current_y)), 
                           font, font_scale, (255, 255, 255), thickness)
                if i < len(texts) - 1:
                    current_y += text_heights[i + 1] + 2
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"✅ Semantic tagging visualization saved to: {save_path}")
        
        return vis_image


def main():
    """Test the CLIP semantic tagger"""
    tagger = CLIPSemanticTagger()
    
    # Create sample detections for testing
    sample_detections = [
        {
            "label": "bottle",
            "confidence": 0.85,
            "bbox": [100, 50, 80, 150],
            "bbox_xyxy": [100, 50, 180, 200]
        },
        {
            "label": "laptop", 
            "confidence": 0.90,
            "bbox": [300, 100, 200, 120],
            "bbox_xyxy": [300, 100, 500, 220]
        }
    ]
    
    # Load test image
    test_image_path = "../src/img.PNG"
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return
    
    image = cv2.imread(test_image_path)
    print(f"Loaded image shape: {image.shape}")
    
    # Run semantic tagging
    enhanced_detections = tagger.tag_objects(image, sample_detections)
    
    print(f"Enhanced {len(enhanced_detections)} detections with semantic tags:")
    for det in enhanced_detections:
        print(f"  - {det['label']} → {det['semantic_category']}")
        print(f"    Affordances: {', '.join(det['affordances'])}")
        print(f"    Confidence: {det['semantic_confidence']:.2f}")
    
    # Visualize results
    vis_image = tagger.visualize_semantic_tags(image, enhanced_detections, 
                                              "../outputs/semantic_tags.jpg")
    
    print("✅ CLIP semantic tagging test completed!")


if __name__ == "__main__":
    import os
    main()