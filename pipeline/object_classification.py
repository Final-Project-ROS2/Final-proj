import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple, Optional
import logging

class ObjectClassification:
    """
    Stage 2: Object Classification using CLIP and DINOv2 (with fallback)
    Uses both models for robust tool classification from bounding boxes
    Falls back to enhanced traditional methods if transformers not available
    """
    def __init__(self, device: str = "auto", use_dino: bool = True):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing classification models on {self.device}...")
        
        # Tool classes for robotic manipulation
        self.tool_classes = [
            "hammer", "screwdriver", "pliers", "wrench", "drill", 
            "saw", "chisel", "file", "clamp", "measuring tape",
            "scissors", "knife", "allen key", "socket wrench", "unknown tool"
        ]
        
        # Text descriptions for better CLIP matching
        self.tool_descriptions = [
            "a hammer tool for hitting nails",
            "a screwdriver for turning screws", 
            "pliers for gripping objects",
            "a wrench for turning nuts and bolts",
            "a power drill for making holes",
            "a saw for cutting materials",
            "a chisel for carving",
            "a metal file for smoothing surfaces",
            "a clamp for holding objects",
            "a measuring tape for measuring distances",
            "scissors for cutting paper or fabric",
            "a utility knife for cutting",
            "an allen key hexagonal tool",
            "a socket wrench with interchangeable heads",
            "an unknown or unidentified tool"
        ]
        
        # Try to initialize CLIP model
        self.clip_model = None
        self.clip_processor = None
        self.use_real_clip = self._try_init_clip()
        
        # Try to initialize DINOv2 model (optional)
        self.dino_model = None
        self.dino_processor = None
        self.use_dino = use_dino and self._try_init_dino()
        
        # Pre-compute embeddings (real or simulated)
        self.tool_embeddings = self._compute_tool_embeddings()
        
    def _try_init_clip(self):
        """Try to initialize CLIP model, fallback if not available"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            print("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()
            print("✅ CLIP model loaded successfully")
            return True
        except ImportError:
            print("⚠️ transformers not available - using enhanced fallback classification")
            return False
        except Exception as e:
            print(f"⚠️ Could not load CLIP model: {e}")
            print("Using enhanced fallback classification...")
            return False
            
    def _try_init_dino(self):
        """Try to initialize DINOv2 model, fallback if not available"""
        try:
            from transformers import AutoImageProcessor, AutoModel
            print("Loading DINOv2 model...")
            self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
            self.dino_model.eval()
            print("✅ DINOv2 model loaded successfully")
            return True
        except ImportError:
            print("⚠️ transformers not available - skipping DINOv2")
            return False
        except Exception as e:
            print(f"⚠️ Could not load DINOv2 model: {e}")
            return False
            
    def _compute_tool_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute tool embeddings (real CLIP or enhanced simulated)"""
        if self.use_real_clip:
            return self._compute_real_clip_embeddings()
        else:
            return self._compute_enhanced_simulated_embeddings()
    
    def _compute_real_clip_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute real CLIP text embeddings for all tool descriptions"""
        import torch.nn.functional as F
        
        with torch.no_grad():
            text_inputs = self.clip_processor(
                text=self.tool_descriptions, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            text_embeddings = self.clip_model.get_text_features(**text_inputs)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
            
        # Convert to dictionary
        embeddings = {}
        for i, tool_class in enumerate(self.tool_classes):
            embeddings[tool_class] = text_embeddings[i].cpu().numpy()
            
        return embeddings
    
    def _compute_enhanced_simulated_embeddings(self) -> Dict[str, np.ndarray]:
        """Create enhanced simulated embeddings based on tool characteristics"""
        embeddings = {}
        
        # Define tool characteristics for better simulation
        tool_characteristics = {
            "hammer": {"weight": 0.9, "metal": 0.8, "handle": 0.9, "striking": 0.9},
            "screwdriver": {"weight": 0.4, "metal": 0.7, "handle": 0.8, "precision": 0.9},
            "pliers": {"weight": 0.6, "metal": 0.9, "gripping": 0.9, "articulated": 0.8},
            "wrench": {"weight": 0.7, "metal": 0.9, "handle": 0.7, "turning": 0.9},
            "drill": {"weight": 0.8, "metal": 0.6, "motor": 0.9, "precision": 0.8},
            "saw": {"weight": 0.5, "metal": 0.8, "cutting": 0.9, "teeth": 0.9},
            "chisel": {"weight": 0.4, "metal": 0.9, "handle": 0.8, "sharp": 0.9},
            "file": {"weight": 0.3, "metal": 0.9, "handle": 0.6, "texture": 0.9},
            "clamp": {"weight": 0.6, "metal": 0.8, "gripping": 0.9, "adjustable": 0.8},
            "measuring tape": {"weight": 0.2, "flexible": 0.9, "numbers": 0.9, "yellow": 0.7},
            "scissors": {"weight": 0.3, "metal": 0.8, "cutting": 0.9, "articulated": 0.9},
            "knife": {"weight": 0.3, "metal": 0.8, "sharp": 0.9, "cutting": 0.9},
            "allen key": {"weight": 0.2, "metal": 0.9, "hexagonal": 0.9, "small": 0.8},
            "socket wrench": {"weight": 0.6, "metal": 0.9, "handle": 0.8, "socket": 0.9},
            "unknown tool": {"weight": 0.5, "metal": 0.5, "handle": 0.5, "generic": 0.5}
        }
        
        for tool_class in self.tool_classes:
            characteristics = tool_characteristics.get(tool_class, {})
            
            # Create a more realistic embedding based on characteristics
            embedding = np.random.randn(512) * 0.1  # Base noise
            
            # Add characteristic-based features
            for i, (char, value) in enumerate(characteristics.items()):
                start_idx = i * 50
                end_idx = min(start_idx + 50, 512)
                embedding[start_idx:end_idx] += value
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings[tool_class] = embedding
            
        return embeddings
        
    def classify_objects(self, image: np.ndarray, masks: List[np.ndarray], 
                        boxes: List[List[float]] = None) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Classify objects using CLIP, DINOv2, or enhanced fallback methods
        
        Args:
            image: RGB image
            masks: List of binary masks
            boxes: Optional list of bounding boxes [x, y, w, h]
            
        Returns:
            labels: List of predicted class labels
            confidences: List of confidence scores
            details: List of detailed classification info
        """
        labels = []
        confidences = []
        details = []
        
        print(f"\n🔍 Classifying {len(masks)} objects using {'CLIP' if self.use_real_clip else 'Enhanced Fallback'}...")
        
        for i, mask in enumerate(masks):
            print(f"   Processing object {i+1}/{len(masks)}...")
            
            # Extract object crop
            if boxes and i < len(boxes):
                object_crop = self._extract_object_crop_from_bbox(image, boxes[i])
            else:
                object_crop = self._extract_object_crop_from_mask(image, mask)
            
            # Get classification results
            classification_result = self._classify_single_object(object_crop, mask)
            
            labels.append(classification_result['label'])
            confidences.append(classification_result['confidence'])
            details.append(classification_result)
            
        return labels, confidences, details
    
    def _extract_object_crop_from_bbox(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Extract object region using bounding box"""
        x, y, w, h = [int(v) for v in bbox]
        
        # Add padding
        padding = 20
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        crop = image[y_start:y_end, x_start:x_end]
        return crop
    
    def _extract_object_crop_from_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract object region using mask"""
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return image[:100, :100]  # Return small patch if mask is empty
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Add padding
        padding = 20
        y_min = max(0, y_min - padding)
        y_max = min(image.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        
        crop = image[y_min:y_max, x_min:x_max]
        return crop
    
    def _classify_single_object(self, object_crop: np.ndarray, mask: np.ndarray = None) -> Dict:
        """Classify a single object using available methods"""
        result = {
            'label': 'unknown tool',
            'confidence': 0.0,
            'clip_scores': {},
            'features': None,
            'method': 'fallback'
        }
        
        # Ensure minimum size
        if object_crop.shape[0] < 10 or object_crop.shape[1] < 10:
            return result
            
        if self.use_real_clip:
            # Use real CLIP
            from PIL import Image
            pil_image = Image.fromarray(cv2.cvtColor(object_crop, cv2.COLOR_BGR2RGB))
            clip_result = self._classify_with_real_clip(pil_image)
            result.update(clip_result)
            result['method'] = 'clip'
            
            # Add DINOv2 if available
            if self.use_dino:
                dino_features = self._extract_dino_features(pil_image)
                result['features'] = dino_features
                result['method'] = 'clip+dino'
        else:
            # Use enhanced fallback
            enhanced_result = self._classify_with_enhanced_fallback(object_crop)
            result.update(enhanced_result)
            result['method'] = 'enhanced_fallback'
            
        return result
    
    def _classify_with_real_clip(self, pil_image) -> Dict:
        """Classify object using real CLIP model"""
        import torch.nn.functional as F
        
        try:
            with torch.no_grad():
                # Process image
                image_inputs = self.clip_processor(
                    images=pil_image, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Get image features
                image_features = self.clip_model.get_image_features(**image_inputs)
                image_features = F.normalize(image_features, p=2, dim=1)
                
                # Calculate similarities with text embeddings
                similarities = []
                for tool_class in self.tool_classes:
                    tool_embedding = torch.tensor(self.tool_embeddings[tool_class]).unsqueeze(0).to(self.device)
                    similarity = torch.matmul(image_features, tool_embedding.T).item()
                    similarities.append(similarity)
                
                similarities = torch.tensor(similarities)
                probs = F.softmax(similarities * 100, dim=0)  # Temperature scaling
                
                # Get best match
                best_idx = torch.argmax(probs)
                best_label = self.tool_classes[best_idx]
                best_confidence = probs[best_idx].item()
                
                # Create scores dictionary
                clip_scores = {}
                for i, (tool, prob) in enumerate(zip(self.tool_classes, probs)):
                    clip_scores[tool] = prob.item()
                
                return {
                    'label': best_label,
                    'confidence': best_confidence,
                    'clip_scores': clip_scores
                }
                
        except Exception as e:
            print(f"   ⚠️ CLIP classification error: {e}")
            return self._classify_with_enhanced_fallback(None)
    
    def _classify_with_enhanced_fallback(self, object_crop: np.ndarray) -> Dict:
        """Enhanced fallback classification using computer vision features"""
        # Extract visual features
        features = self._extract_visual_features(object_crop)
        
        # Compare with tool embeddings using cosine similarity
        similarities = {}
        for tool_class, tool_embedding in self.tool_embeddings.items():
            # Use first part of embedding for comparison
            feature_sim = np.dot(features, tool_embedding[:len(features)])
            similarities[tool_class] = max(0, feature_sim)  # Ensure positive
        
        # Get best match
        best_tool = max(similarities, key=similarities.get)
        best_confidence = similarities[best_tool]
        
        # Normalize confidences
        total_sim = sum(similarities.values())
        if total_sim > 0:
            normalized_scores = {k: v/total_sim for k, v in similarities.items()}
            best_confidence = normalized_scores[best_tool]
        else:
            normalized_scores = similarities
            best_confidence = 0.1
        
        return {
            'label': best_tool,
            'confidence': best_confidence,
            'clip_scores': normalized_scores
        }
    
    def _extract_visual_features(self, object_crop: np.ndarray) -> np.ndarray:
        """Extract visual features for fallback classification"""
        if object_crop is None:
            return np.random.randn(20) * 0.1
            
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(object_crop, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        # Shape features
        features.append(gray.shape[0] / gray.shape[1])  # Aspect ratio
        features.append(np.mean(gray) / 255.0)  # Average brightness
        features.append(np.std(gray) / 255.0)   # Contrast
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.sum(edges > 0) / edges.size)  # Edge density
        
        # Color features (in BGR)
        for channel in range(3):
            channel_data = object_crop[:, :, channel]
            features.append(np.mean(channel_data) / 255.0)
            features.append(np.std(channel_data) / 255.0)
        
        # Texture features (simplified)
        features.append(np.mean(np.abs(np.diff(gray, axis=0))))  # Vertical texture
        features.append(np.mean(np.abs(np.diff(gray, axis=1))))  # Horizontal texture
        
        # More shape analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            features.append(cv2.contourArea(largest_contour) / gray.size)  # Contour density
            features.append(cv2.arcLength(largest_contour, True) / (gray.shape[0] + gray.shape[1]))  # Perimeter ratio
        else:
            features.extend([0.0, 0.0])
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)
            
        return np.array(features[:20])
    
    def _extract_dino_features(self, pil_image) -> Optional[np.ndarray]:
        """Extract DINOv2 features for the object"""
        try:
            import torch.nn.functional as F
            with torch.no_grad():
                inputs = self.dino_processor(images=pil_image, return_tensors="pt").to(self.device)
                outputs = self.dino_model(**inputs)
                
                # Use [CLS] token features
                features = outputs.last_hidden_state[:, 0]  # [CLS] token
                features = F.normalize(features, p=2, dim=1)
                
                return features.cpu().numpy().flatten()
                
        except Exception as e:
            print(f"   ⚠️ DINOv2 feature extraction error: {e}")
            return None
    
    def get_top_k_predictions(self, clip_scores: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k predictions from scores"""
        sorted_scores = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:k]
    
    def visualize_classification(self, image: np.ndarray, boxes: List[List[float]], 
                               labels: List[str], confidences: List[float],
                               save_path: str = None) -> np.ndarray:
        """Visualize classification results on image"""
        vis_image = image.copy()
        
        for i, (bbox, label, conf) in enumerate(zip(boxes, labels, confidences)):
            x, y, w, h = [int(v) for v in bbox]
            
            # Color based on confidence
            if conf > 0.7:
                color = (0, 255, 0)  # Green - high confidence
            elif conf > 0.4:
                color = (0, 165, 255)  # Orange - medium confidence  
            else:
                color = (0, 0, 255)  # Red - low confidence
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with confidence
            label_text = f"{label} ({conf:.2f})"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background rectangle for text
            cv2.rectangle(vis_image, (x, y - 25), (x + label_size[0] + 10, y), color, -1)
            
            # Text
            cv2.putText(vis_image, label_text, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            
        return vis_image
