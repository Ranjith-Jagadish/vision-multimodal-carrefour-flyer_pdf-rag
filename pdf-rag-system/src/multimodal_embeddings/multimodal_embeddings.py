"""
Multimodal embeddings using CLIP for text and image similarity.
Enables searching products by both text descriptions and visual appearance.
"""
import os
from typing import List, Union, Tuple
import numpy as np
from PIL import Image


class MultimodalEmbedder:
    """Generate CLIP embeddings for both text and images."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load CLIP model and processor."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # Use GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Loaded CLIP model: {self.model_name} on {self.device}")
        except ImportError:
            print("⚠️  transformers not installed. Install with: pip install transformers")
            self.model = None
        except Exception as e:
            print(f"⚠️  Error loading CLIP model: {e}")
            self.model = None
    
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)."""
        if not self.model:
            raise RuntimeError("CLIP model not loaded. Install transformers package.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            import torch
            
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                # Normalize
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy()
        except Exception as e:
            print(f"Error embedding text: {e}")
            return np.zeros((len(texts), 512))  # Fallback
    
    def embed_image(self, image_paths: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for image(s)."""
        if not self.model:
            raise RuntimeError("CLIP model not loaded. Install transformers package.")
        
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        try:
            import torch
            
            images = []
            for img_path in image_paths:
                if not os.path.exists(img_path):
                    # Create blank image as fallback
                    images.append(Image.new('RGB', (224, 224), color='white'))
                else:
                    images.append(Image.open(img_path).convert('RGB'))
            
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()
        except Exception as e:
            print(f"Error embedding image: {e}")
            return np.zeros((len(image_paths), 512))  # Fallback
    
    def compute_similarity(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and candidates."""
        # query_embedding: (1, dim) or (dim,)
        # candidate_embeddings: (n, dim)
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Cosine similarity = dot product (since embeddings are normalized)
        similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()
        return similarities
