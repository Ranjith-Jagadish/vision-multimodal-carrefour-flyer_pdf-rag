"""
Vision-Language Model (VLM) extractor for structured product data from PDF images.
Uses VLMs to extract structured product information with spatial relationships.
"""
import os
import json
from typing import List, Dict, Optional
from pathlib import Path
from PIL import Image


class VLMExtractor:
    """Extract structured product data from PDF page images using Vision-Language Models."""
    
    def __init__(self, provider: str = "ollama", api_key: Optional[str] = None):
        self.provider = provider or os.getenv("VLM_PROVIDER", "ollama")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_VISION_MODEL", "granite3.2-vision:2b")
        
    def extract_products_from_image(self, image_path: str, page_number: int) -> List[Dict]:
        """
        Extract structured product data from a PDF page image.
        
        Returns list of product dictionaries with:
        - product: Product name
        - price: Current price
        - original_price: Original price (if discounted)
        - discount: Discount percentage
        - quantity: Product quantity/size
        - image_bbox: Bounding box coordinates for product image
        - text_bbox: Bounding box coordinates for product text
        """
        if not os.path.exists(image_path):
            return []
        
        try:
            if self.provider == "openai":
                return self._extract_with_openai(image_path, page_number)
            elif self.provider == "ollama":
                return self._extract_with_ollama(image_path, page_number)
            else:
                raise ValueError(f"Unsupported VLM provider: {self.provider}")
        except Exception as e:
            print(f"Error extracting products with VLM: {e}")
            return []
    
    def _extract_with_openai(self, image_path: str, page_number: int) -> List[Dict]:
        """Extract products using OpenAI GPT-4V."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            # Read image
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            
            # Create prompt for structured extraction
            prompt = """Analyze this retail catalog page image and extract all product offers in JSON format.

For each product, identify:
1. Product name (e.g., "Oreo original biscuit")
2. Current/discounted price (e.g., "17.49")
3. Original price if discounted (e.g., "36.00")
4. Discount percentage if available (e.g., "-51%")
5. Quantity/size information (e.g., "36.8g 24 pieces")
6. Approximate bounding box coordinates (x0, y0, x1, y1) for the product image area
7. Approximate bounding box coordinates for the price/discount text area

Return a JSON array of products. Example format:
[
  {
    "product": "Oreo original biscuit",
    "price": "17.49",
    "original_price": "36.00",
    "discount": "-51%",
    "quantity": "36.8g 24 pieces",
    "image_bbox": {"x0": 100, "y0": 200, "x1": 300, "y1": 400},
    "text_bbox": {"x0": 100, "y0": 400, "x1": 300, "y1": 500}
  }
]

Extract ALL products visible on this page. Be precise with prices and coordinates."""
            
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data.hex()}" if isinstance(image_data, bytes) else image_path
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            products = json.loads(content)
            
            # Add page number to each product
            for product in products:
                product["page_number"] = page_number
                product["image_path"] = image_path
            
            return products
            
        except Exception as e:
            print(f"OpenAI VLM extraction error: {e}")
            return []
    
    def _extract_with_ollama(self, image_path: str, page_number: int) -> List[Dict]:
        """Extract products using Ollama with vision model (e.g., llava)."""
        try:
            import requests
            import base64
            
            # Read and encode image
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                image_data = base64.b64encode(image_bytes).decode('utf-8')
            
            # Use chat API for LLaVA (better support for images)
            prompt = """Analyze this retail catalog page image and extract all product offers in JSON format.

For each product, identify:
1. Product name (e.g., "Oreo original biscuit")
2. Current/discounted price (e.g., "17.49")
3. Original price if discounted (e.g., "36.00")
4. Discount percentage if available (e.g., "-51%")
5. Quantity/size information (e.g., "36.8g 24 pieces")
6. Approximate bounding box coordinates (x0, y0, x1, y1) for the product image area (as percentages of page width/height)
7. Approximate bounding box coordinates for the price/discount text area

Return a JSON array of products. Example format:
[
  {
    "product": "Oreo original biscuit",
    "price": "17.49",
    "original_price": "36.00",
    "discount": "-51%",
    "quantity": "36.8g 24 pieces",
    "image_bbox": {"x0": 10, "y0": 20, "x1": 30, "y1": 40},
    "text_bbox": {"x0": 10, "y0": 40, "x1": 30, "y1": 50}
  }
]

Extract ALL products visible on this page. Be precise with prices and coordinates."""
            
            # Try chat API first (for LLaVA models)
            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                                "images": [image_data]
                            }
                        ],
                        "stream": False
                    },
                    timeout=180
                )
                response.raise_for_status()
                result = response.json()
                content = result.get("message", {}).get("content", "")
            except:
                # Fallback to generate API (older Ollama versions)
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "images": [image_data],
                        "stream": False
                    },
                    timeout=180
                )
                response.raise_for_status()
                result = response.json()
                content = result.get("response", "")
            
            # Parse JSON from response
            # Try to extract JSON from markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                # Try to find JSON in any code block
                parts = content.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # Code blocks are at odd indices
                        part = part.strip()
                        if part.startswith("json"):
                            part = part[4:].strip()
                        if part.startswith("[") or part.startswith("{"):
                            content = part
                            break
            
            # Clean up content - remove any leading/trailing text
            content = content.strip()
            if not content.startswith("["):
                # Try to find JSON array in the content
                start_idx = content.find("[")
                end_idx = content.rfind("]")
                if start_idx != -1 and end_idx != -1:
                    content = content[start_idx:end_idx+1]
            
            # Parse JSON
            try:
                products = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON parse error: {e}")
                print(f"Content preview: {content[:500]}")
                # Try to extract products manually with regex as fallback
                import re
                products = self._extract_products_fallback(content, page_number)
            
            # Validate and add metadata
            if not isinstance(products, list):
                products = [products] if products else []
            
            for product in products:
                if not isinstance(product, dict):
                    continue
                product["page_number"] = page_number
                product["image_path"] = image_path
                # Ensure bbox coordinates are present (even if empty)
                if "image_bbox" not in product:
                    product["image_bbox"] = {}
                if "text_bbox" not in product:
                    product["text_bbox"] = {}
            
            return products
            
        except Exception as e:
            print(f"Ollama VLM extraction error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _extract_products_fallback(self, content: str, page_number: int) -> List[Dict]:
        """Fallback extraction using regex if JSON parsing fails."""
        import re
        products = []
        
        # Look for product patterns in text
        # This is a simple fallback - VLM should ideally return JSON
        product_pattern = r'(?:product|name)[:\s]+([^,\n]+)'
        price_pattern = r'(?:price|discounted)[:\s]+([\d.]+)'
        discount_pattern = r'(?:discount)[:\s]+(-?\d+%)'
        
        # Try to extract at least one product
        product_matches = re.findall(product_pattern, content, re.IGNORECASE)
        price_matches = re.findall(price_pattern, content, re.IGNORECASE)
        discount_matches = re.findall(discount_pattern, content, re.IGNORECASE)
        
        if product_matches:
            for i, product_name in enumerate(product_matches[:5]):  # Limit to 5
                product = {
                    "product": product_name.strip(),
                    "price": price_matches[i] if i < len(price_matches) else "",
                    "discount": discount_matches[i] if i < len(discount_matches) else "",
                    "quantity": "",
                    "image_bbox": {},
                    "text_bbox": {}
                }
                products.append(product)
        
        return products