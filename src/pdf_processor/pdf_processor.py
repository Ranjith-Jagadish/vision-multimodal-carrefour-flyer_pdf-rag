import os
import pdfplumber
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import hashlib
import re


class PDFProcessor:
    """Process PDF files: extract text, create chunks, and generate page images."""
    
    def __init__(self, upload_dir: str, citation_images_dir: str):
        self.upload_dir = Path(upload_dir)
        self.citation_images_dir = Path(citation_images_dir)
        self.citation_images_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_text_and_metadata(self, pdf_path: str) -> Dict:
        """Extract text from PDF with page-level metadata."""
        chunks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    chunks.append({
                        "page_number": page_num,
                        "text": text.strip(),
                        "total_pages": total_pages,
                        "pdf_path": pdf_path
                    })
        
        return {
            "chunks": chunks,
            "total_pages": total_pages,
            "pdf_path": pdf_path
        }
    
    def generate_page_image(self, pdf_path: str, page_number: int, dpi: int = 200) -> str:
        """Generate an image of a specific PDF page for citation."""
        # Try to find existing image by page number (more flexible)
        # First, try exact hash match
        pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()[:8]
        image_filename = f"{pdf_hash}_page_{page_number}.png"
        image_path = self.citation_images_dir / image_filename
        
        # If exact match exists, return it
        if image_path.exists():
            return str(image_path)
        
        # If not found, search for any image with this page number
        # This handles cases where PDF path changed but images exist
        pattern = f"*_page_{page_number}.png"
        import glob
        existing_images = list(self.citation_images_dir.glob(pattern))
        if existing_images:
            # Return the first matching image
            return str(existing_images[0])
        
        # If no existing image, generate new one
        try:
            # Convert specific page to image
            images = convert_from_path(
                pdf_path,
                first_page=page_number,
                last_page=page_number,
                dpi=dpi
            )
            
            if images:
                images[0].save(image_path, "PNG")
                return str(image_path)
        except Exception as e:
            print(f"Error generating page image: {e}")
            return None
        
        return None

    def generate_citation_crop(
        self,
        pdf_path: str,
        page_number: int,
        query: str,
        dpi: int = 200,
        padding_px: int = 40,
        max_words_scan: int = 2000,
    ) -> Optional[str]:
        """
        Generate a cropped citation image for a specific product/term.

        Strategy:
        - Use `pdfplumber` to locate word bounding boxes on the PDF page that match query tokens.
        - Convert PDF coordinates (points) to image pixels using scale = dpi / 72.
        - Crop the rendered page PNG around the matched region with padding.

        Returns:
            Path to cropped PNG in citations dir, or None if no match/crop possible.
        """
        if not query or not query.strip():
            return None

        # Ensure we have the full page image first
        page_img_path = self.generate_page_image(pdf_path, page_number, dpi=dpi)
        if not page_img_path or not os.path.exists(page_img_path):
            return None

        # Build a deterministic crop filename
        pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()[:8]
        q_hash = hashlib.md5(query.strip().lower().encode()).hexdigest()[:8]
        crop_filename = f"{pdf_hash}_page_{page_number}_crop_{q_hash}.png"
        crop_path = self.citation_images_dir / crop_filename
        if crop_path.exists():
            return str(crop_path)

        # Tokenize query (simple, robust)
        tokens = [t.strip().lower() for t in query.replace("?", " ").replace(",", " ").split() if t.strip()]
        # Prefer longer tokens first (e.g., "oreo" > "on")
        tokens = sorted(set(tokens), key=len, reverse=True)
        if not tokens:
            return None

        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_number < 1 or page_number > len(pdf.pages):
                    return None
                page = pdf.pages[page_number - 1]

                # Extract words with bounding boxes in PDF coordinate space (points)
                words = page.extract_words() or []
                if max_words_scan and len(words) > max_words_scan:
                    words = words[:max_words_scan]

                # Find words matching any query token
                matches = []
                for w in words:
                    text = (w.get("text") or "").strip().lower()
                    if not text:
                        continue
                    for tok in tokens:
                        if len(tok) > 3:
                            if text == tok or tok in text:
                                matches.append(w)
                                break
                        else:
                            if tok in text or text in tok:
                                matches.append(w)
                                break

                if not matches:
                    return None

                # Special handling for single product queries (e.g., "oreo")
                # Use a tile-based crop (column/row) around the product name to avoid other products
                if len(tokens) == 1 and len(tokens[0]) > 3:
                    product_token = tokens[0]
                    product_matches = [m for m in matches if product_token in (m.get("text") or "").lower()]
                    if product_matches:
                        anchor = product_matches[0]
                        x_center = (anchor.get("x0", 0) + anchor.get("x1", 0)) / 2
                        y_center = (anchor.get("top", 0) + anchor.get("bottom", 0)) / 2
                        # Compute column/row tile bounds in PDF points
                        page_width = page.width
                        page_height = page.height
                        col_width = page_width / 3.0
                        row_height = page_height / 3.0

                        col_idx = min(2, max(0, int(x_center / col_width)))
                        row_idx = min(2, max(0, int(y_center / row_height)))

                        left = col_idx * col_width
                        right = left + col_width
                        top = row_idx * row_height
                        bottom = top + row_height

                        # Tighten the box slightly and shift upward to include price tag
                        pad = col_width * 0.04
                        left = max(0.0, left + pad)
                        right = min(page_width, right - pad)
                        top = max(0.0, top + row_height * 0.05)
                        bottom = min(page_height, bottom - row_height * 0.05)
                        top = max(0.0, top - row_height * 0.10)

                        # Convert to image pixels
                        scale = dpi / 72.0
                        left_px = int(left * scale)
                        right_px = int(right * scale)
                        top_px = int(top * scale)
                        bottom_px = int(bottom * scale)

                        with Image.open(page_img_path) as img:
                            width, height = img.size
                            left_px = max(0, min(left_px, width - 1))
                            right_px = max(left_px + 1, min(right_px, width))
                            top_px = max(0, min(top_px, height - 1))
                            bottom_px = max(top_px + 1, min(bottom_px, height))

                            if right_px <= left_px or bottom_px <= top_px:
                                return None

                            cropped = img.crop((left_px, top_px, right_px, bottom_px))
                            cropped.save(crop_path, "PNG")
                            return str(crop_path)

                # Union bounding box in PDF coords
                x0 = min(m["x0"] for m in matches if "x0" in m)
                x1 = max(m["x1"] for m in matches if "x1" in m)
                top = min(m["top"] for m in matches if "top" in m)
                bottom = max(m["bottom"] for m in matches if "bottom" in m)

                # Expand the region a bit in PDF coords (in points)
                # Use smaller padding for product-specific queries to get tighter crops
                # For single-word product queries (like "oreo"), use even tighter padding
                if len(tokens) == 1 and len(tokens[0]) > 3:  # Single product name
                    pad_points = (padding_px * 72.0 / dpi) * 0.2  # Very tight crop (20% padding)
                else:
                    pad_points = (padding_px * 72.0 / dpi) * 0.4  # Standard tight crop
                x0 = max(0.0, x0 - pad_points)
                top = max(0.0, top - pad_points)
                x1 = x1 + pad_points
                bottom = bottom + pad_points

                # Convert to image pixel coordinates
                scale = dpi / 72.0
                left_px = int(x0 * scale)
                top_px = int(top * scale)
                right_px = int(x1 * scale)
                bottom_px = int(bottom * scale)

                # Clamp to image bounds
                with Image.open(page_img_path) as img:
                    width, height = img.size
                    left_px = max(0, min(left_px, width - 1))
                    top_px = max(0, min(top_px, height - 1))
                    right_px = max(left_px + 1, min(right_px, width))
                    bottom_px = max(top_px + 1, min(bottom_px, height))

                    cropped = img.crop((left_px, top_px, right_px, bottom_px))
                    cropped.save(crop_path, "PNG")
                    return str(crop_path)
        except Exception as e:
            print(f"Error generating citation crop: {e}")
            return None
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "start_word": i,
                "end_word": min(i + chunk_size, len(words))
            })
        
        return chunks
    
    def process_pdf(self, pdf_path: str, use_vlm: bool = False) -> Dict:
        """
        Main method to process a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            use_vlm: If True, use VLM to extract structured product data
        """
        # Extract text with page metadata
        print(f"[PROCESS] Extracting text: {pdf_path}")
        extracted_data = self.extract_text_and_metadata(pdf_path)
        print(f"[PROCESS] Extracted: pages={extracted_data['total_pages']} chunks={len(extracted_data['chunks'])}")
        
        # Generate page images for all pages
        page_images = {}
        structured_products = []
        
        for chunk in extracted_data["chunks"]:
            page_num = chunk["page_number"]
            if page_num not in page_images:
                image_path = self.generate_page_image(pdf_path, page_num)
                page_images[page_num] = image_path
                
                # Optionally extract structured products using VLM
                if use_vlm and image_path:
                    try:
                        from ..vlm_extractor.vlm_extractor import VLMExtractor
                        # Use Ollama LLaVA by default
                        vlm = VLMExtractor(provider="ollama")
                        print(f"[PROCESS] VLM extract: page {page_num}")
                        products = vlm.extract_products_from_image(image_path, page_num)
                        print(f"[PROCESS] VLM products: page {page_num} count={len(products)}")
                        
                        # Generate crop images for each product
                        for product in products:
                            # Generate crop if bounding box available
                            if product.get("image_bbox"):
                                crop_path = self._crop_product_image(
                                    image_path, 
                                    product["image_bbox"],
                                    page_num,
                                    product.get("product", "")
                                )
                                if crop_path:
                                    product["crop_image_path"] = crop_path
                        
                        structured_products.extend(products)
                    except Exception as e:
                        print(f"⚠️  VLM extraction failed for page {page_num}: {e}")
        
        return {
            "chunks": extracted_data["chunks"],
            "total_pages": extracted_data["total_pages"],
            "page_images": page_images,
            "pdf_path": pdf_path,
            "structured_products": structured_products  # New: VLM-extracted products
        }
    
    def _crop_product_image(self, page_image_path: str, bbox: Dict, page_num: int, product_name: str) -> Optional[str]:
        """Crop product image using bounding box coordinates (pixels or normalized)."""
        try:
            from PIL import Image
            import hashlib
            
            # Load image
            img = Image.open(page_image_path)
            width, height = img.size
            
            # Extract bbox coordinates (support pixels, 0-1 normalized, or 0-100 percent)
            raw_x0 = bbox.get("x0", 0)
            raw_y0 = bbox.get("y0", 0)
            raw_x1 = bbox.get("x1", width)
            raw_y1 = bbox.get("y1", height)

            def to_px(value, axis_len):
                try:
                    v = float(value)
                except Exception:
                    return 0
                if 0 <= v <= 1:
                    return int(v * axis_len)
                if 0 < v <= 100:
                    return int((v / 100.0) * axis_len)
                if v > 1000:
                    return int((v / 1000.0) * axis_len)
                return int(v)

            x0 = to_px(raw_x0, width)
            y0 = to_px(raw_y0, height)
            x1 = to_px(raw_x1, width)
            y1 = to_px(raw_y1, height)
            
            # Clamp to image bounds
            x0 = max(0, min(x0, width - 1))
            y0 = max(0, min(y0, height - 1))
            x1 = max(x0 + 1, min(x1, width))
            y1 = max(y0 + 1, min(y1, height))
            
            # Crop image
            cropped = img.crop((x0, y0, x1, y1))
            
            # Save cropped image
            pdf_hash = hashlib.md5(page_image_path.encode()).hexdigest()[:8]
            product_hash = hashlib.md5(product_name.encode()).hexdigest()[:8]
            crop_filename = f"{pdf_hash}_page_{page_num}_product_{product_hash}.png"
            crop_path = self.citation_images_dir / crop_filename
            cropped.save(crop_path, "PNG")
            
            return str(crop_path)
        except Exception as e:
            print(f"Error cropping product image: {e}")
            return None

