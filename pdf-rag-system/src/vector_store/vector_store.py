import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Union
import uuid
import numpy as np
import json


class VectorStore:
    """Manage vector embeddings and similarity search using ChromaDB."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 persist_dir: str = "./chroma_db"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.persist_dir = persist_dir
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection for text embeddings
        self.collection = self.client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Get or create collection for image embeddings (multimodal)
        try:
            self.image_collection = self.client.get_or_create_collection(
                name="pdf_images",
                metadata={"hnsw:space": "cosine"}
            )
        except:
            # Fallback if multimodal not available
            self.image_collection = None
        
        # Multimodal embedder (CLIP) - lazy load
        self.multimodal_embedder = None
    
    def add_documents(self, chunks: List[Dict], pdf_name: str):
        """Add document chunks to the vector store."""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Prepare metadata
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            metadata = {
                "page_number": chunk["page_number"],
                "pdf_name": pdf_name,
                "pdf_path": chunk.get("pdf_path", ""),
                "total_pages": chunk.get("total_pages", 0)
            }
            metadatas.append(metadata)
            ids.append(f"{pdf_name}_page_{chunk['page_number']}_chunk_{i}")
        
        # Check for existing IDs to avoid duplicates
        existing_ids = set()
        try:
            existing = self.collection.get(ids=ids)
            if existing and existing.get("ids"):
                existing_ids = set(existing["ids"])
        except:
            pass  # If get fails, proceed with add
        
        # Filter out existing IDs
        new_ids = [id for id in ids if id not in existing_ids]
        if not new_ids:
            print(f"⚠️  All chunks for {pdf_name} already exist in vector store. Skipping duplicate add.")
            return 0
        
        # Filter corresponding data
        new_indices = [i for i, id in enumerate(ids) if id not in existing_ids]
        new_embeddings = [embeddings[i] for i in new_indices]
        new_documents = [texts[i] for i in new_indices]
        new_metadatas = [metadatas[i] for i in new_indices]
        
        # Add only new documents
        if new_embeddings:
            self.collection.add(
                embeddings=new_embeddings,
                documents=new_documents,
                metadatas=new_metadatas,
                ids=new_ids
            )
        
        return len(new_ids)
    
    def search(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents."""
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Build where clause for filtering if needed
        where = filter_dict if filter_dict else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        
        # Format results
        formatted_results = []
        if results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
        
        return formatted_results
    
    def delete_pdf(self, pdf_name: str):
        """Delete all chunks for a specific PDF."""
        # Get all documents for this PDF
        all_docs = self.collection.get()
        
        # Find IDs to delete
        ids_to_delete = [
            doc_id for doc_id, metadata in zip(all_docs["ids"], all_docs["metadatas"])
            if metadata.get("pdf_name") == pdf_name
        ]
        
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        
        return len(ids_to_delete)

    def has_multimodal_data(self, filter_dict: Optional[Dict] = None) -> bool:
        """Check if any multimodal (image) embeddings exist."""
        if not self.image_collection:
            return False
        try:
            if filter_dict:
                return self.image_collection.count(where=filter_dict) > 0
            return self.image_collection.count() > 0
        except Exception:
            return False
    
    def get_all_pdfs(self) -> List[str]:
        """Get list of all unique PDF names in the store."""
        all_docs = self.collection.get()
        pdf_names = set()
        
        for metadata in all_docs["metadatas"]:
            if "pdf_name" in metadata:
                pdf_names.add(metadata["pdf_name"])
        
        return list(pdf_names)
    
    def _get_multimodal_embedder(self):
        """Lazy load multimodal embedder."""
        if self.multimodal_embedder is None:
            try:
                from ..multimodal_embeddings.multimodal_embeddings import MultimodalEmbedder
                self.multimodal_embedder = MultimodalEmbedder()
            except Exception as e:
                print(f"⚠️  Multimodal embedder not available: {e}")
                return None
        return self.multimodal_embedder
    
    def add_multimodal_documents(self, products: List[Dict], pdf_name: str):
        """
        Add structured product data with both text and image embeddings.
        
        Args:
            products: List of product dicts from VLM extractor
            pdf_name: Name of the PDF file
        """
        if not products:
            return 0
        
        embedder = self._get_multimodal_embedder()
        if not embedder or not self.image_collection:
            print("⚠️  Multimodal embeddings not available, falling back to text-only")
            return 0
        
        # Prepare text and image embeddings
        texts = []
        text_embeddings = []
        image_embeddings = []
        metadatas = []
        ids = []
        
        for i, product in enumerate(products):
            # Create text description for embedding
            product_text = f"{product.get('product', '')} {product.get('quantity', '')} {product.get('price', '')} {product.get('discount', '')}"
            texts.append(product_text)
            
            # Generate text embedding
            text_emb = embedder.embed_text(product_text)
            text_embeddings.append(text_emb[0].tolist())
            
            # Generate image embedding if crop exists
            image_path = product.get("crop_image_path") or product.get("image_path")
            if image_path and os.path.exists(image_path):
                img_emb = embedder.embed_image(image_path)
                image_embeddings.append(img_emb[0].tolist())
            else:
                # Use text embedding as fallback
                image_embeddings.append(text_emb[0].tolist())
            
            # Prepare metadata
            metadata = {
                "page_number": product.get("page_number", 0),
                "pdf_name": pdf_name,
                "product": product.get("product", ""),
                "price": product.get("price", ""),
                "original_price": product.get("original_price", ""),
                "discount": product.get("discount", ""),
                "quantity": product.get("quantity", ""),
                "image_path": image_path or "",
                "image_bbox": json.dumps(product.get("image_bbox", {})) if product.get("image_bbox") else "",
                "text_bbox": json.dumps(product.get("text_bbox", {})) if product.get("text_bbox") else "",
            }
            metadatas.append(metadata)
            ids.append(f"{pdf_name}_product_{i}_{product.get('product', '').replace(' ', '_')[:20]}")
        
        # Add text embeddings to main collection
        self.collection.add(
            embeddings=text_embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Add image embeddings to image collection
        if self.image_collection:
            self.image_collection.add(
                embeddings=image_embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        return len(ids)
    
    def hybrid_search(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None,
                     text_weight: float = 0.6, image_weight: float = 0.4) -> List[Dict]:
        """
        Hybrid search combining text and image similarity.
        
        Args:
            query: Text query string
            top_k: Number of results to return
            filter_dict: Optional filter by PDF name
            text_weight: Weight for text similarity (0-1)
            image_weight: Weight for image similarity (0-1)
        """
        embedder = self._get_multimodal_embedder()
        if not embedder or not self.image_collection:
            # Fallback to text-only search
            return self.search(query, top_k, filter_dict)
        
        # Generate query embeddings
        query_text_emb = self.embedding_model.encode([query]).tolist()[0]
        query_image_emb = embedder.embed_text(query)[0]  # Use text embedding as query
        
        where = filter_dict if filter_dict else None
        
        # Search text collection
        text_results = self.collection.query(
            query_embeddings=[query_text_emb],
            n_results=top_k * 2,  # Get more for combining
            where=where
        )
        
        # Search image collection
        image_results = self.image_collection.query(
            query_embeddings=[query_image_emb.tolist()],
            n_results=top_k * 2,
            where=where
        )
        
        # Combine and score results
        combined_scores = {}
        
        # Process text results
        if text_results["documents"] and len(text_results["documents"][0]) > 0:
            for i, doc_id in enumerate(text_results["ids"][0]):
                distance = text_results["distances"][0][i] if "distances" in text_results else 1.0
                similarity = 1 - distance
                combined_scores[doc_id] = {
                    "text_score": similarity * text_weight,
                    "image_score": 0,
                    "metadata": text_results["metadatas"][0][i],
                    "text": text_results["documents"][0][i]
                }
        
        # Process image results
        if image_results["documents"] and len(image_results["documents"][0]) > 0:
            for i, doc_id in enumerate(image_results["ids"][0]):
                distance = image_results["distances"][0][i] if "distances" in image_results else 1.0
                similarity = 1 - distance
                if doc_id in combined_scores:
                    combined_scores[doc_id]["image_score"] = similarity * image_weight
                else:
                    combined_scores[doc_id] = {
                        "text_score": 0,
                        "image_score": similarity * image_weight,
                        "metadata": image_results["metadatas"][0][i],
                        "text": image_results["documents"][0][i]
                    }
        
        # Calculate combined scores and sort
        scored_results = []
        for doc_id, data in combined_scores.items():
            combined_score = data["text_score"] + data["image_score"]
            scored_results.append({
                "text": data["text"],
                "metadata": data["metadata"],
                "distance": 1 - combined_score,  # Convert back to distance
                "text_score": data["text_score"],
                "image_score": data["image_score"],
                "combined_score": combined_score
            })
        
        # Sort by combined score (descending)
        scored_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return scored_results[:top_k]

