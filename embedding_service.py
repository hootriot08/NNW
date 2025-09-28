"""
Embedding Service for LM Studio Integration
Provides easy-to-use functions for text embedding using the local LM Studio server.
"""

import requests
import json
from typing import List, Union, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    A service class for handling text embeddings using LM Studio's local API.
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:1234", model: str = "text-embedding-bge-small-en-v1.5"):
        """
        Initialize the embedding service.
        
        Args:
            base_url (str): The base URL of the LM Studio server
            model (str): The model identifier for embeddings
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.embeddings_endpoint = f"{self.base_url}/v1/embeddings"
        
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for a single text string.
        
        Args:
            text (str): The text to embed
            
        Returns:
            List[float]: The embedding vector, or None if failed
        """
        try:
            response = requests.post(
                self.embeddings_endpoint,
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "input": text
                },
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data['data'][0]['embedding']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    
    def get_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Get embeddings for multiple text strings.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[Optional[List[float]]]: List of embedding vectors (None for failed embeddings)
        """
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[Optional[List[float]]]:
        """
        Get embeddings for multiple texts in batches to avoid overwhelming the server.
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Number of texts to process in each batch
            
        Returns:
            List[Optional[List[float]]]: List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.get_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Small delay between batches to be respectful to the server
            if i + batch_size < len(texts):
                import time
                time.sleep(0.1)
        
        return all_embeddings
    
    def is_server_available(self) -> bool:
        """
        Check if the LM Studio server is available.
        
        Returns:
            bool: True if server is reachable, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_server_info(self) -> Optional[dict]:
        """
        Get information about the available models on the server.
        
        Returns:
            dict: Server model information, or None if failed
        """
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
            return response.json()
        except:
            return None

# Convenience functions for direct use
def get_embedding(text: str, base_url: str = "http://127.0.0.1:1234") -> Optional[List[float]]:
    """
    Quick function to get a single embedding.
    
    Args:
        text (str): Text to embed
        base_url (str): LM Studio server URL
        
    Returns:
        List[float]: Embedding vector, or None if failed
    """
    service = EmbeddingService(base_url)
    return service.get_embedding(text)

def get_embeddings(texts: List[str], base_url: str = "http://127.0.0.1:1234") -> List[Optional[List[float]]]:
    """
    Quick function to get multiple embeddings.
    
    Args:
        texts (List[str]): List of texts to embed
        base_url (str): LM Studio server URL
        
    Returns:
        List[Optional[List[float]]]: List of embedding vectors
    """
    service = EmbeddingService(base_url)
    return service.get_embeddings(texts)

def check_server_status(base_url: str = "http://127.0.0.1:1234") -> bool:
    """
    Quick function to check if the server is available.
    
    Args:
        base_url (str): LM Studio server URL
        
    Returns:
        bool: True if server is available
    """
    service = EmbeddingService(base_url)
    return service.is_server_available()

if __name__ == "__main__":
    # Example usage
    print("Testing Embedding Service...")
    
    # Check server status
    if check_server_status():
        print("✅ Server is available")
        
        # Test single embedding
        test_text = "This is a test sentence for embedding."
        embedding = get_embedding(test_text)
        
        if embedding:
            print(f"✅ Single embedding successful (dimension: {len(embedding)})")
        else:
            print("❌ Single embedding failed")
        
        # Test multiple embeddings
        test_texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        
        embeddings = get_embeddings(test_texts)
        successful_embeddings = [e for e in embeddings if e is not None]
        
        print(f"✅ Multiple embeddings: {len(successful_embeddings)}/{len(test_texts)} successful")
        
    else:
        print("❌ Server is not available. Make sure LM Studio is running.")
