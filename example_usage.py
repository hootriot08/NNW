"""
Example usage of the Embedding Service
Demonstrates various ways to use the embedding functionality.
"""

from embedding_service import EmbeddingService, get_embedding, get_embeddings, check_server_status
import numpy as np

def example_basic_usage():
    """Basic usage examples"""
    print("=== Basic Usage Examples ===")
    
    # Check if server is available
    if not check_server_status():
        print("❌ LM Studio server is not available. Please start LM Studio first.")
        return
    
    print("✅ Server is available!")
    
    # Single embedding
    text = "The quick brown fox jumps over the lazy dog."
    embedding = get_embedding(text)
    
    if embedding:
        print(f"✅ Single embedding successful!")
        print(f"   Text: '{text}'")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
    else:
        print("❌ Single embedding failed")
    
    print()

def example_batch_processing():
    """Example of processing multiple texts"""
    print("=== Batch Processing Example ===")
    
    # Multiple texts
    texts = [
        "Machine learning is fascinating.",
        "Natural language processing helps computers understand text.",
        "Embeddings represent text as numerical vectors.",
        "Vector similarity can find related documents.",
        "AI models can generate human-like text."
    ]
    
    embeddings = get_embeddings(texts)
    successful_embeddings = [e for e in embeddings if e is not None]
    
    print(f"✅ Processed {len(successful_embeddings)}/{len(texts)} embeddings successfully")
    
    if successful_embeddings:
        print(f"   Embedding dimension: {len(successful_embeddings[0])}")
        print("   Sample embeddings:")
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            if embedding:
                print(f"   {i+1}. '{text[:30]}...' -> {len(embedding)} dimensions")
    
    print()

def example_similarity_search():
    """Example of finding similar texts using embeddings"""
    print("=== Similarity Search Example ===")
    
    # Sample documents
    documents = [
        "Python is a popular programming language for data science.",
        "Machine learning algorithms can predict future outcomes.",
        "Data visualization helps understand complex datasets.",
        "Python libraries like pandas make data analysis easier.",
        "Deep learning models require large amounts of training data."
    ]
    
    # Query
    query = "What programming language is good for data analysis?"
    
    # Get embeddings
    doc_embeddings = get_embeddings(documents)
    query_embedding = get_embedding(query)
    
    if query_embedding and all(e is not None for e in doc_embeddings):
        # Calculate similarities
        similarities = []
        for doc_embedding in doc_embeddings:
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append(similarity)
        
        # Sort by similarity
        results = list(zip(documents, similarities))
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Query: '{query}'")
        print("Most similar documents:")
        for i, (doc, sim) in enumerate(results):
            print(f"   {i+1}. (similarity: {sim:.3f}) {doc}")
    else:
        print("❌ Failed to get embeddings for similarity search")
    
    print()

def example_service_class():
    """Example using the EmbeddingService class directly"""
    print("=== Service Class Example ===")
    
    # Initialize service
    service = EmbeddingService()
    
    # Check server info
    server_info = service.get_server_info()
    if server_info:
        print("✅ Server info retrieved:")
        print(f"   Available models: {len(server_info.get('data', []))}")
    
    # Batch processing with the service
    texts = ["Service class example", "Batch processing", "Error handling"]
    embeddings = service.get_embeddings_batch(texts, batch_size=2)
    
    successful = [e for e in embeddings if e is not None]
    print(f"✅ Batch processing: {len(successful)}/{len(texts)} successful")
    
    print()

def main():
    """Run all examples"""
    print("🚀 Embedding Service Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_batch_processing()
        example_similarity_search()
        example_service_class()
        
        print("✅ All examples completed!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")

if __name__ == "__main__":
    main()
