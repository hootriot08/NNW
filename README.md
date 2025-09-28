# NNW - Embedding Service Integration

This project integrates with LM Studio's local embedding API to provide easy-to-use text embedding functionality.

## Setup

### Prerequisites
1. **LM Studio**: Download and install [LM Studio](https://lmstudio.ai/)
2. **Model**: Load the `text-embedding-bge-small-en-v1.5` model in LM Studio
3. **Server**: Start the LM Studio server on `http://127.0.0.1:1234`

### Installation
```bash
# Install Python dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start
```python
from embedding_service import get_embedding, get_embeddings

# Single embedding
embedding = get_embedding("Your text here")
print(f"Embedding dimension: {len(embedding)}")

# Multiple embeddings
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = get_embeddings(texts)
```

### Service Class Usage
```python
from embedding_service import EmbeddingService

# Initialize service
service = EmbeddingService()

# Check server status
if service.is_server_available():
    # Get single embedding
    embedding = service.get_embedding("Sample text")
    
    # Get multiple embeddings
    embeddings = service.get_embeddings(["Text 1", "Text 2"])
    
    # Batch processing for large datasets
    embeddings = service.get_embeddings_batch(large_text_list, batch_size=10)
```

### Example Applications

#### 1. Document Similarity
```python
from embedding_service import get_embeddings, get_embedding
import numpy as np

# Your documents
documents = ["Document 1 text", "Document 2 text", "Document 3 text"]
query = "Search query"

# Get embeddings
doc_embeddings = get_embeddings(documents)
query_embedding = get_embedding(query)

# Calculate similarities
similarities = []
for doc_embedding in doc_embeddings:
    similarity = np.dot(query_embedding, doc_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
    )
    similarities.append(similarity)

# Find most similar document
most_similar_idx = np.argmax(similarities)
print(f"Most similar: {documents[most_similar_idx]}")
```

#### 2. Text Classification
```python
# Get embeddings for training data
training_texts = ["Positive text 1", "Negative text 1", "Positive text 2"]
training_embeddings = get_embeddings(training_texts)

# Use embeddings with your favorite ML library (scikit-learn, etc.)
```

## API Reference

### EmbeddingService Class

#### `__init__(base_url, model)`
- `base_url`: LM Studio server URL (default: "http://127.0.0.1:1234")
- `model`: Model identifier (default: "text-embedding-bge-small-en-v1.5")

#### Methods
- `get_embedding(text)`: Get embedding for single text
- `get_embeddings(texts)`: Get embeddings for multiple texts
- `get_embeddings_batch(texts, batch_size)`: Process texts in batches
- `is_server_available()`: Check if server is reachable
- `get_server_info()`: Get server model information

### Convenience Functions
- `get_embedding(text, base_url)`: Quick single embedding
- `get_embeddings(texts, base_url)`: Quick multiple embeddings
- `check_server_status(base_url)`: Quick server check

## Running Examples

```bash
# Test the embedding service
python embedding_service.py

# Run comprehensive examples
python example_usage.py
```

## Configuration

The service uses these default settings:
- **Server URL**: `http://127.0.0.1:1234`
- **Model**: `text-embedding-bge-small-en-v1.5`
- **Timeout**: 30 seconds for requests
- **Batch size**: 10 texts per batch

You can customize these by passing parameters to the `EmbeddingService` constructor.

## Troubleshooting

### Server Not Available
- Ensure LM Studio is running
- Check that the server is started on port 1234
- Verify the model is loaded in LM Studio

### Connection Errors
- Check firewall settings
- Ensure the server URL is correct
- Try increasing the timeout value

### Memory Issues
- Use batch processing for large datasets
- Reduce batch size if experiencing memory issues
- Consider processing texts in smaller chunks

## Files

- `embedding_service.py`: Main service module
- `example_usage.py`: Comprehensive usage examples
- `requirements.txt`: Python dependencies
- `eps_comparison_comprehensive_with_eps_values.csv.csv`: Your EPS data file
