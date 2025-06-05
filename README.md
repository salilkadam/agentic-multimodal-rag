# Agentic Multimodal RAG System

An advanced Retrieval-Augmented Generation (RAG) system with Graphiti knowledge graph integration, multimodal processing capabilities, and vector database support.

## Features

- **Agentic RAG**: Multi-step reasoning with LLM-powered agents
- **Knowledge Graph Integration**: Uses Graphiti for entity relationship mapping
- **Multimodal Processing**: Support for text and image embeddings
- **Vector Database**: Milvus for efficient similarity search
- **Graph Database**: Neo4j for complex relationship queries
- **Caching**: Redis for performance optimization
- **Offline Mode**: Pre-downloaded models for air-gapped environments

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   Milvus        │    │   Neo4j         │
│   (RAG Engine)  │────│  (Vectors)      │    │  (Knowledge)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              │
         │              ┌─────────────────┐             │
         └──────────────│     Redis       │─────────────┘
                        │   (Cache)       │
                        └─────────────────┘
```

## Quick Start

### Prerequisites

- Kubernetes cluster (k3s/k8s)
- Container runtime (Docker/Podman)
- kubectl configured

### Deployment

1. **Build the image:**
   ```bash
   cd rag
   ./scripts/build_and_push.sh
   ```

2. **Deploy to Kubernetes:**
   ```bash
   kubectl apply -f k8s/
   ```

3. **Check deployment status:**
   ```bash
   kubectl get pods -n rag
   ```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | Sentence transformer model | `sentence-transformers/all-MiniLM-L6-v2` |
| `ENABLE_MULTIMODAL` | Enable image processing | `false` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `REDIS_HOST` | Redis hostname | `redis-service` |
| `NEO4J_URI` | Neo4j connection URI | `bolt://neo4j:7687` |
| `MILVUS_HOST` | Milvus hostname | `milvus-service` |

### Dependencies

Key Python packages:
- `graphiti-core==0.11.6` - Knowledge graph processing
- `sentence-transformers>=2.7.0` - Text embeddings
- `pymilvus==2.3.4` - Vector database client
- `neo4j>=5.23.0` - Graph database client
- `fastapi>=0.115.0` - Web framework

## API Endpoints

### Core RAG Operations

- `POST /query` - Process queries with agentic reasoning
- `POST /document` - Add documents to the knowledge base
- `POST /upload-image` - Upload and process images

### Health & Monitoring

- `GET /health` - Health check endpoint
- `GET /ready` - Readiness probe
- `GET /metrics` - Prometheus metrics

## Example Usage

### Text Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key features of machine learning?",
    "user_id": "user123"
  }'
```

### Document Upload
```bash
curl -X POST "http://localhost:8000/document" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Machine learning is a subset of artificial intelligence...",
    "metadata": {"source": "ml_guide.pdf"}
  }'
```

## Development

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start services (Docker Compose):**
   ```bash
   docker-compose up -d
   ```

3. **Run the application:**
   ```bash
   python app/main.py
   ```

### Building Docker Image

The system supports multiple container runtimes:

```bash
# Using the build script (auto-detects runtime)
./scripts/build_and_push.sh

# Or manually with specific runtime
podman build -t rag-system .
docker build -t rag-system .
```

## Troubleshooting

### Common Issues

1. **Model Download Failures**
   - Ensure internet access during build
   - Use offline mode for air-gapped environments

2. **Database Connection Issues**
   - Check service DNS resolution
   - Verify network policies
   - Use IP addresses if DNS fails

3. **Memory Issues**
   - Reduce model size or increase resource limits
   - Monitor pod resource usage

### Logs

```bash
# Check application logs
kubectl logs -f deployment/rag-system -n rag

# Check all pods in namespace
kubectl logs -f -l app=rag-system -n rag
```

## Architecture Details

### Knowledge Graph (Graphiti)

The system uses Graphiti for building and querying knowledge graphs:
- Automatic entity extraction
- Relationship mapping
- Graph-based reasoning

### Vector Database (Milvus)

Milvus provides efficient similarity search:
- Text embeddings (384-dimensional)
- Image embeddings (CLIP model)
- Cosine similarity search

### Caching Strategy

Redis caching improves performance:
- Query result caching
- Model output caching
- Session management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Create a GitHub issue
- Check the troubleshooting guide
- Review the logs for error details