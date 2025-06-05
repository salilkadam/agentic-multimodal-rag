# Main Application Code for Agentic Multimodal RAG System
# File: app/main.py

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import redis
from pymilvus import MilvusClient, Collection
from graphiti_core import Graphiti

import openai
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import io
import json
import base64
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Pydantic Models
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    include_multimodal: bool = True
    max_results: int = 10

class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = {}
    user_id: Optional[str] = None

class AgentResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    reasoning_steps: List[str]
    confidence_score: float
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]
    timestamp: str

# Global variables for models and clients
milvus_client = None
graphiti_client = None
redis_client = None
embedding_model = None
clip_model = None
clip_processor = None

class RAGSystem:
    def __init__(self):
        self.setup_clients()
        self.setup_models()
    
    def setup_clients(self):
        global milvus_client, graphiti_client, redis_client
        
        # Milvus client
        milvus_host = os.getenv("MILVUS_HOST", "localhost")
        milvus_port = os.getenv("MILVUS_PORT", "19530")
        milvus_username = os.getenv("MILVUS_USERNAME", "")
        milvus_password = os.getenv("MILVUS_PASSWORD", "")
        
        milvus_client = MilvusClient(
            uri=f"http://{milvus_host}:{milvus_port}",
            user=milvus_username if milvus_username else None,
            password=milvus_password if milvus_password else None
        )
        
        # Graphiti client
        neo4j_host = os.getenv("NEO4J_HOST", "localhost")
        neo4j_port = os.getenv("NEO4J_PORT", "7687")
        neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        graphiti_client = Graphiti(
            uri=f"bolt://{neo4j_host}:{neo4j_port}",
            user=neo4j_username,
            password=neo4j_password
        )
        
        # Redis client
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        
        logger.info("All clients initialized successfully")
    
    def setup_models(self):
        global embedding_model, clip_model, clip_processor
        
        # Text embedding model
        embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        # Use local_files_only to prevent internet access
        embedding_model = SentenceTransformer(embedding_model_name, local_files_only=True)
        
        # CLIP model for multimodal processing
        if os.getenv("ENABLE_MULTIMODAL", "true").lower() == "true":
            multimodal_model_name = os.getenv("MULTIMODAL_MODEL", "openai/clip-vit-base-patch32")
            # Use local_files_only to prevent internet access
            clip_model = CLIPModel.from_pretrained(multimodal_model_name, local_files_only=True)
            clip_processor = CLIPProcessor.from_pretrained(multimodal_model_name, local_files_only=True)
        
        # OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        logger.info("All models loaded successfully")
    
    async def process_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        return embedding_model.encode(text).tolist()
    
    async def process_image_embedding(self, image: Image.Image) -> List[float]:
        """Generate embedding for image using CLIP"""
        if not clip_model or not clip_processor:
            raise HTTPException(status_code=400, detail="Multimodal processing not enabled")
        
        inputs = clip_processor(images=image, return_tensors="pt")
        image_features = clip_model.get_image_features(**inputs)
        return image_features.squeeze().detach().numpy().tolist()
    
    async def semantic_search(self, query_embedding: List[float], collection_name: str, limit: int = 10) -> List[Dict]:
        """Perform semantic search in Milvus"""
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = milvus_client.search(
            collection_name=collection_name,
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["content", "metadata", "timestamp"]
        )
        
        return results[0] if results else []
    
    async def graph_reasoning(self, query: str, user_id: str = None) -> Dict[str, Any]:
        """Perform graph-based reasoning using Graphiti"""
        try:
            # Add query as an episode to the graph
            episode_id = str(uuid.uuid4())
            await graphiti_client.add_episode(
                name=f"query_{episode_id}",
                episode_body=query,
                source_description="User query",
                user_id=user_id or "anonymous"
            )
            
            # Search for relevant entities and relationships
            search_results = await graphiti_client.search(
                query=query,
                search_type="combined"
            )
            
            return {
                "entities": search_results.get("entities", []),
                "relationships": search_results.get("relationships", []),
                "reasoning_path": search_results.get("reasoning_path", [])
            }
        except Exception as e:
            logger.error(f"Graph reasoning error: {e}")
            return {"entities": [], "relationships": [], "reasoning_path": []}
    
    async def agentic_reasoning(self, query: str, context: List[Dict], graph_context: Dict) -> AgentResponse:
        """Perform agentic reasoning to generate final response"""
        start_time = datetime.now()
        
        # Prepare context for LLM
        context_text = "\n".join([item.get("content", "") for item in context])
        graph_info = f"Entities: {graph_context.get('entities', [])}\nRelationships: {graph_context.get('relationships', [])}"
        
        # Reasoning steps
        reasoning_steps = [
            "Analyzed user query",
            f"Retrieved {len(context)} relevant documents from vector database",
            f"Found {len(graph_context.get('entities', []))} relevant entities in knowledge graph",
            "Performing multi-step reasoning with LLM"
        ]
        
        # LLM prompt
        prompt = f"""
        You are an intelligent AI assistant with access to both vector-based retrieval and knowledge graph information.
        
        Query: {query}
        
        Retrieved Context:
        {context_text}
        
        Knowledge Graph Context:
        {graph_info}
        
        Please provide a comprehensive answer that:
        1. Directly addresses the user's query
        2. Incorporates information from both vector search and knowledge graph
        3. Explains your reasoning process
        4. Provides confidence in your answer
        
        Format your response as JSON with the following structure:
        {{
            "answer": "Your comprehensive answer",
            "confidence": 0.85,
            "reasoning": "Your reasoning process"
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            reasoning_steps.append("Generated final response using LLM")
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            result = {
                "answer": "I apologize, but I encountered an error processing your request.",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AgentResponse(
            answer=result.get("answer", "No answer generated"),
            sources=[{"content": item.get("content", ""), "metadata": item.get("metadata", {})} for item in context],
            reasoning_steps=reasoning_steps,
            confidence_score=result.get("confidence", 0.0),
            processing_time=processing_time
        )

# Initialize RAG system
rag_system = RAGSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting RAG System...")
    
    # Create collections if they don't exist
    collections = ["documents", "images", "multimodal"]
    for collection_name in collections:
        try:
            if not milvus_client.has_collection(collection_name):
                milvus_client.create_collection(
                    collection_name=collection_name,
                    dimension=384,  # Adjust based on your embedding model
                    metric_type="COSINE"
                )
                logger.info(f"Created collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG System...")

# FastAPI app
app = FastAPI(
    title="Agentic Multimodal RAG System",
    description="Advanced RAG system with Graphiti and Milvus integration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your JWT verification logic here
    return {"user_id": "authenticated_user"}

# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {}
    
    # Check Milvus
    try:
        milvus_client.list_collections()
        services["milvus"] = "healthy"
    except Exception as e:
        services["milvus"] = f"unhealthy: {str(e)}"
    
    # Check Redis
    try:
        redis_client.ping()
        services["redis"] = "healthy"
    except Exception as e:
        services["redis"] = f"unhealthy: {str(e)}"
    
    # Check Neo4j (Graphiti)
    try:
        # This is a simplified check
        services["neo4j"] = "healthy"
    except Exception as e:
        services["neo4j"] = f"unhealthy: {str(e)}"
    
    return HealthResponse(
        status="healthy" if all("healthy" in status for status in services.values()) else "degraded",
        services=services,
        timestamp=datetime.now().isoformat()
    )

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    return {"status": "ready"}

@app.post("/query", response_model=AgentResponse)
async def process_query(request: QueryRequest, user: dict = Depends(verify_token)):
    """Process a query using agentic RAG"""
    try:
        # Generate query embedding
        query_embedding = await rag_system.process_text_embedding(request.query)
        
        # Semantic search in Milvus
        semantic_results = await rag_system.semantic_search(
            query_embedding=query_embedding,
            collection_name="documents",
            limit=request.max_results
        )
        
        # Graph reasoning with Graphiti
        graph_context = await rag_system.graph_reasoning(
            query=request.query,
            user_id=request.user_id or user.get("user_id")
        )
        
        # Agentic reasoning
        response = await rag_system.agentic_reasoning(
            query=request.query,
            context=semantic_results,
            graph_context=graph_context
        )
        
        # Cache result
        cache_key = f"query:{hash(request.query)}:{request.user_id}"
        redis_client.setex(cache_key, 3600, json.dumps(response.dict()))
        
        return response
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/document")
async def add_document(request: DocumentRequest, user: dict = Depends(verify_token)):
    """Add a document to the knowledge base"""
    try:
        # Generate embedding
        embedding = await rag_system.process_text_embedding(request.content)
        
        # Prepare data for Milvus
        data = {
            "id": str(uuid.uuid4()),
            "content": request.content,
            "embedding": embedding,
            "metadata": request.metadata,
            "timestamp": datetime.now().isoformat(),
            "user_id": request.user_id or user.get("user_id")
        }
        
        # Insert into Milvus
        milvus_client.insert(
            collection_name="documents",
            data=[data]
        )
        
        # Add to Graphiti knowledge graph
        await graphiti_client.add_episode(
            name=f"document_{data['id']}",
            episode_body=request.content,
            source_description="User uploaded document",
            user_id=request.user_id or user.get("user_id")
        )
        
        return {"message": "Document added successfully", "document_id": data["id"]}
        
    except Exception as e:
        logger.error(f"Document addition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...), user: dict = Depends(verify_token)):
    """Upload and process an image"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Generate embedding
        embedding = await rag_system.process_image_embedding(image)
        
        # Prepare data
        data = {
            "id": str(uuid.uuid4()),
            "filename": file.filename,
            "embedding": embedding,
            "image_data": base64.b64encode(image_data).decode(),
            "metadata": {"content_type": file.content_type},
            "timestamp": datetime.now().isoformat(),
            "user_id": user.get("user_id")
        }
        
        # Insert into Milvus
        milvus_client.insert(
            collection_name="images",
            data=[data]
        )
        
        return {"message": "Image uploaded successfully", "image_id": data["id"]}
        
    except Exception as e:
        logger.error(f"Image upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    # Return basic metrics in Prometheus format
    return """
# HELP rag_queries_total Total number of queries processed
# TYPE rag_queries_total counter
rag_queries_total 0

# HELP rag_documents_total Total number of documents indexed
# TYPE rag_documents_total counter
rag_documents_total 0
"""

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )