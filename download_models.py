#!/usr/bin/env python3
"""
Script to download and cache models for offline use in the RAG system.
This script is run during Docker build to ensure models are available offline.
"""

import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

def download_models():
    print('Starting model download...')
    
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # Download SentenceTransformer model
    print(f'Downloading SentenceTransformer: {model_name}')
    model = SentenceTransformer(model_name)
    print(f'SentenceTransformer downloaded successfully')
    
    # Also explicitly download the base transformers components
    print(f'Downloading Transformers components for: {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    
    print('All models downloaded and cached successfully')
    
    # Verify models can be loaded in offline mode
    print('Verifying offline access...')
    model_offline = SentenceTransformer(model_name, local_files_only=True)
    print('Offline verification successful!')
    
    # Show cache directories
    print(f'HF_HOME: {os.environ.get("HF_HOME", "Not set")}')
    print(f'TRANSFORMERS_CACHE: {os.environ.get("TRANSFORMERS_CACHE", "Not set")}')
    print(f'SENTENCE_TRANSFORMERS_HOME: {os.environ.get("SENTENCE_TRANSFORMERS_HOME", "Not set")}')

if __name__ == '__main__':
    download_models() 