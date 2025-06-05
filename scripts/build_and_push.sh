# scripts/build_and_push.sh
#!/bin/bash

# Build and push script for the RAG system

set -e

# Configuration
REGISTRY=${REGISTRY:-"docker.io/docker4zerocool"}
IMAGE_NAME=${IMAGE_NAME:-"rag-system"}
VERSION=${VERSION:-"latest"}
PLATFORM=${PLATFORM:-"linux/amd64,linux/arm64"}

echo "Building and pushing RAG System Docker image..."

# Function to build with Docker
build_with_docker() {
    echo "Attempting to build with Docker..."
    # Remove existing builder if it exists and create a new one
    docker buildx rm rag-builder 2>/dev/null || true
    docker buildx create --use --name rag-builder
    docker buildx build \
        --platform ${PLATFORM} \
        --tag ${REGISTRY}/${IMAGE_NAME}:${VERSION} \
        --tag ${REGISTRY}/${IMAGE_NAME}:latest \
        --push \
        --file Dockerfile \
        .
}

# Function to build with nerdctl (for k3s/containerd)
build_with_nerdctl() {
    echo "Attempting to build with nerdctl..."
    nerdctl build \
        --tag ${REGISTRY}/${IMAGE_NAME}:${VERSION} \
        --tag ${REGISTRY}/${IMAGE_NAME}:latest \
        --file Dockerfile \
        .
    nerdctl push ${REGISTRY}/${IMAGE_NAME}:${VERSION}
    nerdctl push ${REGISTRY}/${IMAGE_NAME}:latest
}

# Function to build with podman
build_with_podman() {
    echo "Attempting to build with podman..."
    podman build \
        --tag ${REGISTRY}/${IMAGE_NAME}:${VERSION} \
        --tag ${REGISTRY}/${IMAGE_NAME}:latest \
        --file Dockerfile \
        .
    podman push ${REGISTRY}/${IMAGE_NAME}:${VERSION}
    podman push ${REGISTRY}/${IMAGE_NAME}:latest
}

# Try different build methods
BUILD_SUCCESS=false

# Try Docker first
if command -v docker &> /dev/null; then
    if build_with_docker; then
        BUILD_SUCCESS=true
        echo "Successfully built with Docker!"
    else
        echo "Docker build failed, trying other methods..."
    fi
fi

# Try nerdctl if Docker failed
if ! $BUILD_SUCCESS && command -v nerdctl &> /dev/null; then
    if build_with_nerdctl; then
        BUILD_SUCCESS=true
        echo "Successfully built with nerdctl!"
    else
        echo "Nerdctl build failed, trying other methods..."
    fi
fi

# Try podman if both Docker and nerdctl failed
if ! $BUILD_SUCCESS && command -v podman &> /dev/null; then
    if build_with_podman; then
        BUILD_SUCCESS=true
        echo "Successfully built with podman!"
    else
        echo "Podman build failed..."
    fi
fi

if $BUILD_SUCCESS; then
    echo "Image built and pushed successfully!"
    echo "Image: ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
else
    echo "ERROR: All build methods failed. Please check your container runtime setup."
    echo "Available runtimes:"
    command -v docker &> /dev/null && echo "  - Docker: $(docker --version 2>/dev/null || echo 'Not working')"
    command -v nerdctl &> /dev/null && echo "  - nerdctl: $(nerdctl --version 2>/dev/null || echo 'Not working')"
    command -v podman &> /dev/null && echo "  - podman: $(podman --version 2>/dev/null || echo 'Not working')"
    exit 1
fi
