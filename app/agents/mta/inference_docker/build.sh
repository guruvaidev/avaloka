#!/bin/bash
# Build script for inference Docker image

set -e

# Configuration
IMAGE_NAME="${IMAGE_NAME:-avaloka-inference}"
# Use timestamp or version tag if provided, otherwise default to latest
if [ -z "$IMAGE_TAG" ]; then
    # Generate timestamp-based tag for versioning
    IMAGE_TAG="$(date +%Y%m%d-%H%M%S)-fixed"
fi
REGISTRY="${REGISTRY:-gcr.io}"
PROJECT_ID="${GCP_PROJECT_ID}"

# Full image name
FULL_IMAGE_NAME="${REGISTRY}/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building Docker image: ${FULL_IMAGE_NAME}"

# Build image
docker build -t "${FULL_IMAGE_NAME}" .

echo "✓ Image built successfully: ${FULL_IMAGE_NAME}"

# Also tag as latest for convenience
LATEST_IMAGE_NAME="${REGISTRY}/${PROJECT_ID}/${IMAGE_NAME}:latest"
docker tag "${FULL_IMAGE_NAME}" "${LATEST_IMAGE_NAME}"
echo "✓ Also tagged as: ${LATEST_IMAGE_NAME}"

# Ask if user wants to push
echo ""
read -p "Push image to GCR? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pushing image to registry..."
    docker push "${FULL_IMAGE_NAME}"
    docker push "${LATEST_IMAGE_NAME}"
    echo "✓ Image pushed successfully"
    echo ""
    echo "Image available at:"
    echo "  ${FULL_IMAGE_NAME}"
    echo "  ${LATEST_IMAGE_NAME}"
else
    echo ""
    echo "To push the image manually, run:"
    echo "  docker push ${FULL_IMAGE_NAME}"
    echo "  docker push ${LATEST_IMAGE_NAME}"
fi

