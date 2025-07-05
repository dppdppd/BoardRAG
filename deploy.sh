#!/bin/bash

# deploy.sh - Deploy changes to both GitHub and Hugging Face
# Usage: ./deploy.sh "commit message"

set -e  # Exit on any error

# Check if commit message is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 \"commit message\""
    echo "Example: $0 \"Add new feature\""
    exit 1
fi

COMMIT_MSG="$1"

echo "🔄 Deploying changes..."
echo "📝 Commit message: $COMMIT_MSG"
echo

# Add all changes
echo "📦 Adding changes..."
git add .

# Commit with provided message
echo "💾 Committing..."
git commit -m "$COMMIT_MSG"

# Push to GitHub (origin)
echo "🐙 Pushing to GitHub..."
git push origin main

# Push to Hugging Face Space
echo "🤗 Pushing to Hugging Face..."
git push huggingface main

echo "✅ Deploy complete!"
echo "🔗 GitHub: https://github.com/dppdppd/BoardRAG"
echo "🚀 Space: https://huggingface.co/spaces/mysterydough/BoardRAG" 