#!/bin/bash

# Generate static blog. Copy the content to .deploy/vontasa.github.io
# Push to github
# github credential may be needed during the process

echo "Deploy to master branch"
hexo generate
hexo deploy

echo "Update branches"

