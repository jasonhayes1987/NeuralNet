#!/bin/bash

REPO_PATH="."  # Change this if your repo is in another directory

echo "# Project Codebase" > repo_code.md

for f in $(find $REPO_PATH -type f -name "*.py" -o -name "*.md" -o -name "*.json" -o -name "*.yaml" -o -name "*.sh"); do
    echo -e "\n\n## File: ${f#$REPO_PATH/}\n\`\`\`${f##*.}" >> repo_code.md
    cat "$f" >> repo_code.md
    echo -e "\n\`\`\`\n" >> repo_code.md
done

echo "Markdown file generated: repo_code.md"
