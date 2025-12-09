#!/bin/bash
# Script to clear HuggingFace cache for Qwen-VL-Chat to force use of local code

echo "Clearing HuggingFace cache for Qwen-VL-Chat..."
echo "This will force the model to use local modeling_qwen.py with the bug fix."

# Clear the cached model code
rm -rf ~/.cache/huggingface/modules/transformers_modules/_dot_
rm -rf ~/.cache/huggingface/modules/transformers_modules/Qwen

echo "Cache cleared!"
echo "Now run: python testqwen.py"
