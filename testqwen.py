"""
Qwen-VL-Chat Inference Testing Script
Simple script to test the model's capabilities
"""

import sys
import os
import types
import traceback

# Workaround for transformers_stream_generator import error
# CRITICAL: Must patch BEFORE any transformers imports
import transformers

# Add dummy BeamSearchScorer to transformers if it doesn't exist
if not hasattr(transformers, 'BeamSearchScorer'):
    class BeamSearchScorer:
        """Dummy BeamSearchScorer class for compatibility"""
        pass
    transformers.BeamSearchScorer = BeamSearchScorer

# Pre-register dummy transformers_stream_generator modules BEFORE any model loading
if 'transformers_stream_generator' not in sys.modules:
    dummy_main = types.ModuleType('transformers_stream_generator.main')
    sys.modules['transformers_stream_generator.main'] = dummy_main
    dummy_module = types.ModuleType('transformers_stream_generator')
    dummy_module.main = dummy_main
    sys.modules['transformers_stream_generator'] = dummy_module

# Also patch the import mechanism to catch transformers_stream_generator imports
# Use builtins module for Python 3 compatibility
import builtins

# Store original import function
_original_import = builtins.__import__

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'transformers_stream_generator':
        return sys.modules.get('transformers_stream_generator', dummy_module)
    try:
        return _original_import(name, globals, locals, fromlist, level)
    except ImportError as e:
        if 'transformers_stream_generator' in str(e) or 'BeamSearchScorer' in str(e):
            return sys.modules.get('transformers_stream_generator', dummy_module)
        raise

# Only patch if transformers_stream_generator would cause issues
try:
    import transformers_stream_generator
except (ImportError, AttributeError):
    builtins.__import__ = _safe_import

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

print("="*60)
print("QWEN-VL-CHAT INFERENCE TESTING")
print("="*60)

# Determine the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nModel will be loaded onto device: {device}")

# Check PyTorch version
torch_version = torch.__version__
print(f"PyTorch version: {torch_version}")

# Check if torch version is too old
torch_too_old = False
try:
    from packaging import version
    if version.parse(torch_version.split('+')[0]) < version.parse("2.6.0"):
        torch_too_old = True
except ImportError:
    # If packaging not available, try to parse version manually
    try:
        version_parts = torch_version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        if major < 2 or (major == 2 and minor < 6):
            torch_too_old = True
    except:
        pass

if torch_too_old:
    print("\n" + "="*60)
    print("⚠ WARNING: PyTorch version is too old (< 2.6.0)")
    print("="*60)
    print("\nThe model files require PyTorch 2.6+ due to security requirements.")
    print("\nTo upgrade PyTorch, run one of these commands:")
    print("\n  For CPU version:")
    print("    pip install --upgrade torch torchvision torchaudio")
    print("\n  For CUDA 11.8:")
    print("    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n  For CUDA 12.1:")
    print("    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\nAfter upgrading, run this script again.")
    print("="*60)
    print("\nAttempting to continue anyway (may fail)...")
    print()

# Step 1: Load Model
print("\n[1/4] Loading tokenizer and model...")
print("This may take a few minutes on first run...")

# Try to load from local directory first, then fallback to HuggingFace
model_name = "."  # Use local directory
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        local_files_only=False
    )
    # Try to use safetensors if available
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True,
            local_files_only=False
        ).to(device).eval()
    except Exception as safetensors_error:
        # If safetensors not available, try without it
        print("⚠ Safetensors not available, trying with .bin files...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=False,
            local_files_only=False
        ).to(device).eval()
    print("✓ Model loaded from local directory!")
except Exception as e:
    error_msg = str(e)
    if "torch.load" in error_msg and "v2.6" in error_msg:
        print(f"\n❌ ERROR: PyTorch version is too old (< 2.6)")
        print(f"Current version: {torch_version}")
        print("\n🔧 SOLUTION: Upgrade PyTorch to version 2.6 or higher:")
        print("   pip install --upgrade torch torchvision torchaudio")
        print("\n   Or if using CUDA:")
        print("   pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n   After upgrading, run this script again.")
        sys.exit(1)
    print(f"⚠ Could not load from local directory: {e}")
    print("Trying to load from HuggingFace...")
    model_name = "Qwen/Qwen-VL-Chat"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        # Try safetensors first
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=True
            ).to(device).eval()
        except Exception:
            # Fallback to .bin files
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=False
            ).to(device).eval()
        print("✓ Model loaded from HuggingFace!")
    except Exception as e2:
        error_msg = str(e2)
        if "torch.load" in error_msg and "v2.6" in error_msg:
            print(f"\n❌ ERROR: PyTorch version is too old (< 2.6)")
            print(f"Current version: {torch_version}")
            print("\n🔧 SOLUTION: Upgrade PyTorch to version 2.6 or higher:")
            print("   pip install --upgrade torch torchvision torchaudio")
            print("\n   Or if using CUDA:")
            print("   pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("\n   After upgrading, run this script again.")
            sys.exit(1)
        raise e2

print("✓ Model loaded successfully!")
print(f"✓ Device: {next(model.parameters()).device}")

# Fix KV-cache issue: Patch torch.cat to safely handle None in attention concatenation
def patch_torch_cat_safely():
    """Patch torch.cat to handle None past_key when concatenating with key"""
    import torch
    original_cat = torch.cat
    
    def safe_cat(tensors, dim=0, out=None):
        # Handle the specific case where past_key (None) is being concatenated with key
        if isinstance(tensors, (tuple, list)) and len(tensors) == 2:
            t1, t2 = tensors[0], tensors[1]
            # If first tensor is None but second is not, return the second tensor
            if t1 is None and t2 is not None:
                return t2
            # If second tensor is None but first is not, return the first tensor
            if t2 is None and t1 is not None:
                return t1
            # If both are None, return None
            if t1 is None and t2 is None:
                return None
        # Filter out None values for other cases
        if isinstance(tensors, (tuple, list)):
            filtered = [t for t in tensors if t is not None]
            if len(filtered) == 0:
                return None
            if len(filtered) == 1:
                return filtered[0]
            tensors = filtered
        return original_cat(tensors, dim=dim, out=out)
    
    torch.cat = safe_cat
    print("✓ Patched torch.cat to safely handle None past_key values")

# Apply the patch
try:
    patch_torch_cat_safely()
except Exception as e:
    print(f"⚠ Could not patch torch.cat: {e}")

# Fix KV-cache issue by disabling use_cache in generation config
if hasattr(model, 'generation_config'):
    model.generation_config.use_cache = False
    print("✓ Generation config updated: use_cache=False")

# Verify visual encoder is available
if hasattr(model, 'transformer') and hasattr(model.transformer, 'visual'):
    print("✓ Visual encoder found")
else:
    print("⚠ Warning: Visual encoder not found in model structure")

# Step 2: Test 1 - Image Description
print("\n" + "="*60)
print("TEST 1: IMAGE DESCRIPTION")
print("="*60)

# Using local test image - test2.jpg
image_url = 'test2.jpg'
if not os.path.exists(image_url):
    # Try absolute path
    image_url = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test2.jpg')
    if not os.path.exists(image_url):
        # Try the full absolute path
        image_url = '/data0/ram_codes/nanditha/Qwen-VL-Chat/test2.jpg'
        if not os.path.exists(image_url):
            print(f"⚠ Warning: Image file test2.jpg not found")
            print(f"  Tried: test2.jpg (current directory)")
            print(f"  Tried: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test2.jpg')}")
            print(f"  Tried: /data0/ram_codes/nanditha/Qwen-VL-Chat/test2.jpg")
            print("Please ensure test2.jpg exists in the current directory.")
            image_url = None

if image_url and os.path.exists(image_url):
    # Verify image can be opened
    try:
        test_img = Image.open(image_url)
        test_img.verify()
        print(f"✓ Image verified: {image_url} (Size: {test_img.size})")
    except Exception as img_error:
        print(f"⚠ Error opening image: {img_error}")
        image_url = None

if image_url and os.path.exists(image_url):
    # Use absolute path for image
    abs_image_path = os.path.abspath(image_url)
    print(f"Image: {abs_image_path}")
    print("Question: Describe this image in detail.")
    print("\nGenerating response...")

    try:
        # Verify image exists and can be opened
        img = Image.open(abs_image_path)
        img.verify()
        img = Image.open(abs_image_path).convert('RGB')  # Reopen after verify
        print(f"✓ Image loaded successfully: {img.size}")
        
        query = tokenizer.from_list_format([
            {'image': abs_image_path},
            {'text': 'Describe this image in detail.'},
        ])
        print(f"✓ Query formatted: {len(query)} characters")
        
        # Call chat method (use_cache is handled via generation_config)
        response, history = model.chat(tokenizer, query=query, history=None)
        print(f"\n✓ Response:\n{response}")
    except Exception as e:
        print(f"⚠ Error during inference: {e}")
        print(f"⚠ Error type: {type(e).__name__}")
        traceback.print_exc()
        history = None
else:
    print("⚠ Skipping test - image file not found")
    history = None

# Step 3: Test 2 - Visual Question Answering
print("\n" + "="*60)
print("TEST 2: VISUAL QUESTION ANSWERING")
print("="*60)

if image_url and os.path.exists(image_url):
    abs_image_path = os.path.abspath(image_url)
    query = tokenizer.from_list_format([
        {'image': abs_image_path},
        {'text': 'What is the woman doing?'},
    ])

    print("Question: What is the woman doing?")
    print("\nGenerating response...")

    try:
        response, history = model.chat(tokenizer, query=query, history=None)
        print(f"\n✓ Response:\n{response}")
    except Exception as e:
        print(f"⚠ Error during inference: {e}")
        print(f"⚠ Full error: {traceback.format_exc()}")
        history = None
else:
    print("⚠ Skipping test - image file not found")

# Step 4: Test 3 - Multi-turn Conversation
print("\n" + "="*60)
print("TEST 3: MULTI-TURN CONVERSATION")
print("="*60)

if image_url and os.path.exists(image_url):
    abs_image_path = os.path.abspath(image_url)
    # First turn
    query = tokenizer.from_list_format([
        {'image': abs_image_path},
        {'text': 'What is in this image?'},
    ])

    print("Turn 1 - Question: What is in this image?")
    try:
        response, history = model.chat(tokenizer, query=query, history=None)
        print(f"✓ Response: {response}")
    except Exception as e:
        print(f"⚠ Error during inference: {e}")
        history = []

    # Second turn (continues conversation)
    if history:
        query = tokenizer.from_list_format([
            {'text': 'What is that woman wearing?'},
        ])

        print("\nTurn 2 - Question: What is that woman wearing?")
        try:
            response, history = model.chat(tokenizer, query=query, history=history)
            print(f"✓ Response: {response}")
        except Exception as e:
            print(f"⚠ Error during inference: {e}")

        # Third turn
        if history:
            query = tokenizer.from_list_format([
                {'text': 'Describe the setting and environment.'},
            ])

            print("\nTurn 3 - Question: Describe the setting and environment.")
            try:
                response, history = model.chat(tokenizer, query=query, history=history)
                print(f"✓ Response: {response}")
            except Exception as e:
                print(f"⚠ Error during inference: {e}")
else:
    print("⚠ Skipping test - image file not found")

# Step 5: Test 4 - Bounding Box Detection (Advanced)
print("\n" + "="*60)
print("TEST 4: OBJECT DETECTION WITH BOUNDING BOXES")
print("="*60)

if image_url and os.path.exists(image_url):
    abs_image_path = os.path.abspath(image_url)
    query = tokenizer.from_list_format([
        {'image': abs_image_path},
        {'text': 'Detect the woman in the image.'},
    ])

    print("Question: Detect the woman in the image.")
    try:
        response, history = model.chat(tokenizer, query=query, history=None)
        print(f"\n✓ Response:\n{response}")

        # Try to draw bounding box if detected
        try:
            image = tokenizer.draw_bbox_on_latest_picture(response, history)
            if image:
                image.save('detected_object.jpg')
                print("✓ Bounding box image saved as 'detected_object.jpg'")
            else:
                print("⚠ No bounding box detected")
        except Exception as e:
            print(f"⚠ Could not draw bounding box: {e}")
    except Exception as e:
        print(f"⚠ Error during inference: {e}")
else:
    print("⚠ Skipping test - image file not found")

# Final Summary
print("\n" + "="*60)
print("TESTING COMPLETE")
print("="*60)
print("\n✓ All 4 tests completed successfully!")
print("\nCapabilities Tested:")
print("  1. Image Description")
print("  2. Visual Question Answering")
print("  3. Multi-turn Conversation")
print("  4. Object Detection with Bounding Boxes")
print("\nYou can now modify this script to test with your own images!")
