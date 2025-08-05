# Model Storage Locations - Complete Inventory

## 📍 MODEL STORAGE SUMMARY

All downloaded models are stored in **3 main locations** on your system:

## 🏠 1. LOCAL PROJECT DIRECTORY
**Location**: `/Users/themanaspandey/Documents/GitHub/Akasha/models/`

### Gemma 3 27B Model (15.9 GB total)
```
📦 models/gemma-3-27b-it-qat-4bit/
├── model-00001-of-00004.safetensors (5.0 GB)
├── model-00002-of-00004.safetensors (5.0 GB) 
├── model-00003-of-00004.safetensors (4.9 GB)
├── model-00004-of-00004.safetensors (1.0 GB)
├── config.json
├── tokenizer_config.json
├── special_tokens_map.json
└── ... (11 more config files)
```

## 💾 2. HUGGING FACE CACHE
**Location**: `~/.cache/huggingface/hub/`

### Downloaded Models (6.0 GB total)
| Model | Size | Purpose |
|-------|------|---------|
| **openai/clip-vit-base-patch32** | 878 MB | Multimodal text/image embeddings |
| **sentence-transformers/all-MiniLM-L6-v2** | 87 MB | Fast text embeddings (384 dim) |
| **ResembleAI/chatterbox** | 3.0 GB | Voice processing model |
| **opendatalab/PDF-Extract-Kit-1.0** | 1.9 GB | Advanced PDF processing |
| **cross-encoder/ms-marco-MiniLM-L-2-v2** | 61 MB | Reranking model |
| **google/gemma-2-27b-it** | 4 KB | Model metadata |
| **google/gemma-2b-it** | 4 KB | Model metadata |

## 🔧 3. SPECIALIZED CACHES

### MLX Cache (Not Found)
**Location**: `~/.cache/mlx/`
- ❌ Empty (MLX uses models from local directory)

### Sentence Transformers Cache (Not Found)  
**Location**: `~/.cache/torch/sentence_transformers/`
- ❌ Empty (models stored in HuggingFace cache instead)

## 📊 TOTAL STORAGE USAGE

| Location | Storage Used | Key Models |
|----------|--------------|------------|
| **Local Models** | **15.9 GB** | Gemma 3 27B (4-bit quantized) |
| **HuggingFace Cache** | **6.0 GB** | CLIP, Sentence Transformers, PDF Kit |
| **Total** | **~22 GB** | All AI models combined |

## 🎯 MODEL ACCESS PATHS

### For Code References:
```python
# Gemma 3 27B (Local)
model_path = "models/gemma-3-27b-it-qat-4bit"

# CLIP (HuggingFace Cache - auto-loaded)
model_name = "openai/clip-vit-base-patch32"

# Sentence Transformers (HuggingFace Cache - auto-loaded)  
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# PDF Processing Kit (HuggingFace Cache - auto-loaded)
model_name = "opendatalab/PDF-Extract-Kit-1.0"
```

## 🧹 CACHE MANAGEMENT

### To Clear Caches (if needed):
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/

# Clear MLX cache  
rm -rf ~/.cache/mlx/

# Clear PyTorch cache
rm -rf ~/.cache/torch/
```

### To Preserve Models:
- **Keep Local Models**: The `models/` directory contains your main Gemma model
- **HuggingFace Auto-Download**: Other models will re-download automatically when needed
- **Estimated Re-download Time**: ~30-60 minutes for all models on good internet

## 🚀 CURRENT STATUS

✅ **All models successfully installed and accessible**
✅ **Total storage: ~22 GB (well within system capacity)**  
✅ **Models optimized for Apple Silicon M4 Pro**
✅ **Ready for production use**

## 🔍 MODEL VERIFICATION

All models have been tested and are working correctly:
- ✅ **Gemma 3 27B**: Loaded and generating text with MLX
- ✅ **CLIP ViT-B/32**: Generating 512-dim embeddings  
- ✅ **all-MiniLM-L6-v2**: Generating 384-dim embeddings
- ✅ **PDF-Extract-Kit**: Available for advanced PDF processing
- ✅ **Cross-encoder**: Ready for reranking tasks

Your system now has a complete AI model ecosystem ready for advanced RAG operations! 🎉