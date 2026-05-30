# Text-to-Image Generation Challenge

This project generates images from text prompts using Stable Diffusion XL, evaluated via a combined score of YOLO object detection (F1) and CLIP semantic similarity.

---

## How It Works

```
Text prompt
    ↓
spaCy POS tagging → expected nouns (YOLO targets)
    ↓
SDXL generates image (1024×1024px)
    ↓
YOLOv8n detects objects → CLIP aligns detected to expected nouns
    ↓
F1 = semantic match (expected vs detected)
    ↓
final_score = 0.5 × CLIP score + 0.5 × F1
```

> **Note:** The evaluation uses semantic matching via CLIP embeddings — not exact class name matching. A detected "bicycle" can match an expected "bike" if their CLIP representations are similar enough (threshold: cosine similarity > 0.25).

---

## Project Structure

```
text_to_image/
├── data/
│   ├── DreamLayer-Prompt-Kaggle.txt   # 50 public prompts
│   └── prompts.csv                    # generated after Part 1
├── notebooks/
│   ├── DreamLayer_Generate.ipynb      # Part 1 — image generation
│   └── DreamLayer_Part2_Evaluate      # Part 2 — evaluation & scoring
└── output/
|   ├── 0001.png … 0050.png            # generated images
└── evaluate/
    ├── clip_results.csv
    ├── yolo_clip_results.csv
    ├── final_results.csv
    ├── clip_scores.png
    └── semantic_f1.png
```

> Part 1 and Part 2 are separate notebooks — SDXL and YOLO+CLIP together exceed available RAM on Colab free tier. Run Part 1 first, then close it before opening Part 2.

---

## Part 1 — Image Generation

### Setup

```bash
# Run Cell 1, then: Runtime → Restart session → Ctrl+F9
```

Compatible version pinset for Colab Python 3.12:

```
numpy>=2.0.0,<3.0      # Colab ships numpy 2.x — do not downgrade
scipy>=1.14.0           # first build compiled against numpy 2.x
huggingface_hub>=0.24.0,<1.0
diffusers==0.32.2
transformers==4.46.3
accelerate==0.34.2
safetensors>=0.4.3
pandas>=2.2.0
Pillow>=10.0,<13.0
```

### Model

**Stable Diffusion XL Base 1.0** — `stabilityai/stable-diffusion-xl-base-1.0`

No Hugging Face token required.

| Parameter | Value | Reason |
|---|---|---|
| Scheduler | DPM-Solver++ (Karras) | 30 steps equivalent to 1000 DDPM steps |
| Guidance scale | 10.0 | 42% of prompts have 4+ objects — high guidance forces all into frame |
| Resolution | 1024 × 1024 px | Native SDXL resolution, sufficient for YOLO detection |
| Steps | 30 | Balance of quality and speed (~60s per image on T4) |
| dtype | float16 | Halves VRAM: ~14GB → ~7GB |
| Seed | 42 | Reproducibility |

### Memory optimizations (T4 15GB)

```python
pipe.enable_model_cpu_offload()   # moves layers between CPU and GPU on demand
pipe.enable_attention_slicing(1)  # splits attention into chunks
pipe.enable_vae_slicing()         # decodes image in horizontal strips
pipe.enable_vae_tiling()          # tiles large latents
```

### Prompt engineering

Every prompt gets a quality suffix before generation:

```
photorealistic, sharp focus, clear objects, well-lit, uncluttered composition, DSLR photograph
```

Negative prompt:

```
blurry, abstract, painting, sketch, cartoon, anime, distorted, deformed, watermark, text, lowres, worst quality
```

Both are chosen to maximize YOLO detectability — the detector performs best on realistic, well-lit photographs.

### Generation features

- **Resume support** — skips already-generated images if the session is interrupted
- **OOM fallback** — automatically retries at 768px if 1024px causes out-of-memory error
- **Periodic cleanup** — `torch.cuda.empty_cache()` every 5 images to prevent memory fragmentation

---

## Part 2 — Evaluation

### Setup

```
numpy>=2.0.0,<3.0
scipy>=1.14.0
ultralytics>=8.2,<9.0
spacy>=3.7,<4.0
open-clip-torch
pandas>=2.2.0
Pillow>=10.0,<13.0
```

### Pipeline

**Step 1 — CLIP evaluation**

For each image, computes cosine similarity between the CLIP image embedding and the CLIP text embedding of the original prompt. Uses `ViT-B/32` from OpenAI.

```python
score = (image_features @ text_features.T).item()
```

Saves results to `evaluate/clip_results.csv` and `evaluate/clip_scores.png`.

**Step 2 — YOLO + semantic F1**

1. YOLOv8n detects objects in each image (confidence threshold: 0.25)
2. spaCy extracts expected nouns from the prompt (POS tags: NOUN, PROPN)
3. CLIP semantic matching — for each expected noun, finds the best-matching detected object using cosine similarity (threshold: > 0.25)
4. Computes precision, recall, F1

```python
# Semantic matching — not exact string comparison
sim = exp_feat @ det_feat.T   # CLIP similarity matrix
for each expected noun:
    find best-matching detected object
    if similarity > 0.25: count as true positive
```

Saves results to `evaluate/yolo_clip_results.csv` and `evaluate/semantic_f1.png`.

**Step 3 — Final score**

```python
final_score = 0.5 * clip_score + 0.5 * f1
```

Saves to `evaluate/final_results.csv`.

---

## Results

| Metric | Value |
|---|---|
| Mean CLIP score | ~0.33 |
| Mean F1 (semantic) | ~0.50 |
| **Final score** | **0.4198** |
| Images generated | 50 / 50 |

---

## Data Analysis

Key findings from the 50 public prompts that informed model configuration:

| Finding | Value | Impact on config |
|---|---|---|
| Avg prompt length | 10.7 words | Confirms 1024px sufficient |
| Avg objects per prompt | 3.5 | Justifies guidance_scale=10 |
| Prompts with 4+ objects | 21/50 (42%) | Justifies 30 steps |
| Top category | people (24%) | Justifies photorealistic suffix |
| YOLO-detectable nouns | ~20% | Explains F1 ceiling |

---

## Requirements

- Python 3.12
- Google Colab (T4 GPU, 15GB VRAM)
- Google Drive (for storing images between Part 1 and Part 2)
