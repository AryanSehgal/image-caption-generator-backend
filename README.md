# Image Caption Generator — Backend

A Flask REST API that generates natural-language captions for uploaded images using a CNN + LSTM deep learning pipeline (ResNet50 for image feature extraction, a custom-trained LSTM decoder for language generation).

**Live API:** https://image-caption-generator-backend.onrender.com
**Live App (frontend):** https://image-caption-generator-frontend.vercel.app/

## Related Repositories

| Repo | Description |
|---|---|
| [image-caption-generator-frontend](https://github.com/AryanSehgal/image-caption-generator-frontend) | React + TypeScript UI that consumes this API |
| [image-captioning-case-study](https://github.com/AryanSehgal/image-captioning-case-study) | Jupyter notebook used to train the caption model from scratch on the Flickr8k dataset |

## Overview

This service exposes a single `/caption` endpoint. A client uploads an image, the server stores it via Cloudinary, extracts a 2048-dimension feature vector from it using a pretrained ResNet50, and feeds that vector through a custom-trained LSTM-based decoder model to generate a caption word-by-word.

## Tech Stack

- **Framework:** Flask 3, served in production via Gunicorn
- **ML:** TensorFlow / Keras (`tensorflow-cpu`), ResNet50 (ImageNet weights), a custom encoder-decoder captioning model
- **Image hosting:** Cloudinary (temporary storage for uploaded images, returns a public URL)
- **Deployment:** Docker container on Render (free tier, CPU-only)

## API

### `GET /`
Health check. Returns:
```json
{ "status": "Image captioning API is running" }
```

### `POST /caption`
Accepts a `multipart/form-data` request with a single file field named `userfile`.

**Response:**
```json
{
  "image": "https://res.cloudinary.com/.../uploaded-image.jpg",
  "caption": "a dog running through the grass"
}
```

**Error response:**
```json
{ "error": "description of what went wrong" }
```

## Pipeline

```
Client uploads image
        │
        ▼
Uploaded to Cloudinary (returns public URL)
        │
        ▼
Image downloaded and preprocessed
  (resized to 224x224, converted to RGB)
        │
        ▼
ResNet50 (ImageNet weights, pretrained — not trained by us)
  extracts a 2048-dim feature vector
        │
        ▼
Custom LSTM decoder model (model_9.h5)
  generates caption one word at a time,
  greedy decoding, using a ~1,848-word vocabulary
        │
        ▼
JSON response returned to client
  { image, caption }
```

The captioning model itself (`weights/model_9.h5`) was trained separately — see the [image-captioning-case-study](https://github.com/AryanSehgal/image-captioning-case-study) repo for the full training notebook and methodology.

## Project Structure

```
├── caption_gen.py       # Loads models, preprocesses images, generates captions
├── wsgi.py               # Flask app: routes, Cloudinary upload handling
├── requirements.txt      # Python dependencies
├── Dockerfile             # Container build + gunicorn start command
├── weights/
│   └── model_9.h5         # Trained caption-generation model (epoch 9 checkpoint)
└── storage/
    ├── word_to_idx.pkl    # Vocabulary: word → index mapping
    └── idx_to_word.pkl    # Vocabulary: index → word mapping
```

## Running Locally

```bash
git clone https://github.com/AryanSehgal/image-caption-generator-backend.git
cd image-caption-generator-backend
pip install -r requirements.txt
```

Set the following environment variables (a free [Cloudinary](https://cloudinary.com) account provides these):
```
CLOUD_NAME=<your_cloudinary_cloud_name>
API_KEY=<your_cloudinary_api_key>
API_SECRET=<your_cloudinary_api_secret>
```

Run:
```bash
export FLASK_APP=wsgi
flask run
```

## Deployment

Deployed as a Docker container on [Render](https://render.com) (free tier). Key production considerations baked into this setup:

- **Single worker, single thread** (`gunicorn --workers 1 --threads 1`) to minimize peak memory — TensorFlow + ResNet50 + the caption model together are memory-heavy, and Render's free tier caps at 512MB RAM.
- **`tensorflow-cpu`** instead of full `tensorflow`, since there's no GPU available on the free tier and the CPU-only package has a smaller footprint.
- **Explicit garbage collection** after loading ResNet50, to release the unused full classifier head (`model_temp`) once only its trimmed feature-extraction layers (`model_resnet`) are needed.
- **CORS enabled** (`flask-cors`) so the separately-hosted frontend (on Vercel) can call this API cross-origin.
- Images are uploaded to **Cloudinary** rather than saved to local disk, since Render's filesystem is ephemeral.

## Known Limitations

The caption model was trained on the **Flickr8k dataset** — roughly 8,000 images, each with 5 human-written captions, overwhelmingly featuring **people, dogs, and everyday outdoor scenes**. Two structural factors follow directly from this:

1. **Narrow visual domain.** The model has effectively never seen categories like butterflies, vehicles, food close-ups, buildings, etc., and will produce a caption based on the closest pattern it does know — which can be confidently wrong for out-of-domain images.
2. **Small vocabulary.** Words appearing fewer than 10 times across the training captions were excluded entirely, leaving a vocabulary of **1,848 words** (including start/end tokens). The model is structurally incapable of outputting any word outside this set.

As a result, **captioning accuracy is best for images of people, dogs, and outdoor/everyday scenes** — this is a characteristic of the training data, not a bug in the deployed pipeline. See the [training repo](https://github.com/AryanSehgal/image-captioning-case-study) for the full data preparation and vocabulary-building process.
