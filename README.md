---
title: Golden RAG Evaluation Studio
emoji: 🧠
colorFrom: indigo
colorTo: cyan
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Multi-Hop RAG Evaluation for Deep Learning
---

# 🧠 Golden RAG Evaluation Studio

A production-quality RAG evaluation environment centered around deep learning content (3Blue1Brown's Neural Network series).

## Features
- **📊 Dataset Viewer**: Explore a curated set of multi-hop reasoning QA pairs.
- **🔍 Ask Questions**: Test the RAG engine with a grounded answer simulation.
- **📄 Transcript Explorer**: Browse timestamped transcripts with video metadata.

## Setup
1. Define your `GOOGLE_API_KEY` in HF Space secrets.
2. The system uses `sentence-transformers` for embeddings and `gemini-1.5-flash` (or newer) for answer synthesis.

## Tech Stack
- **UI**: Gradio
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **LLM**: Google Gemini API
- **Retrieval**: Cosine Similarity via Scikit-Learn
