# Intelligent Mental Health System - AI Manual 🧠

## Overview
This application is powered by a **Self-Correcting, Research-Grade AI** (Llama 3.2). It goes beyond standard chatbots by learning from your feedback and verifying its own answers.

## Key Features

### 1. Self-Correction Engine
- **What it does**: Before showing you an answer, the AI critiques itself using a trained "Reward Model".
- **Visual Indicator**: Look for the `(Verified by Kiro Self-Correction Engine)` badge on high-quality responses.
- **How to help**: Keep clicking "Thumbs Up" 👍 or "Thumbs Down" 👎. The system learns your preferences in real-time.

### 2. Active Learning (Uncertainty Detection)
- **What it does**: The AI knows when it is unsure (e.g., on controversial topics).
- **Visual Indicator**: Responses marked as **[UNCERTAIN]** or `[Requesting Feedback]`.
- **Action**: Your feedback on these specific messages is **10x more valuable** for training. Please rate them!

### 3. Advanced RAG (Search Optimization)
- **What it does**: If you type "tired", the AI secretly rewrites it to "causes of chronic fatigue" to find medical articles.
- **Result**: You get professional, evidence-based advice even from vague queries.

## Troubleshooting

### "AI is Thinking..." forever?
- Ensure **Ollama** is running: Open a terminal and type `ollama serve`.
- Ensure **Llama 3.2** is pulled: Type `ollama pull llama3.2`.

### "No Reward Model found"?
- The system works fine without it, but Self-Correction will be disabled.
- To train it, simply use the app and give feedback! The model trains in the background.

## Quick Start 🚀

- **Start App**: Double-click `start_kiro.bat`
- **Stop App**: Double-click `stop_kiro.bat`

## Cloud Deployment ☁️
This project is configured for cloud deployment (e.g., Render, Railway, Heroku).
- **Files Included**: `Procfile`, `runtime.txt`, `requirements.txt`.
- **Note**: For cloud use, ensure the cloud provider supports Ollama or switch to an external LLM API.

## Privacy Note
- All AI processing happens **LOCALLY** on your device (default mode).
- Your journal entries and queries **NEVER** leave your computer.
