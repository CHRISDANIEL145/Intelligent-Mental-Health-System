# Cloud Deployment Guide ☁️

This project is configured for PaaS deployment (Render, Heroku, Railway).

## Prerequisites
- A GitHub repository containing this project.
- An account on [Render](https://render.com) or [Heroku](https://heroku.com).
- Note: Ollama (LLM) is difficult to run on free tier cloud. **For Cloud**, you should ideally switch `LLMInterface` to use an external API (like OpenAI or Groq) unless you have a GPU instance.

## Deploying to Render (Recommended)

1.  **New Web Service**: Connect your GitHub repo.
2.  **Runtime**: Python 3.
3.  **Build Command**: `pip install -r requirements.txt`.
4.  **Start Command**: `gunicorn app:app`.
5.  **Environment Variables**:
    - `PYTHON_VERSION`: `3.10.12`
    - `SECRET_KEY`: `(generate a random string)`

## Deploying to Heroku

1.  **Install CLI**: `heroku login`.
2.  **Create App**: `heroku create my-kiro-app`.
3.  **Deploy**: `git push heroku main`.

## Important: Database & Persistence
- The default setup uses **SQLite/In-Memory** fallback if MongoDB is missing.
- For production, set `MONGO_URI` env var to a real MongoDB Atlas connection string.
