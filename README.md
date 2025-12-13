# DC Well Being AI 🧠
### Advanced Mental Health assessment & Support System

**DC Well Being AI** is a comprehensive mental health platform that combines **Traditional Machine Learning** with **Generative AI** to provide accurate assessments, personalized insights, and empathetic support.

---

## 🌟 Key Features

### 1. Multi-Dimensional Assessment
- **Sleep Quality**: Analyzes screen time and stress to predict sleep issues.
- **Burnout Risk**: Evaluates work conditions for healthcare/professional burnout.
- **Depression & Anxiety**: Screenings based on clinical factors.

### 2. Intelligent AI Companion
- Powered by **Llama 3.2** (via Ollama).
- **RAG (Retrieval-Augmented Generation)**: Answers are grounded in verified mental health documents (`ChromaDB`).
- **Empathetic & Direct**: Fine-tuned system prompts ensure helpful, non-robotic responses.

### 3. Personal Growth Tools
- **Mood Journal**: Track daily emotions with streak counters.
- **Profile Analysis**: Long-term tracking of risk levels.
- **PDF Reports**: Export comprehensive health summaries for professional review.

---

## 🛠️ Tech Stack

- **Frontend**: HTML5, Tailwind CSS (Dark Mode UI)
- **Backend**: Flask (Python)
- **AI/ML**: 
  - `scikit-learn` (Random Forest Models)
  - `Ollama` (Llama 3.2 LLM)
  - `ChromaDB` (Vector Search)
- **Database**: 
  - `MongoDB` (Primary)
  - `JSON` (Persistent Local Fallback)

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running (`ollama serve`).

### 1. Clone & Install
```bash
git clone https://github.com/your-repo/dc-well-being.git
cd dc-well-being
pip install -r requirements.txt
```

### 2. Setup AI
Ensure Ollama is running and pull the model:
```bash
ollama serve
ollama pull llama3.2
```

### 3. Run Application
Use the startup script to launch everything:
```bash
start_dc_wellbeing.bat
```
Visit `http://localhost:5000` in your browser.

---

## 📊 Methodology
For a detailed breakdown of the machine learning pipeline, system architecture, and diagrammatic flows, please refer to [PROJECT_REPORT.md](./PROJECT_REPORT.md).

---

## 🛡️ Privacy & Disclaimer
*This system is for educational and screening purposes only. It does not replace professional medical diagnosis or treatment.*
All data is stored locally (or in your local MongoDB instance) to ensure privacy.
