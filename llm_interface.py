
import requests
import json
import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

class LLMInterface:
    def __init__(self):
        self.available = self.check_availability()
        self.reward_model = self.load_reward_model()
        # We need an encoder here for the reward model if it's not passed in
        # Ideally we share the one from RAG engine, but for simplicity/robustness loading a small one here or passing it in generate
        try:
             self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except: self.encoder = None

        if self.available:
            print(f"✓ Connected to Ollama ({MODEL_NAME})")
        else:
            print("⚠ Ollama not detected. Using Logic-Based Fallback.")

    def load_reward_model(self):
        try:
            data = joblib.load("model/rlhf_model.joblib")
            print("✓ Loaded RLHF Models (Reward + VPL)")
            return data
        except:
            print("⚠ No RLHF Models found. Self-Correction disabled.")
            return None

    def check_availability(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                if any(MODEL_NAME in m for m in models):
                    return True
                print(f"⚠ Ollama running but '{MODEL_NAME}' not found. Please run 'ollama pull {MODEL_NAME}'")
            return False
        except:
            return False

    def score_response(self, query, response_text):
        """Calculate Reward Score & Uncertainty using VPL"""
        if not self.reward_model or not self.encoder:
            return 0.5, 0.0 # Neutral score, no uncertainty
        
        try:
            full_text = f"Q: {query} A: {response_text}"
            embedding = self.encoder.encode([full_text])
            
            # Use VPL if available (Pluralistic Alignment)
            if 'vpl' in self.reward_model:
                vpl = self.reward_model['vpl']
                mean_reward, std_reward = vpl.predict_reward(embedding)
                score = 1 / (1 + np.exp(-mean_reward[0]))
                uncertainty = std_reward[0]
                return score, uncertainty
            
            # Fallback to simple RM
            elif 'rm' in self.reward_model:
                score = self.reward_model['rm'].forward(embedding)[0]
                return 1 / (1 + np.exp(-score)), 0.0
                
            return 0.5, 0.0 
        except Exception as e:
            print(f"Scoring Error: {e}")
            return 0.5, 0.0

    def generate_response(self, user_query, rag_context, user_profile=None, journal_context=None):
        if not self.available:
            return self.fallback_response(user_query, rag_context)

        # Self-Correction Loop
        best_response = ""
        best_score = -1.0
        final_uncertainty = 0.0
        
        system_prompt = self.build_system_prompt(user_profile)
        context_str = self.format_context(rag_context, journal_context)
        
        for attempt in range(2):
            prompt = self.build_prompt(user_query, context_str, attempt, best_response)
            
            current_response = self.call_ollama(prompt, system_prompt)
            current_score, current_uncertainty = self.score_response(user_query, current_response)
            
            print(f"Attempt {attempt+1}: Score={current_score:.2f}, Uncertainty={current_uncertainty:.2f}")
            
            if current_score > best_score:
                best_score = current_score
                best_response = current_response
                final_uncertainty = current_uncertainty
                
            if current_score > 0.7: 
                break
                
        # Active Learning Trigger (Silent)
        # We still calculate uncertainty for logging, but we don't annoy the user.
        if final_uncertainty > 0.1:
            pass # Log internally if needed
        elif best_score > 0.8:
            pass # Log internally
            
        return best_response

    def rewrite_query(self, user_query):
        """Optimize user query for vector search using LLM"""
        if not self.available:
            return user_query # Fallback
            
        system_prompt = "You are an expert search engine optimizer. Your goal is to rewrite queries to be precise, medical, and context-rich for a mental health database."
        prompt = f"""
        Original Query: {user_query}
        
        Task: Rewrite this query to improve retrieval of relevant mental health advice. 
        - Expand vague terms (e.g. "tired" -> "fatigue burnout symptoms").
        - Keep it concise (max 10 words).
        - Remove conversational filler.
        - Output ONLY the rewritten query string.
        """
        
        rewritten = self.call_ollama(prompt, system_prompt)
        
        # Clean up response (some models are chatty)
        clean = rewritten.replace('"', '').replace("Here is the rewritten query:", "").strip()
        print(f"RAG Optimization: '{user_query}' -> '{clean}'")
        return clean

    def build_prompt(self, query, context, attempt, previous_bad_response):
        prompt = f"""
        Context Information:
        {context}

        User Question: {query}

        Instructions:
        Answer the user's question using the provided context. 
        Be empathetic, professional, and concise. 
        """
        if attempt > 0:
            prompt += f"\nCRITIQUE: Your previous answer was: '{previous_bad_response[:100]}...'. It was not helpful enough. Improve it."
            
        return prompt

    def call_ollama(self, prompt, system_prompt):
        try:
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_ctx": 4096
                }
            }
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get('response', "I'm having trouble thinking right now.")
            return f"Error: {response.text}"
        except Exception as e:
            print(f"Gen Error: {e}")
            return "Thinking failed."

    def format_context(self, rag_context, journal_context):
        context_str = ""
        for item in rag_context:
            context_str += f"- [{item['source']}] {item['title']}: {item['content'][:300]}...\n"
        if journal_context:
             context_str += f"\n[User Journal Context]: {journal_context}\n"
        return context_str

    def build_system_prompt(self, user_profile):
        base_prompt = (
            "You are the DC Well Being Assistant, a knowledgeable mental health expert. "
            "Your goal is to answer the user's questions DIRECTLY and FACTUALLY. "
            "DO NOT act like a therapist asking reflection questions (e.g. 'How does that make you feel?'). "
            "DO NOT say 'I'm here to support you' or 'I glad you reached out'. "
            "If the user asks for an explanation (e.g. 'explain x'), provide the explanation immediately. "
            "Be empathetic but conscise. "
        )
        if user_profile:
            name = user_profile.get('name', 'User')
            risk = user_profile.get('overall_risk', 'unknown')
            base_prompt += f"\nUser: {name}. Risk: '{risk}'."
        return base_prompt

    def fallback_response(self, query, rag_context):
        """Legacy logic-based response if LLM is down"""
        text = "Based on my database:\n\n"
        for i, r in enumerate(rag_context[:3], 1):
            text += f"**{r['title']}**\n{r['content'][:200]}...\n\n"
        text += "_[Ollama Llama 3.2 is offline]_"
        return text
