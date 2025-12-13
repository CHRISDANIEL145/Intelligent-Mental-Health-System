from llm_interface import LLMInterface
import time

print("--- DIAGNOSTIC START ---")
llm = LLMInterface()
print(f"LLM Initial Availability: {llm.available}")

# Force check again
print("Checking availability via direct request...")
try:
    import requests
    r = requests.get("http://localhost:11434/api/tags")
    print(f"Ollama API Status: {r.status_code}")
    print(f"Models: {r.text}")
except Exception as e:
    print(f"Ollama Connection Error: {e}")

print("\nAttempting Generation...")
start = time.time()
response = llm.generate_response("Write a haiku", [])
duration = time.time() - start
print(f"Generation took: {duration:.2f}s")
print(f"Response: {response}")

if "Ollama" in response and "offline" in response:
    print("RESULT: FAIL (Fallback Used)")
else:
    print("RESULT: SUCCESS (LLM Used)")
