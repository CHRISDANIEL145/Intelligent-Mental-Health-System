from llm_interface import LLMInterface

print("Testing LLM Interface...")
llm = LLMInterface()

print(f"LLM Available: {llm.available}")

# Test Simple Generation
response = llm.generate_response(
    "What is stress?", 
    rag_context=[{'title': 'Stress Info', 'content': 'Stress is a reaction...', 'source': 'Test'}],
    user_profile={'name': 'Test User', 'overall_risk': 'Low'}
)

print("\n--- LLM Response ---")
print(response)
print("--------------------")

if "Ollama" in response or "Stress is a reaction" in response:
    print("✓ Test Passed: Response generated (either via LLM or Fallback)")
else:
    print("✗ Test Failed: Unexpected response format")
