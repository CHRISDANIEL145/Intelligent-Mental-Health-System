
import requests
import json

try:
    response = requests.post(
        "http://127.0.0.1:5000/chat_api",
        json={"message": "how does stress affect sleep?"},
        timeout=10
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print("Response Content:")
    print(data['response'])
    
except Exception as e:
    print(f"Test failed: {e}")
