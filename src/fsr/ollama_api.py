import requests
import json

# URL to Ollama local API
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def generate_text(prompt: str, model: str = "llama3.1"):
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,  # model name，such as "llama3"
        "prompt": prompt,
        # "max_tokens": 131072,
        "options":{
            "temperature": 0.3,
        #    "num_predict": 512,
        #    "repeat_penalty": 1.2
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=data, headers=headers)
        response.raise_for_status()

        response_text = response.text
        generated_text = ""

        for line in response_text.splitlines():
            if line:
                line_data = json.loads(line)
                generated_text += line_data.get("response", "")
        
        return generated_text

    except requests.exceptions.RequestException as e:
        print(f"API 请求失败: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        return None


# Test Ollama local API
if __name__ == "__main__":
    prompt = "What is the capital of USA?"
    result = generate_text(prompt)
    
    if result:
        print(f"Generated text: {result}")

