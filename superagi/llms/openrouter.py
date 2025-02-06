import requests
from superagi.llms.base_llm import BaseLlm

class OpenRouter(BaseLlm):
    def __init__(self, api_key, model='default-model', temperature=0.6, max_tokens=800, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def chat_completion(self, messages):
        url = f"https://api.openrouter.ai/v1/models/{self.model}/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        return {"response": response_data, "content": response_data.get("choices", [{}])[0].get("message", {}).get("content", "")}

    def get_source(self):
        return "openrouter"

    def get_api_key(self):
        return self.api_key

    def get_model(self):
        return self.model

    def get_models(self):
        url = "https://api.openrouter.ai/v1/models"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        response = requests.get(url, headers=headers)
        response_data = response.json()
        return [model["id"] for model in response_data.get("data", [])]

    def verify_access_key(self):
        url = "https://api.openrouter.ai/v1/models"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        response = requests.get(url, headers=headers)
        return response.status_code == 200
