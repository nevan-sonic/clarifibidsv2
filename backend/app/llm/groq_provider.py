from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import httpx
from pydantic import BaseModel
from app.core.config import settings

class LLMResponse(BaseModel):
    content: str
    model_name: str
    provider: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: float

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.1, 
        max_tokens: int = 512
    ) -> LLMResponse:
        pass

class GroqProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: Optional[str] = None):
        self.api_key = api_key
        self.model_name = model_name or settings.GROQ_MODEL
        self.fallback_models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.1, 
        max_tokens: int = 512
    ) -> LLMResponse:
        if not self.api_key or self.api_key.startswith("gsk_placeholder"):
            raise ValueError(
                "Real GROQ_API_KEY is missing in backend/.env. Please configure your actual Groq API key."
            )
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Try primary model first, followed by fallbacks if transient failure/timeout
        candidate_models = [self.model_name] + [m for m in self.fallback_models if m != self.model_name]
        last_error = None

        for model in candidate_models:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(self.base_url, headers=headers, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                        usage = data.get("usage", {})
                        return LLMResponse(
                            content=content,
                            model_name=model,
                            provider="groq",
                            prompt_tokens=usage.get("prompt_tokens"),
                            completion_tokens=usage.get("completion_tokens"),
                            latency_ms=0.0
                        )
                    else:
                        last_error = f"Groq {model} HTTP {response.status_code}: {response.text}"
            except Exception as e:
                err_text = str(e).strip() or type(e).__name__
                last_error = f"Groq {model} request failed: {err_text}"

        raise RuntimeError(last_error or "All Groq model endpoints timed out or failed.")

def get_llm_provider() -> BaseLLMProvider:
    if settings.LLM_PROVIDER.lower() == "groq":
        return GroqProvider(api_key=settings.GROQ_API_KEY, model_name=settings.GROQ_MODEL)
    raise NotImplementedError(f"Provider {settings.LLM_PROVIDER} not supported in current MVP.")
