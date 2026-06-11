# """
# models.py — Inference backends for each evaluated system
# =========================================================
# Supported backends:
#   1. HuggingFace Inference API  (free tier, no GPU needed, rate-limited)
#   2. Ollama local               (best for repeated runs, needs Ollama installed)
#   3. vLLM local                 (production-grade, needs GPU)

# Model registry follows (Chen et al., 2023) Meditron and (Labrak et al., 2024)
# BioMistral paper conventions: 7B models for lightweight eval, 70B for full eval.

# Usage:
#     from models import get_model
#     model = get_model('llama3', backend='hf_api', hf_token='hf_...')
#     response = model.generate("What is the treatment for burns?")
# """

# import os
# import time
# import json
# import requests
# from abc import ABC, abstractmethod
# from typing import Optional


# # ─── Base class ───────────────────────────────────────────────────────────────

# class BaseModel(ABC):
#     def __init__(self, name: str):
#         self.name = name

#     @abstractmethod
#     def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
#         ...

#     def __repr__(self):
#         return f"{self.__class__.__name__}(name={self.name})"


# # ─── HuggingFace Inference API ────────────────────────────────────────────────

# class HFInferenceModel(BaseModel):
#     """
#     Uses the free HuggingFace Inference API.
#     Rate limit: ~300 req/hour on free tier.
#     For research: use a small subset (50-100 samples) or HF Pro.
#     """
#     HF_API = "https://api-inference.huggingface.co/models/{model_id}"

#     MODEL_IDS = {
#         'llama3':    'meta-llama/Meta-Llama-3-8B-Instruct',
#         'meditron':  'epfl-llm/meditron-7b',          # base; use chat-ft for instruction
#         'biomistral':'BioMistral/BioMistral-7B',
#     }

#     def __init__(self, model_key: str, hf_token: str, retry: int = 3):
#         model_id = self.MODEL_IDS.get(model_key, model_key)
#         super().__init__(model_id)
#         self.model_key = model_key
#         self.model_id = model_id
#         self.hf_token = hf_token
#         self.retry = retry
#         self.url = self.HF_API.format(model_id=model_id)
#         self.headers = {"Authorization": f"Bearer {hf_token}"}

#     def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
#         payload = {
#             "inputs": prompt,
#             "parameters": {
#                 "max_new_tokens": max_new_tokens,
#                 "temperature": 0.1,      # low temp for factual medical QA
#                 "do_sample": False,
#                 "return_full_text": False,
#             }
#         }
#         for attempt in range(self.retry):
#             try:
#                 resp = requests.post(
#                     self.url, headers=self.headers,
#                     json=payload, timeout=60
#                 )
#                 if resp.status_code == 503:
#                     # Model loading — wait and retry
#                     wait = resp.json().get('estimated_time', 20)
#                     print(f"    Model loading, waiting {wait:.0f}s...")
#                     time.sleep(min(wait, 60))
#                     continue
#                 if resp.status_code == 429:
#                     print("    Rate limited, sleeping 30s...")
#                     time.sleep(30)
#                     continue
#                 resp.raise_for_status()
#                 data = resp.json()
#                 if isinstance(data, list) and data:
#                     return data[0].get('generated_text', '').strip()
#                 return str(data)
#             except Exception as e:
#                 if attempt == self.retry - 1:
#                     print(f"    [ERROR] HF API failed after {self.retry} attempts: {e}")
#                     return ""
#                 time.sleep(5)
#         return ""


# # ─── Ollama local backend ─────────────────────────────────────────────────────

# class OllamaModel(BaseModel):
#     """
#     Local inference via Ollama (https://ollama.com).
#     Install: curl -fsSL https://ollama.ai/install.sh | sh
#     Pull models:
#         ollama pull llama3
#         ollama pull meditron   (community: ollama pull medllama2 as proxy)
#         ollama pull mistral    (BioMistral base)
#     """
#     OLLAMA_API = "http://localhost:5000/api/generate"

#     OLLAMA_NAMES = {
#         'llama3':    'llama3:latest',
#         'meditron':  'meditron',
#         'biomistral':'biomistral',
#     }

#     def __init__(self, model_key: str):
#         ollama_name = self.OLLAMA_NAMES.get(model_key, model_key)
#         super().__init__(ollama_name)
#         self.model_key = model_key

#     def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
#         payload = {
#         "model": self.name,
#         "prompt": prompt,
#         "stream": False,
#         "options": {"num_predict": max_new_tokens, "temperature": 0.1}
#     }
#         try:
#             resp = requests.post(self.OLLAMA_API, json=payload, timeout=120)
#             if resp.status_code != 200:
#                 print(f"    [ERROR] Ollama returned {resp.status_code}: {resp.text[:200]}")
#                 return ""
#             return resp.json().get('response', '').strip()
#         except requests.exceptions.ConnectionError:
#             print(f"    [ERROR] Ollama not reachable at {self.OLLAMA_API} — is 'ollama serve' still running?")
#             return ""
#         except Exception as e:
#             print(f"    [ERROR] Ollama call failed: {e}")
#             return ""
#     # def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
#     #     payload = {
#     #         "model": self.name,
#     #         "prompt": prompt,
#     #         "stream": False,
#     #         "options": {"num_predict": max_new_tokens, "temperature": 0.1}
#     #     }
#     #     try:
#     #         resp = requests.post(self.OLLAMA_API, json=payload, timeout=120)
#     #         resp.raise_for_status()
#     #         return resp.json().get('response', '').strip()
#     #     # except Exception as e:
#     #     #     print(f"    [ERROR] Ollama call failed: {e}")
#     #     #     return ""
#     #     except Exception as e:
#     #         print(f"    [ERROR] Ollama call failed: {e}")
#     #         return ""  # was missing — caused the crash


# # ─── Prompt templates ─────────────────────────────────────────────────────────

# def build_prompt(question: str, model_key: str, context: Optional[str] = None) -> str:
#     """
#     Prompt formatting per each model's paper conventions.
    
#     - Llama-3: standard instruction format (Meta guidelines)
#     - Meditron: uses "You are a medical expert..." system prompt (Chen et al., 2023)
#     - BioMistral: standard [INST]...[/INST] Mistral format (Labrak et al., 2024)
#     - RAG: prepends retrieved context before the question
#     """
#     if model_key == 'llama3':
#         system = "You are a helpful first aid expert. Answer the question concisely and accurately."
#         if context:
#             system += f"\n\nRelevant medical context:\n{context}"
#         return (
#             f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
#             f"{system}<|eot_id|>"
#             f"<|start_header_id|>user<|end_header_id|>\n"
#             f"{question}<|eot_id|>"
#             f"<|start_header_id|>assistant<|end_header_id|>\n"
#         )

#     elif model_key == 'meditron':
#         # From Meditron paper (Chen et al., 2023) - Fig 1 prompt format
#         system = "You are a medical expert. Your task is to address each query about a medical scenario with precision and provide a concise, accurate answer."
#         ctx = f"\nContext: {context}" if context else ""
#         return (
#             f"<|im_start|>system\n{system}<|im_end|>\n"
#             f"<|im_start|>question\n{question}{ctx}<|im_end|>\n"
#             f"<|im_start|>answer\n"
#         )

#     elif model_key == 'biomistral':
#         # Mistral [INST] format (Labrak et al., 2024)
#         system = "You are a biomedical expert. Answer the following medical question concisely."
#         ctx = f"\nContext: {context}" if context else ""
#         return f"[INST] {system}\n\nQuestion: {question}{ctx} [/INST]"

#     else:
#         # Generic fallback
#         ctx = f"\nContext: {context}\n" if context else ""
#         return f"Question: {question}{ctx}\nAnswer:"


# # ─── Factory ──────────────────────────────────────────────────────────────────

# def get_model(model_key: str,
#               backend: str = 'hf_api',
#               hf_token: str = '',
#               **kwargs) -> BaseModel:
#     """
#     Factory function.
#     Args:
#         model_key: 'llama3' | 'meditron' | 'biomistral'
#         backend:   'hf_api' | 'ollama'
#         hf_token:  HuggingFace token (required for hf_api)
#     """
#     if backend == 'hf_api':
#         if not hf_token:
#             hf_token = os.environ.get('HF_TOKEN', '')
#         if not hf_token:
#             raise ValueError(
#                 "HF token required for hf_api backend. "
#                 "Set HF_TOKEN env var or pass hf_token=... "
#                 "Get free token at: https://huggingface.co/settings/tokens"
#             )
#         return HFInferenceModel(model_key, hf_token)
#     elif backend == 'ollama':
#         return OllamaModel(model_key)
#     else:
#         raise ValueError(f"Unknown backend: {backend}. Choose 'hf_api' or 'ollama'")


"""
models.py — Inference backends for each evaluated system
=========================================================
Supported backends:
  1. Groq API      (recommended — free, fast, no GPU needed)
  2. HuggingFace   (fallback for Meditron which isn't on Groq)
  3. Ollama local  (local, needs sufficient RAM)

Model registry follows (Chen et al., 2023) Meditron and (Labrak et al., 2024)
BioMistral paper conventions: 7B models for lightweight eval, 70B for full eval.

Usage:
    # Set env vars first:
    #   set GROQ_API_KEY=your_groq_key
    #   set HF_TOKEN=your_hf_token   (only needed for meditron)

    from models import get_model
    model = get_model('llama3', backend='groq')
    response = model.generate("What is the treatment for burns?")
"""

import os
import time
import requests
from abc import ABC, abstractmethod
from typing import Optional


# ─── Base class ───────────────────────────────────────────────────────────────

class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


# ─── Groq API ─────────────────────────────────────────────────────────────────

class GroqModel(BaseModel):
    """
    Groq Cloud inference — free tier, very fast (LPU hardware).
    Models available: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768
    Get free API key: https://console.groq.com

    Rate limits (free tier):
      - llama3-8b:   30 req/min, 14,400 req/day
      - mixtral-8x7b: 30 req/min, 14,400 req/day
    More than enough for 50-500 sample evaluation.
    """
    GROQ_API = "https://api.groq.com/openai/v1/chat/completions"

    # Maps our model keys to Groq model IDs
    # mixtral-8x7b is used for BioMistral: same Mistral base architecture,
    # larger MoE variant — better performance proxy than a missing 7B
    # GROQ_MODEL_IDS = {
    #     'llama3':    'llama3-8b-8192',
    #     'biomistral':'mixtral-8x7b-32768',   # Mistral-family proxy
    # }
    GROQ_MODEL_IDS = {
        'llama3':    'llama-3.1-8b-instant',       # replaces decommissioned llama3-8b-8192
        'biomistral':'llama-3.3-70b-versatile',    # mixtral-8x7b-32768 also removed; using Llama3.3-70B
    }

    # System prompts per paper conventions
    SYSTEM_PROMPTS = {
        'llama3':    "You are a helpful first aid expert. Answer the question concisely and accurately. Provide only the answer, no extra commentary.",
        'biomistral':"You are a biomedical expert. Answer the following medical question concisely and accurately. Provide only the answer.",
    }

    def __init__(self, model_key: str, api_key: str, retry: int = 3):
        model_id = self.GROQ_MODEL_IDS.get(model_key, model_key)
        super().__init__(model_id)
        self.model_key = model_key
        self.api_key = api_key or os.environ.get('GROQ_API_KEY', '') or "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"
        self.retry = retry
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate(self, prompt: str, max_new_tokens: int = 256,
                 context: Optional[str] = None) -> str:
        system = self.SYSTEM_PROMPTS.get(self.model_key,
                 "You are a helpful medical assistant. Answer concisely.")
        if context:
            system += f"\n\nRelevant context:\n{context}"

        # Groq uses OpenAI-compatible chat format
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]
        payload = {
            "model": self.name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.1,
        }

        for attempt in range(self.retry):
            try:
                resp = requests.post(
                    self.GROQ_API, headers=self.headers,
                    json=payload, timeout=30
                )
                if resp.status_code == 429:
                    # Rate limited — wait and retry
                    retry_after = int(resp.headers.get('retry-after', 10))
                    print(f"    Rate limited, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                if resp.status_code != 200:
                    print(f"    [ERROR] Groq returned {resp.status_code}: {resp.text[:200]}")
                    return ""
                data = resp.json()
                return data['choices'][0]['message']['content'].strip()
            except requests.exceptions.ConnectionError as e:
                print(f"    [ERROR] Cannot reach Groq API. Check internet connection: {e}")
                return ""
            except Exception as e:
                if attempt == self.retry - 1:
                    print(f"    [ERROR] Groq call failed after {self.retry} attempts: {e}")
                    return ""
                time.sleep(3)
        return ""


# ─── HuggingFace Inference API (used for Meditron) ───────────────────────────

class HFInferenceModel(BaseModel):
    """
    HuggingFace Serverless Inference API.
    Used specifically for Meditron-7B which isn't available on Groq.
    Rate limit: ~300 req/hour on free tier.
    """
    HF_API = "https://api-inference.huggingface.co/models/{model_id}"

    MODEL_IDS = {
        'meditron': 'epfl-llm/meditron-7b',
        'llama3':   'meta-llama/Meta-Llama-3-8B-Instruct',
        'biomistral':'BioMistral/BioMistral-7B',
    }

    def __init__(self, model_key: str, hf_token: str, retry: int = 3):
        model_id = self.MODEL_IDS.get(model_key, model_key)
        super().__init__(model_id)
        self.model_key = model_key
        self.hf_token = hf_token
        self.retry = retry
        self.url = self.HF_API.format(model_id=model_id)
        self.headers = {"Authorization": f"Bearer {hf_token}"}

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.1,
                "do_sample": False,
                "return_full_text": False,
            }
        }
        for attempt in range(self.retry):
            try:
                resp = requests.post(
                    self.url, headers=self.headers,
                    json=payload, timeout=60
                )
                if resp.status_code == 503:
                    wait = resp.json().get('estimated_time', 20)
                    print(f"    Model loading on HF, waiting {wait:.0f}s...")
                    time.sleep(min(wait, 60))
                    continue
                if resp.status_code == 429:
                    print("    HF rate limited, sleeping 30s...")
                    time.sleep(30)
                    continue
                if resp.status_code != 200:
                    print(f"    [ERROR] HF returned {resp.status_code}: {resp.text[:200]}")
                    return ""
                data = resp.json()
                if isinstance(data, list) and data:
                    return data[0].get('generated_text', '').strip()
                return str(data)
            except requests.exceptions.ConnectionError as e:
                print(f"    [ERROR] Cannot reach HuggingFace API: {e}")
                return ""
            except Exception as e:
                if attempt == self.retry - 1:
                    print(f"    [ERROR] HF API failed after {self.retry} attempts: {e}")
                    return ""
                time.sleep(5)
        return ""


# ─── Ollama local backend ─────────────────────────────────────────────────────

class OllamaModel(BaseModel):
    """Local inference via Ollama. Needs sufficient RAM (5GB+ per 7B model)."""

    def __init__(self, model_key: str, host: str = "localhost", port: int = 5000):
        ollama_names = {
            'llama3':    'llama3:latest',
            'meditron':  'meditron:latest',
            'biomistral':'biomistral:latest',
        }
        name = ollama_names.get(model_key, f"{model_key}:latest")
        super().__init__(name)
        self.model_key = model_key
        self.api_url = f"http://{host}:{port}/api/generate"

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        payload = {
            "model": self.name,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_new_tokens, "temperature": 0.1}
        }
        try:
            resp = requests.post(self.api_url, json=payload, timeout=120)
            if resp.status_code != 200:
                print(f"    [ERROR] Ollama returned {resp.status_code}: {resp.text[:200]}")
                return ""
            return resp.json().get('response', '').strip()
        except requests.exceptions.ConnectionError:
            print(f"    [ERROR] Ollama not reachable at {self.api_url}. Is 'ollama serve' running?")
            return ""
        except Exception as e:
            print(f"    [ERROR] Ollama call failed: {e}")
            return ""


# ─── Prompt builder ───────────────────────────────────────────────────────────

def build_prompt(question: str, model_key: str, context: Optional[str] = None) -> str:
    """
    For Groq backend: just return the question — system prompt is set in GroqModel.
    For HF backend: use model-specific chat templates.
    """
    if model_key == 'meditron':
        # Meditron paper (Chen et al., 2023) prompt format
        system = "You are a medical expert. Your task is to address each query about a medical scenario with precision and provide a concise, accurate answer."
        ctx = f"\nContext: {context}" if context else ""
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>question\n{question}{ctx}<|im_end|>\n"
            f"<|im_start|>answer\n"
        )
    elif model_key == 'biomistral':
        # BioMistral paper (Labrak et al., 2024) Mistral [INST] format
        system = "You are a biomedical expert. Answer the following medical question concisely."
        ctx = f"\nContext: {context}" if context else ""
        return f"[INST] {system}\n\nQuestion: {question}{ctx} [/INST]"
    elif model_key == 'llama3':
        system = "You are a helpful first aid expert. Answer the question concisely and accurately."
        if context:
            system += f"\n\nRelevant medical context:\n{context}"
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{question}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
    else:
        ctx = f"\nContext: {context}\n" if context else ""
        return f"Question: {question}{ctx}\nAnswer:"


# ─── Local BioMistral backend (downloaded model → unlimited calls) ────────────

class LocalBioMistralModel(BaseModel):
    """
    Standalone BioMistral-7B baseline running the genuine local model
    (BioMistral/BioMistral-7B), with API fallback if no GPU is available.
    Shares the singleton backend with the KG pipeline so the 7B weights are
    only loaded into memory once.
    """
    # Tells the evaluation loop to pass the raw question (this model applies its
    # own chat template internally, so build_prompt() must NOT be used here).
    wants_raw_question = True

    SYSTEM_PROMPT = ("You are a biomedical expert. Answer the following medical "
                     "question concisely and accurately. Provide only the answer.")

    def __init__(self, prefer_local: bool = True, load_in_4bit: bool = True,
                 groq_api_key: str = ''):
        super().__init__("BioMistral-7B (local)")
        self.model_key = 'biomistral'
        from biomistral_backend import get_biomistral_backend
        self.backend = get_biomistral_backend(
            prefer_local=prefer_local,
            load_in_4bit=load_in_4bit,
            groq_api_key=groq_api_key,
        )

    def generate(self, prompt: str, max_new_tokens: int = 256,
                 context: Optional[str] = None) -> str:
        system = self.SYSTEM_PROMPT
        if context:
            system += f"\n\nRelevant context:\n{context}"
        return self.backend.chat(system, prompt, max_new_tokens=max_new_tokens)


# ─── Local Llama-3.1-8B-Instruct backend ─────────────────────────────────────

class LocalLlamaModel(BaseModel):
    """
    Standalone Llama-3.1-8B-Instruct baseline running the genuine local model,
    with Groq API fallback (llama-3.1-8b-instant) if no GPU is available.

    NOTE: The HuggingFace model is gated. Before the first local run:
      1. Accept licence at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
      2. Run: from huggingface_hub import login; login(token="your_hf_token")
    If the local load fails, the backend silently falls back to Groq API
    (llama-3.1-8b-instant is the same model — not a proxy).
    """
    wants_raw_question = True  # skip build_prompt; backend applies its own template

    SYSTEM_PROMPT = ("You are a first aid expert. Answer the following question "
                     "concisely and accurately. Provide only the answer.")

    def __init__(self, prefer_local: bool = True, load_in_4bit: bool = True,
                 groq_api_key: str = ''):
        super().__init__("Llama-3.1-8B-Instruct (local)")
        self.model_key = 'llama'
        from llama_backend import get_llama_backend
        self.backend = get_llama_backend(
            prefer_local=prefer_local,
            load_in_4bit=load_in_4bit,
            groq_api_key=groq_api_key,
        )

    def generate(self, prompt: str, max_new_tokens: int = 256,
                 context: Optional[str] = None) -> str:
        system = self.SYSTEM_PROMPT
        if context:
            system += f"\n\nRelevant context:\n{context}"
        return self.backend.chat(system, prompt, max_new_tokens=max_new_tokens)


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_model(model_key: str,
              backend: str = 'groq',
              groq_api_key: str = '',
              hf_token: str = '',
              ollama_port: int = 5000,
              **kwargs) -> BaseModel:
    """
    Factory function.

    Args:
        model_key:    'llama3' | 'llama' | 'meditron' | 'biomistral'
        backend:      'groq' | 'local' | 'hf_api' | 'ollama'
        groq_api_key: Groq API key (or set GROQ_API_KEY env var)
        hf_token:     HuggingFace token (needed for meditron via hf_api)
        ollama_port:  Port Ollama is running on (default 5000)

    Routing:
        backend='local'  + biomistral → genuine local BioMistral-7B (unlimited),
                                         API fallback (Groq llama-3.3-70b proxy) if no GPU
        backend='local'  + llama      → genuine local Llama-3.1-8B-Instruct (unlimited),
                                         API fallback (Groq llama-3.1-8b-instant) if no GPU
        backend='local'  + other      → falls back to Groq for that model
        llama3 / llama   → groq (llama-3.1-8b-instant) unless backend='local'
        biomistral       → groq (llama-3.3-70b proxy)  unless backend='local'
        meditron         → hf_api (epfl-llm/meditron-7b, not available on Groq)
    """
    # backend='local' → run BioMistral or Llama locally
    if backend == 'local':
        if model_key == 'biomistral':
            return LocalBioMistralModel(groq_api_key=groq_api_key)
        if model_key in ('llama', 'llama3'):
            return LocalLlamaModel(groq_api_key=groq_api_key)
        print(f"    [INFO] backend='local' for '{model_key}' not implemented; "
              f"routing to Groq.")
        backend = 'groq'

    # Auto-route meditron to HF API regardless of backend
    # (it's not available on Groq)
    effective_backend = backend
    if model_key == 'meditron' and backend == 'groq':
        effective_backend = 'hf_api'
        print(f"    [INFO] Meditron not on Groq — routing to HuggingFace Inference API")


    if effective_backend == 'groq':
        key = groq_api_key or os.environ.get('GROQ_API_KEY', '')
        if not key:
            raise ValueError(
                "Groq API key required. Set env var: set GROQ_API_KEY=your_key\n"
                "Get free key at: https://console.groq.com"
            )
        return GroqModel(model_key, key)

    elif effective_backend == 'hf_api':
        token = hf_token or os.environ.get('HF_TOKEN', '')
        if not token:
            raise ValueError(
                "HuggingFace token required for meditron. "
                "Set env var: set HF_TOKEN=your_token\n"
                "Get free token at: https://huggingface.co/settings/tokens"
            )
        return HFInferenceModel(model_key, token)

    elif effective_backend == 'ollama':
        return OllamaModel(model_key, port=ollama_port)

    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose: groq | hf_api | ollama")