"""
biomistral_backend.py — Shared BioMistral-7B inference backend
================================================================
Single source of truth for BioMistral inference across the whole project
(KG + LLM pipeline, constrained ablation, and the standalone BioMistral baseline).

Priority order (this is exactly what you asked for):
  1. LOCAL  — download `BioMistral/BioMistral-7B` once and run it on-device.
              This gives UNLIMITED calls (no API rate limits) — ideal for Colab GPU.
  2. API    — if the local model cannot be loaded (no GPU / no torch / OOM /
              download blocked), fall back to a hosted API using your existing key.
              BioMistral is not served on Groq, so the API fallback uses
              llama-3.3-70b-versatile as a documented Mistral-family proxy
              (same convention the original models.py used). A warning is printed
              so results stay honest in your write-up.

The local model is loaded ONCE and shared (module-level singleton) so that the
KG pipeline, the constrained pipeline and the baseline all reuse the same weights
instead of loading three copies and blowing up GPU memory.

Usage:
    from biomistral_backend import get_biomistral_backend
    bm = get_biomistral_backend()                 # loads once, cached
    text = bm.chat(system="You are a biomedical expert.",
                   user="What should you do if someone is choking?")
"""

import os
import time
import requests
from typing import Optional

# ── API fallback key (kept on purpose — do NOT remove) ────────────────────────
# If the local model is unavailable, we fall back to the hosted API using this
# key (env var GROQ_API_KEY overrides it).
GROQ_API_KEY = os.environ.get(
    "GROQ_API_KEY",
    "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",
)
# Real BioMistral is not on Groq — this Llama-3.3-70B model is used only as a
# Mistral-family API proxy when the local BioMistral cannot be loaded.
GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"

# Hugging Face model id for the genuine BioMistral-7B (public, non-gated).
HF_MODEL_ID = os.environ.get("BIOMISTRAL_MODEL_ID", "BioMistral/BioMistral-7B")


# ══════════════════════════════════════════════════════════════════════════════
# Backend
# ══════════════════════════════════════════════════════════════════════════════

class BioMistralBackend:
    """
    BioMistral inference with local-first, API-fallback routing.

    Parameters
    ----------
    prefer_local : bool
        Try to load the real BioMistral-7B locally first (recommended).
    load_in_4bit : bool
        Use 4-bit quantisation (bitsandbytes) when a CUDA GPU is present.
        ~4.5 GB VRAM instead of ~14 GB — needed on a free Colab T4.
    groq_api_key : str
        Key for the API fallback (defaults to GROQ_API_KEY above / env var).
    max_new_tokens : int
        Default generation length.
    """

    def __init__(self,
                 prefer_local: bool = True,
                 load_in_4bit: bool = True,
                 groq_api_key: str = "",
                 max_new_tokens: int = 256):
        self.mode = "none"
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.default_max_new_tokens = max_new_tokens
        self.groq_api_key = groq_api_key or GROQ_API_KEY

        if prefer_local:
            self._try_load_local(load_in_4bit)

        if self.mode == "none":
            self._setup_api_fallback()

    # ── Local loader ─────────────────────────────────────────────────────────
    def _try_load_local(self, load_in_4bit: bool):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            print(f"  [BioMistral] transformers/torch unavailable ({e}) — "
                  f"will use API fallback.")
            return

        has_cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        try:
            print(f"  [BioMistral] Loading local model '{HF_MODEL_ID}' "
                  f"(this downloads ~weights on first run)...")
            self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            quant_config = None
            if has_cuda and load_in_4bit:
                try:
                    import bitsandbytes  # noqa: F401
                    from transformers import BitsAndBytesConfig
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    print("  [BioMistral] Using 4-bit quantisation (bitsandbytes).")
                except Exception as e:
                    print(f"  [BioMistral] bitsandbytes not usable ({e}) — "
                          f"loading in fp16 instead.")

            if quant_config is not None:
                self.model = AutoModelForCausalLM.from_pretrained(
                    HF_MODEL_ID, quantization_config=quant_config,
                    device_map="auto", torch_dtype=torch.float16,
                )
                self.device = "cuda"
            elif has_cuda:
                self.model = AutoModelForCausalLM.from_pretrained(
                    HF_MODEL_ID, torch_dtype=torch.float16, device_map="auto",
                )
                self.device = "cuda"
            else:
                print("  [BioMistral] No CUDA GPU detected — loading on CPU "
                      "(works, but slow). For speed use a Colab GPU runtime.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    HF_MODEL_ID, torch_dtype=torch.float32,
                )
                self.device = "cpu"

            self.model.eval()
            self._torch = torch
            self.mode = "local"
            print(f"  [BioMistral] ✅ Local model ready on {self.device.upper()} — "
                  f"unlimited calls enabled.")
        except Exception as e:
            print(f"  [BioMistral] Local load failed ({e}) — using API fallback.")
            self.model = None
            self.tokenizer = None
            self.mode = "none"

    # ── API fallback setup ─────────────────────────────────────────────────────
    def _setup_api_fallback(self):
        if not self.groq_api_key:
            print("  [BioMistral] ⚠️ No local model AND no API key — generation "
                  "will return empty strings. Set GROQ_API_KEY or enable a GPU.")
            self.mode = "none"
            return
        self.mode = "api"
        self._groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self._groq_headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        print(f"  [BioMistral] Using API fallback (Groq '{GROQ_FALLBACK_MODEL}' "
              f"as Mistral-family proxy — real BioMistral is not hosted on Groq).")

    # ── Prompt formatting (Mistral [INST] convention, Labrak et al. 2024) ──────
    @staticmethod
    def _format_mistral(system: str, user: str) -> str:
        sys_part = (system.strip() + "\n\n") if system and system.strip() else ""
        return f"[INST] {sys_part}{user.strip()} [/INST]"

    # ── Unified chat interface ─────────────────────────────────────────────────
    def chat(self, system: str, user: str,
             max_new_tokens: Optional[int] = None) -> str:
        max_new_tokens = max_new_tokens or self.default_max_new_tokens
        if self.mode == "local":
            return self._generate_local(system, user, max_new_tokens)
        if self.mode == "api":
            return self._generate_api(system, user, max_new_tokens)
        return ""

    # ── Local generation ─────────────────────────────────────────────────────
    def _generate_local(self, system: str, user: str, max_new_tokens: int) -> str:
        torch = self._torch
        prompt = self._format_mistral(system, user)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=4096)
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,            # greedy → deterministic for research
                num_beams=1,
                repetition_penalty=1.1,     # avoids the 7B "loop" failure mode
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][input_len:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        return text

    # ── API generation ─────────────────────────────────────────────────────────
    def _generate_api(self, system: str, user: str, max_new_tokens: int,
                      retry: int = 3) -> str:
        payload = {
            "model": GROQ_FALLBACK_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.1,
        }
        for attempt in range(retry):
            try:
                resp = requests.post(self._groq_url, headers=self._groq_headers,
                                     json=payload, timeout=45)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("retry-after", 15))
                    print(f"    [BioMistral-API] rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    print(f"    [BioMistral-API] ERROR {resp.status_code}: "
                          f"{resp.text[:150]}")
                    return ""
                return resp.json()["choices"][0]["message"]["content"].strip()
            except requests.exceptions.ConnectionError:
                print("    [BioMistral-API] cannot reach API.")
                return ""
            except Exception as e:
                if attempt == retry - 1:
                    print(f"    [BioMistral-API] failed: {e}")
                    return ""
                time.sleep(3)
        return ""

    def __repr__(self):
        return (f"BioMistralBackend(mode={self.mode}, device={self.device}, "
                f"model={HF_MODEL_ID})")


# ══════════════════════════════════════════════════════════════════════════════
# Module-level singleton — load the heavy model ONCE and reuse everywhere
# ══════════════════════════════════════════════════════════════════════════════

_BACKEND_SINGLETON: Optional[BioMistralBackend] = None


def get_biomistral_backend(prefer_local: bool = True,
                           load_in_4bit: bool = True,
                           groq_api_key: str = "",
                           max_new_tokens: int = 256) -> BioMistralBackend:
    """Return the shared BioMistral backend, creating it on first call."""
    global _BACKEND_SINGLETON
    if _BACKEND_SINGLETON is None:
        _BACKEND_SINGLETON = BioMistralBackend(
            prefer_local=prefer_local,
            load_in_4bit=load_in_4bit,
            groq_api_key=groq_api_key,
            max_new_tokens=max_new_tokens,
        )
    return _BACKEND_SINGLETON


def download_biomistral(load_in_4bit: bool = True):
    """
    Convenience: pre-download + load BioMistral-7B so subsequent calls are instant.
    Call this once at the top of your Colab notebook.
    """
    print("Downloading / loading BioMistral-7B (first run can take a few minutes)...")
    bm = get_biomistral_backend(prefer_local=True, load_in_4bit=load_in_4bit)
    print(f"Backend ready: {bm}")
    return bm


if __name__ == "__main__":
    # Quick smoke test
    bm = get_biomistral_backend()
    print(bm)
    print("---")
    print(bm.chat(
        system="You are a biomedical first aid expert. Answer concisely.",
        user="What should you do if someone is choking?",
        max_new_tokens=128,
    ))
