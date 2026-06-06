# -*- coding: utf-8 -*-
"""
generator/model_registry.py
============================
Central registry of all medical LLM generators supported for comparison.

Each entry maps a short model key to:
  - ollama_tag  : exact tag to pass to `ollama pull` / Ollama API
  - display     : human-friendly name for reports
  - param_size  : approximate parameter count (for sorting / filtering)
  - notes       : brief note for the comparison report header
"""

MODEL_REGISTRY: dict[str, dict] = {
    "medgemma": {
        "ollama_tag":        "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M",
        "display":          "MedGemma-4B (Q4_K_M)",
        "param_size":       "4B",
        "supports_few_shot": True,
        "prompt_style":     "chat",   # full structured prompt with rules
        "notes":            "Google MedGemma 1.5 4B instruction-tuned, GGUF Q4_K_M via Unsloth",
    },
    "meditron": {
        "ollama_tag":        "meditron",
        "display":          "Meditron-7B",
        "param_size":       "7B",
        "supports_few_shot": False,
        "prompt_style":     "simple", # Vicuna format, no numbered rules
        "notes":            "EPFL Meditron 7B, fine-tuned on medical literature (PubMed + guidelines)",
    },
    "medalpaca": {
        "ollama_tag":        "hf.co/mradermacher/Med-Alpaca-2-7b-chat-GGUF:Q4_K_M",
        "display":          "Med-Alpaca-2-7B-chat (Q4_K_M)",
        "param_size":       "7B",
        "supports_few_shot": False,
        "prompt_style":     "simple",
        "notes":            "LLaMA-2 chat fine-tuned on medical QA/dialogues (Med-Alpaca-2); GGUF Q4_K_M",
    },
    "biomistral": {
        "ollama_tag":        "adrienbrault/biomistral-7b:Q4_K_M",
        "display":          "BioMistral-7B (Q4_K_M)",
        "param_size":       "7B",
        "supports_few_shot": False,
        "prompt_style":     "simple",
        "notes":            "Mistral 7B fine-tuned on PubMed Central biomedical literature; Q4_K_M via adrienbrault",
    },
    "llama3med42": {
        "ollama_tag":        "hf.co/QuantFactory/Llama3-Med42-8B-GGUF:Q4_K_M",
        "display":          "Llama3-Med42-8B (Q4_K_M)",
        "param_size":       "8B",
        "supports_few_shot": True,
        "prompt_style":     "chat",
        "notes":            "M42 Health Llama-3 8B fine-tuned on medical corpora; GGUF Q4_K_M via QuantFactory",
    },
    "pmcllama": {
        "ollama_tag":        "hf.co/bartowski/pmc-llama-13b-GGUF:Q4_K_M",
        "display":          "PMC-LLaMA-13B (Q4_K_M)",
        "param_size":       "13B",
        "supports_few_shot": False,
        "prompt_style":     "simple",
        "notes":            "LLaMA 13B fine-tuned on 4.8M PubMed Central papers; GGUF Q4_K_M",
    },
}

# Ordered list for default CLI run (smallest → largest)
DEFAULT_ORDER = ["medgemma", "meditron", "medalpaca", "biomistral", "llama3med42", "pmcllama"]


def get_model(key: str) -> dict:
    """Return registry entry for *key*; raises KeyError with helpful message."""
    if key not in MODEL_REGISTRY:
        valid = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model key '{key}'. Valid keys: {valid}")
    return MODEL_REGISTRY[key]


def list_models() -> None:
    """Print a formatted table of all registered models."""
    print(f"\n{'Key':<14} {'Display':<28} {'Size':<6}  Ollama Tag")
    print("-" * 90)
    for key in DEFAULT_ORDER:
        m = MODEL_REGISTRY[key]
        print(f"  {key:<12} {m['display']:<28} {m['param_size']:<6}  {m['ollama_tag']}")
    print()
