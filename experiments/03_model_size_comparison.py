"""
AION Experiment 3: Threshold Sensitivity Across Model Sizes
Paper: AION: Adaptive Inference Optimization Network
Author: Karthick Gunasekaran (IEEE Senior Member)
GPU: Tesla T4 (Google Colab)
Models: facebook/opt-1.3b vs facebook/opt-2.7b, FP16
Results:
  - OPT-1.3B separation gap: 0.2006
  - OPT-2.7B separation gap: 0.3323 (+65.9% improvement)
  - Novel finding: threshold calibration must be domain-sensitive
"""

import torch
import json
import math
import gc
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "What programming language was Python written in?",
    "How many days are in a week?",
    "What did the CEO of Anthropic say at the 2024 IEEE conference?",
    "What is the exact revenue of OpenAI in Q3 2024?",
    "Who won the 2025 Nobel Prize in Physics?",
    "What are the specific terms of the EU AI Act Article 47?",
]

def compute_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    return -(probs * log_probs).sum(dim=-1).mean().item()

def run_experiment(model_name, prompts, threshold=0.75):
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=128
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=64, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
            pad_token_id=tokenizer.eos_token_id
        )

    max_entropy = math.log(model.config.vocab_size)
    steps = len(outputs.scores)
    results = []

    for b in range(len(prompts)):
        mean_ent = sum(
            compute_entropy(outputs.scores[s][b])
            for s in range(steps)
        ) / steps
        confidence = 1.0 - (mean_ent / max_entropy)
        results.append({
            "prompt": prompts[b][:40],
            "confidence": round(confidence, 4),
            "flagged": confidence < threshold
        })

    flagged = [r for r in results if r["flagged"]]
    passed  = [r for r in results if not r["flagged"]]
    avg_f   = sum(r["confidence"] for r in flagged) / max(1, len(flagged))
    avg_p   = sum(r["confidence"] for r in passed)  / max(1, len(passed))

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "flagged_count": len(flagged),
        "avg_confidence_flagged": round(avg_f, 4),
        "avg_confidence_passed":  round(avg_p, 4),
        "separation_gap":         round(avg_p - avg_f, 4),
        "per_prompt": results
    }

r1 = run_experiment("facebook/opt-1.3b", PROMPTS)
r2 = run_experiment("facebook/opt-2.7b", PROMPTS)

print("\n" + "="*55)
print("COMPARISON RESULTS")
print("="*55)
print(f"{'Metric':<30} {'OPT-1.3B':>12} {'OPT-2.7B':>12}")
print("-"*55)
print(f"{'Flagged count':<30} {r1['flagged_count']:>12} {r2['flagged_count']:>12}")
print(f"{'Avg flagged confidence':<30} {r1['avg_confidence_flagged']:>12} {r2['avg_confidence_flagged']:>12}")
print(f"{'Avg passed confidence':<30} {r1['avg_confidence_passed']:>12} {r2['avg_confidence_passed']:>12}")
print(f"{'Separation gap':<30} {r1['separation_gap']:>12} {r2['separation_gap']:>12}")
improvement = (r2['separation_gap'] - r1['separation_gap']) / r1['separation_gap'] * 100
print(f"{'Gap improvement':<30} {'':>12} {improvement:>11.1f}%")

print("\nFull results:")
print(json.dumps([r1, r2], indent=2))
