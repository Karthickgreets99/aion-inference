"""
AION Experiment 2: Confidence Estimation using Entropy H(t)
Paper: AION: Adaptive Inference Optimization Network
Author: Karthick Gunasekaran (IEEE Senior Member)
GPU: Tesla T4 (Google Colab), Model: facebook/opt-1.3b, FP16
Results:
  - Flagged avg confidence: 0.6701
  - Passed avg confidence:  0.8707
  - Separation gap: 0.2006 (29.9%)
  - Overhead: -0.27% (negligible)
"""

import torch
import time
import json
import math
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "facebook/opt-1.3b"
THETA = 0.75  # confidence threshold

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda"
)
model.eval()
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4 factual + 4 hallucination-prone prompts
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

def run_confidence_experiment(prompts, threshold=0.75):
    inputs = tokenizer(
        prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=128
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id
        )

    vocab_size = model.config.vocab_size
    max_entropy = math.log(vocab_size)
    steps = len(outputs.scores)

    results = []
    for b in range(len(prompts)):
        mean_ent = sum(
            compute_entropy(outputs.scores[s][b])
            for s in range(steps)
        ) / steps
        confidence = 1.0 - (mean_ent / max_entropy)
        results.append({
            "prompt": prompts[b],
            "confidence": round(confidence, 4),
            "flagged": confidence < threshold
        })
    return results

print(f"Running confidence estimation at θ={THETA}\n")
results = run_confidence_experiment(PROMPTS, threshold=THETA)

flagged  = [r for r in results if r["flagged"]]
passed   = [r for r in results if not r["flagged"]]

print(f"{'Prompt':<45} | {'Confidence':>10} | {'Flagged':>8}")
print("-" * 70)
for r in results:
    print(f"{r['prompt'][:45]:<45} | {r['confidence']:>10.4f} | {'YES' if r['flagged'] else 'no':>8}")

avg_flagged = sum(r['confidence'] for r in flagged) / max(1, len(flagged))
avg_passed  = sum(r['confidence'] for r in passed)  / max(1, len(passed))
gap         = avg_passed - avg_flagged

print(f"\nFlagged: {len(flagged)}/8 ({100*len(flagged)//8}%)")
print(f"Avg confidence — flagged: {avg_flagged:.4f} | passed: {avg_passed:.4f}")
print(f"Separation gap: {gap:.4f} ({100*gap/avg_flagged:.1f}%)")

# Overhead measurement
inputs4 = tokenizer(
    PROMPTS[:4], return_tensors="pt",
    padding=True, truncation=True, max_length=64
).to("cuda")

torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    model.generate(**inputs4, max_new_tokens=64, do_sample=False,
                  pad_token_id=tokenizer.eos_token_id)
torch.cuda.synchronize()
time_without = time.perf_counter() - t0

torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    model.generate(**inputs4, max_new_tokens=64, do_sample=False,
                  return_dict_in_generate=True, output_scores=True,
                  pad_token_id=tokenizer.eos_token_id)
torch.cuda.synchronize()
time_with = time.perf_counter() - t0

overhead = ((time_with - time_without) / time_without) * 100
print(f"\nOverhead: {overhead:.2f}%")
