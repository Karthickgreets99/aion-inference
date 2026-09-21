"""
AION Experiment 1: Throughput Baseline Benchmark
Paper: AION: Adaptive Inference Optimization Network for High-Throughput,
       Hallucination-Aware, and Feedback-Driven LLM Serving
Author: Karthick Gunasekaran (IEEE Senior Member)
GPU: Tesla T4 (Google Colab), Model: facebook/opt-1.3b, FP16
Results: 10.2 tok/s (batch=1) to 412.9 tok/s (batch=8) — 40x improvement
"""

import torch
import time
import json
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "facebook/opt-1.3b"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory/1e9, 1)} GB")
print(f"Loading {MODEL_NAME}...")

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
print("Model loaded\n")

PROMPTS = [
    "Explain what a transformer model is in simple terms.",
    "What are the main causes of inflation in modern economies?",
    "Summarize the key principles of distributed systems.",
    "How do neural networks learn from training data?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain the attention mechanism in deep learning.",
    "How does gradient descent optimization work?",
    "What are the main benefits of containerization?",
]

results = []
print("Batch | Throughput tok/s | Latency ms/req")
print("-" * 45)

for batch_size in [1, 2, 4, 8]:
    batch = PROMPTS[:batch_size]
    inputs = tokenizer(
        batch, return_tensors="pt",
        padding=True, truncation=True, max_length=64
    ).to("cuda")

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=32,
                      do_sample=False,
                      pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()

    times, token_counts = [], []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        new_toks = (out.shape[1] - inputs["input_ids"].shape[1]) * batch_size
        times.append(elapsed)
        token_counts.append(new_toks)

    avg_t  = sum(times) / 3
    avg_tk = sum(token_counts) / 3
    tput   = avg_tk / avg_t
    lat    = (avg_t / batch_size) * 1000

    results.append({
        "batch_size": batch_size,
        "throughput_tokens_per_sec": round(tput, 1),
        "latency_ms_per_request": round(lat, 1),
        "avg_output_tokens": round(avg_tk / batch_size)
    })
    print(f"{batch_size:>5} | {tput:>17.1f} | {lat:>14.1f}")

print("\nResults:")
print(json.dumps(results, indent=2))
