# AION: Adaptive Inference Optimization Network

Experimental code for the IEEE paper:

**"AION: Adaptive Inference Optimization Network for High-Throughput,
Hallucination-Aware, and Feedback-Driven Large Language Model Serving"**

*Karthick Gunasekaran, IEEE Senior Member*

## Experiments

| Script | What It Measures | Key Result |
|---|---|---|
| `01_throughput_benchmark.py` | Throughput vs batch size | 40x improvement batch 1→8 |
| `02_confidence_estimation.py` | Entropy-based confidence scoring + overhead | 0.2006 separation gap, −0.27% overhead |
| `03_model_size_comparison.py` | Threshold sensitivity across model sizes | 65.9% better separation at 2.7B vs 1.3B |

## Requirements

- Google Colab (free tier) or any machine with NVIDIA GPU
- Python 3.10+
- `pip install transformers torch accelerate`

## Hardware Used

- GPU: Tesla T4 (15GB VRAM)
- CUDA 13.0
- Models: facebook/opt-1.3b, facebook/opt-2.7b (FP16)

## Citation

If you use this code, please cite the paper once published in IEEE Computer.
