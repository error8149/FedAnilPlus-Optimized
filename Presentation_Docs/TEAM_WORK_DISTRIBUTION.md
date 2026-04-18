# FedAnilPlus — Team Work Distribution

This document outlines the division of work among the three team members for the optimization and bug-fixing phase of the FedAnilPlus project.

---

## 👨‍💻 Muhammad Hadi
**Role:** System Architect & Performance Engineer

| Task | File(s) | Impact |
| --- | --- | --- |
| **GPU Acceleration** | `main.py` | Implementation of CUDA detection and `cudnn.benchmark` for 10x speedup. |
| **Mixed Precision (AMP)** | `Enterprise.py` | Integration of `torch.cuda.amp` (GradScaler/Autocast) for 2x faster training. |
| **Optimization Tuning** | `Enterprise.py` | Implementation of `set_to_none=True` in `zero_grad` for efficient memory management. |

---

## 👨‍💻 Muhammad Jahangir
**Role:** AI Model & Optimization Specialist

| Task | File(s) | Impact |
| --- | --- | --- |
| **Batch Normalization** | `Models.py` | Strategic placement of `BatchNorm2d` layers to boost accuracy by +5%. |
| **Weight Decay (L2)** | `Enterprise.py` | Implementation of L2 regularization in SGD/Adam to prevent overfitting. |
| **Gradient Clipping** | `Enterprise.py` | Integration of `clip_grad_norm_` to stabilize training in heterogeneous federated environments. |

---

## 👨‍💻 Talha Safique
**Role:** Reliability & Memory Management Engineer

| Task | File(s) | Impact |
| --- | --- | --- |
| **GPU Memory (OOM) Fix** | `Enterprise.py` | Creation of `_deepcopy_to_cpu` helper to resolve CUDA Out-of-Memory crashes. |
| **System Compatibility** | `Models.py` | Resolution of CUDA-to-NumPy errors by implementing CPU-tensor conversion for KMedoids. |
| **Memory Lifecycle** | `main.py` | Implementation of periodic `torch.cuda.empty_cache()` and system documentation. |

---

### Summary Checklist for Presentation
- [x] **Performance:** 10x-15x faster execution compared to baseline.
- [x] **Accuracy:** Stable convergence with reduced overfitting.
- [x] **Scalability:** Large-scale enterprise simulation (100 nodes) now runs on 2GB VRAM without crashing.
