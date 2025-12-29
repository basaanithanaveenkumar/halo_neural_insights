# Halo Neural Insights

## Overview
A comprehensive knowledge repository for deep learning insights, neural network architectures, and AI research learnings. This project serves as a personal reference hub for documenting neural architecture innovations, experimental findings, and practical implementation insights across computer vision, autonomous driving perception, and foundation models.

## Purpose
To maintain a structured collection of:
- Neural network architecture analysis and breakdowns
- Research paper implementations and insights
- Experimental results and ablation studies
- Best practices and optimization techniques
- Industry trends and emerging methodologies




# Breakthrough Papers in Deep Learning

A curated collection of foundational and breakthrough papers in deep learning, computer vision, and autonomous driving.

---

## Table of Contents
1. [Transformers & Attention](#transformers--attention)
2. [Object Detection](#object-detection)
3. [3D Perception & BEV](#3d-perception--bev)
4. [Backbone Architectures](#backbone-architectures)
5. [Additional Resources](#additional-resources)

---

## Transformers & Attention

| Paper | Authors | Year | ArXiv/Link | Key Contribution |
|-------|---------|------|------------|------------------|
| **Attention Is All You Need** | Vaswani et al. (Google) | 2017 | [arxiv:1706.03762](https://arxiv.org/abs/1706.03762) | Introduced the Transformer architecture based purely on attention mechanisms, eliminating recurrence and convolutions |
| **Vision Transformer (ViT)** | Dosovitskiy et al. (Google) | 2020 | [arxiv:2010.11929](https://arxiv.org/abs/2010.11929) | Applied pure transformers to image recognition by treating images as sequences of patches |
| **Swin Transformer** | Liu et al. (Microsoft) | 2021 | [ICCV 2021](https://arxiv.org/abs/2103.14030) | Hierarchical vision transformer using shifted windows for efficient multi-scale feature learning |

---

## Object Detection

| Paper | Authors | Year | ArXiv/Link | Key Contribution |
|-------|---------|------|------------|------------------|
| **DETR** (End-to-End Object Detection with Transformers) | Carion et al. (Facebook AI) | 2020 | [arxiv:2005.12872](https://arxiv.org/abs/2005.12872) | First transformer-based object detector using set-based global loss and bipartite matching, eliminating NMS |
| **Deformable DETR** | Zhu et al. | 2021 | [ICLR 2021](https://arxiv.org/abs/2010.04159) | Improved DETR with deformable attention modules for faster convergence (10× less training epochs) and better small object detection |

---

## 3D Perception & BEV

| Paper | Authors | Year | ArXiv/Link | Key Contribution |
|-------|---------|------|------------|------------------|
| **BEVFormer** | Li et al. | 2022 | [arxiv:2203.17270](https://arxiv.org/abs/2203.17270) | Spatiotemporal transformer for unified BEV representation learning from multi-camera images with spatial cross-attention and temporal self-attention |
| **BEVDet** | Huang et al. | 2021 | [arxiv:2112.11790](https://arxiv.org/abs/2112.11790) | High-performance multi-camera 3D object detection in BEV with efficient paradigm achieving 9.2× faster inference than FCOS3D |

---

## Backbone Architectures

### CNN Backbones

| Paper | Authors | Year | ArXiv/Link | Key Contribution |
|-------|---------|------|------------|------------------|
| **ResNet** (Deep Residual Learning) | He et al. (Microsoft) | 2015 | [arxiv:1512.03385](https://arxiv.org/abs/1512.03385) | Introduced skip connections/residual blocks enabling training of very deep networks (152+ layers) |
| **EfficientNet** | Tan & Le (Google) | 2019 | [arxiv:1905.11946](https://arxiv.org/abs/1905.11946) | Compound scaling method that uniformly scales network width, depth, and resolution |
| **ConvNeXt** | Liu et al. (Facebook AI) | 2022 | [arxiv:2201.03545](https://arxiv.org/abs/2201.03545) | Modernized ConvNet architecture matching or exceeding transformer performance |
| **RegNet** | Radosavovic et al. (Facebook AI) | 2020 | [arxiv:2003.13678](https://arxiv.org/abs/2003.13678) | Design space exploration for network architecture design |

### Transformer Backbones

| Paper | Authors | Year | ArXiv/Link | Key Contribution |
|-------|---------|------|------------|------------------|
| **ViT** (Vision Transformer) | Dosovitskiy et al. | 2020 | [arxiv:2010.11929](https://arxiv.org/abs/2010.11929) | Pure transformer backbone for vision tasks |
| **Swin Transformer** | Liu et al. | 2021 | [arxiv:2103.14030](https://arxiv.org/abs/2103.14030) | Hierarchical transformer with shifted windows |
| **DeiT** (Data-efficient Image Transformers) | Touvron et al. (Facebook AI) | 2021 | [arxiv:2012.12877](https://arxiv.org/abs/2012.12877) | Training ViTs efficiently with distillation |

---

# Self-Supervised Learning Techniques

## Core Self-Supervised Learning Methods

| Paper | Authors | Year | ArXiv/Link | Key Contribution |
|-------|---------|------|------------|------------------|
| **DINO** (Self-Distillation with No Labels) | Caron et al. (Meta AI) | 2021 | [arxiv:2104.14294](https://arxiv.org/abs/2104.14294) | Self-distillation framework for ViTs without labels; produces explicit semantic information; 80.1% ImageNet top-1 |
| **DINOv2** (Learning Robust Visual Features without Supervision) | Oquab et al. (Meta AI) | 2023 | [arxiv:2304.07193](https://arxiv.org/abs/2304.07193) | Foundation model for all-purpose visual features; 1B ViT parameters; surpasses OpenCLIP on multiple benchmarks |
| **Momentum Contrast (MoCo)** | He et al. (FAIR, Facebook) | 2020 | [arxiv:1911.05722](https://arxiv.org/abs/1911.05722) | Dynamic dictionary with queue and momentum encoder; contrastive learning; outperforms supervised pre-training on 7 detection/segmentation tasks |
| **SimCLR** (A Simple Framework for Contrastive Learning of Visual Representations) | Chen et al. (Google) | 2020 | [arxiv:2002.05709](https://arxiv.org/abs/2002.05709) | Large batch contrastive learning; simple and effective; demonstrates importance of scale and augmentation |
| **BYOL** (Bootstrap Your Own Latent) | Grill et al. (DeepMind) | 2020 | [arxiv:2006.07733](https://arxiv.org/abs/2006.07733) | Contrastive learning without negative pairs; momentum encoder and predictor network; competitive with supervised learning |
| **SwAV** (Unsupervised Learning of Visual Features by Contrasting Cluster Assignments) | Caron et al. (Meta AI) | 2020 | [arxiv:2006.09882](https://arxiv.org/abs/2006.09882) | Contrastive learning with online clustering; efficient on multiple GPUs; 75.3% ImageNet top-1 |
| **VICReg** (Variance-Invariance-Covariance Regularization) | Bardes et al. (Meta AI) | 2022 | [arxiv:2205.00559](https://arxiv.org/abs/2205.00559) | Avoids collapse through explicit regularization; variance, invariance, covariance terms; robust training without momentum encoder |
| **Barlow Twins** | Zbontar et al. (Meta AI) | 2021 | [arxiv:2103.03230](https://arxiv.org/abs/2103.03230) | Self-supervised learning by redundancy reduction; cross-correlation matrix objective; simple yet effective |
| **NNCLR** (Nearest-Neighbor Contrastive Learning of Visual Representations) | Dwibedi et al. (Google) | 2021 | [arxiv:2104.14548](https://arxiv.org/abs/2104.14548) | Combines contrastive learning with nearest neighbors; improves convergence and downstream performance |
| **MAE** (Masked Autoencoders Are Scalable Vision Learners) | He et al. (Meta AI) | 2021 | [arxiv:2111.06377](https://arxiv.org/abs/2111.06377) | Masked image modeling; randomly masks patches and reconstructs; simple, scalable, and effective |

---

## Related Self-Supervised Approaches

| Paper | Authors | Year | ArXiv/Link | Key Contribution |
|-------|---------|------|------------|------------------|
| **Temporal Contrastive Learning** (Video SSL) | Wang et al. | 2021 | [arxiv:2008.02531](https://arxiv.org/abs/2008.02531) | Self-supervised learning from video; temporal contrast between frames |
| **Dense Contrastive Learning** | Wang et al. (FAIR) | 2021 | [arxiv:2011.09157](https://arxiv.org/abs/2011.09157) | Pixel-level or region-level contrastive learning; applicable to dense prediction tasks |
| **Self-Supervised Learning from Rotation** | Gidaris et al. | 2018 | [arxiv:1803.07728](https://arxiv.org/abs/1803.07728) | Predicting rotation angles as pretext task; early self-supervised learning method |
| **Contrastive Learning with Domain Knowledge** | Khosla et al. | 2020 | [arxiv:2004.11362](https://arxiv.org/abs/2004.11362) | Supervised contrastive learning; learns from labeled data more effectively than standard supervised learning |

---

## Complementary Techniques and Applications

### Vision Transformer Backbones for SSL

| Paper | Authors | Year | ArXiv/Link | Key Contribution |
|-------|---------|------|------------|------------------|
| **ViT** (Vision Transformer) | Dosovitskiy et al. | 2020 | [arxiv:2010.11929](https://arxiv.org/abs/2010.11929) | Pure transformer backbone for vision; enables effective SSL with DINO and DINOv2 |
| **Swin Transformer** | Liu et al. | 2021 | [arxiv:2103.14030](https://arxiv.org/abs/2103.14030) | Hierarchical transformer with shifted windows; efficient and effective |
| **DeiT** (Data-efficient Image Transformers) | Touvron et al. (Facebook AI) | 2021 | [arxiv:2012.12877](https://arxiv.org/abs/2012.12877) | Training ViTs efficiently with distillation; reduces computational requirements |

### Multi-Modal SSL

| Paper | Authors | Year | ArXiv/Link | Key Contribution |
|-------|---------|------|------------|------------------|
| **CLIP** (Contrastive Language-Image Pre-training) | Radford et al. (OpenAI) | 2021 | [arxiv:2103.00020](https://arxiv.org/abs/2103.00020) | Self-supervised learning from image-text pairs; zero-shot transfer capabilities |
| **BLIP** (Bootstrapping Language-Image Pre-training) | Li et al. | 2022 | [arxiv:2201.12086](https://arxiv.org/abs/2201.12086) | Unified vision-language model; multimodal understanding from image-text data |
| **DreamSim** (Learning New Image Similarities from Self-Supervised Learning) | Derrick et al. | 2023 | [arxiv:2402.19277](https://arxiv.org/abs/2402.19277) | Distills SSL model similarities for efficient image similarity learning |

### 3D and Point Cloud SSL

| Paper | Authors | Year | ArXiv/Link | Key Contribution |
|-------|---------|------|------------|------------------|
| **DepthContrast** (Self-Supervised Learning from Depth and Camera Motion) | Wang et al. | 2021 | [arxiv:2104.02960](https://arxiv.org/abs/2104.02960) | SSL from monocular videos using geometric constraints (depth and motion) |
| **DepthAnything** (Unleashing Unmanned Aerial Vehicle Potential for Single-Image Depth Estimation) | Yang et al. | 2024 | [arxiv:2401.10891](https://arxiv.org/abs/2401.10891) | SSL for monocular depth estimation; zero-shot transfer to diverse datasets |
| **MiDaS** (Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer) | Ranftl et al. | 2020 | [arxiv:1907.01341](https://arxiv.org/abs/1907.01341) | Zero-shot depth estimation through mixing datasets and SSL techniques |
| **PointContrast** (Self-Supervised Learning from Raw Point Clouds via Separating Mixed Shapes) | Xie et al. | 2021 | [arxiv:2101.02691](https://arxiv.org/abs/2101.02691) | Contrastive learning on 3D point clouds; geometric transforms as augmentations |

---

## Key SSL Concepts and Terminology

### Contrastive Learning
- **Positive Pairs**: Similar views of the same image (augmented versions)
- **Negative Pairs**: Views from different images
- **Objective**: Maximize similarity between positives, minimize between negatives

### Non-Contrastive Methods
- **Momentum Encoder**: Slowly updated copy of the main encoder (MoCo, BYOL)
- **Predictor Network**: Additional projection head (BYOL, VICReg)
- **Redundancy Reduction**: Prevents representation collapse (Barlow Twins, VICReg)

### Masked Image Modeling (MIM)
- **MAE**: Masks random patches (75%) and reconstructs from visible patches
- **BEiT**: Tokens are quantized; masked token prediction
- **Scalable**: Works well with large models and datasets

---



## Additional Resources

### Self-Supervised Learning
- **DINO** (Self-Distillation with No Labels): [arxiv:2104.14294](https://arxiv.org/abs/2104.14294)
- **DINOv2**: [arxiv:2304.07193](https://arxiv.org/abs/2304.07193)
- **Momentum Contrast (MoCo)**: [arxiv:1911.05722](https://arxiv.org/abs/1911.05722)

### Depth Estimation
- **Depth Anything**: [arxiv:2401.10891](https://arxiv.org/abs/2401.10891)
- **MiDaS** (Towards Robust Monocular Depth Estimation): [arxiv:1907.01341](https://arxiv.org/abs/1907.01341)

### Multi-Modal Learning
- **CLIP** (Contrastive Language-Image Pre-training): [arxiv:2103.00020](https://arxiv.org/abs/2103.00020)
- **BLIP** (Bootstrapping Language-Image Pre-training): [arxiv:2201.12086](https://arxiv.org/abs/2201.12086)

### Mixture of Experts
- **Mixture of Experts Explained** (Hugging Face): [Blog Post](https://huggingface.co/blog/moe)
- **Switch Transformers**: [arxiv:2101.03961](https://arxiv.org/abs/2101.03961)
- **Mixtral 8x7B** (Mistral AI): [Blog Post](https://mistral.ai/news/mixtral-of-experts/)

---


## Vision-Language Models

### VLM Architecture Insights
- Multi-modal fusion strategies
- Vision encoder vs. language encoder design choices
- Cross-modal attention patterns
- Token prediction strategies

### Training from Scratch
Document your learnings on:
- Custom tokenizer development
- Multi-token prediction approaches
- Chain-of-thought reasoning integration
- Dataset preparation


# Model Engineering

# Pippy - library
PiPPy: Pipeline Parallelism for PyTorch
Overview

PiPPy (Pipeline Parallelism for PyTorch) is an official PyTorch library that provides automated pipeline parallelism for distributed training and inference of large-scale deep learning models. It has been migrated into PyTorch as torch.distributed.pipelining and is currently in alpha state.


# 🚀 High-Performance Deep Learning Stack

This repository explores **end-to-end efficiency** for large deep learning models:
- Faster attention with **FlashAttention**
- Cheaper training with **FP16 / mixed precision**
- Lightweight **inference** via pruning & quantization (incl. QAT)
- Scalable distributed training & serving with **DeepSpeed**

---

## ⚡ FlashAttention

FlashAttention is an IO-aware exact attention kernel that computes softmax attention in **O(n²)** time but with dramatically reduced memory traffic.[web:50] Instead of materializing the full attention matrix, it tiles the computation to stay in high-bandwidth on-chip memory (SRAM), which:
- Reduces memory reads/writes and activation recomputation.[web:50]
- Enables longer sequence lengths on the same GPU memory budget.
- Often yields **2–4× speedups** over naïve attention implementations on modern hardware.[web:50]

**Why it matters for you:**
- Train **long-context transformers** without exploding memory.
- Reduce wall-clock time while keeping attention *exact* (no approximation).[web:50]

> ✅ Ideal for: LLMs, long-sequence vision transformers, BEV transformers, and any attention-heavy architecture.

---

## 🎯 FP16 / Mixed Precision Training

Mixed precision training uses **FP16/BF16 for most ops** and **FP32 for critical numerics** (e.g., loss scaling, master weights) to speed up training and reduce memory use.[web:57]

**Benefits:**
- **Throughput:** Better Tensor Core utilization → significant speedups on NVIDIA GPUs.[web:57]
- **Memory:** Almost halves activation and weight memory (where FP16 is used), enabling larger models or larger batch sizes.[web:57]
- **Compatibility:** Widely supported in modern DL frameworks via built‑in autocast and scaler utilities.[web:57]

**Typical recipe:**
- Enable autocast around forward passes.
- Maintain FP32 master weights.
- Use dynamic loss scaling to prevent underflow.

> ✅ Combine **FlashAttention + FP16** to maximize compute density for transformer workloads.[web:50][web:57]

---

## 🧠 Inference Optimization

High-quality models are often **over-parameterized for inference**. This section focuses on reducing latency and memory while preserving accuracy.

### 🌿 Pruning

Pruning removes redundant parameters (weights, channels, heads, or even layers) based on some importance criterion (e.g., magnitude, sensitivity).[web:47]

**Key ideas:**
- **Unstructured pruning:** Zero out individual weights (high sparsity, but needs sparse kernels to get speedups).[web:47]
- **Structured pruning:** Remove channels, blocks, or entire attention heads for predictable latency improvements.[web:47]
- **Pipeline:** Train → prune → fine-tune to recover accuracy.

> ✅ Use structured pruning for production because it maps better to real hardware speedups.

---

### 🧮 Quantization

Quantization compresses weights and activations from FP32/FP16 to lower bitwidths (e.g., INT8, INT4) to reduce memory footprint and improve throughput.[web:47]

**Static / Post-Training Quantization (PTQ):**
- Calibrate scales using a representative dataset.
- No or minimal re-training.
- Good for well-behaved models and moderate compression (e.g., INT8).[web:47]

**Dynamic Quantization:**
- Quantize weights, keep activations at higher precision on the fly.
- Useful for RNNs/transformers on CPU backends.[web:48]

> ✅ Quantization shines for **inference-only** deployments where small degradation in accuracy is acceptable for big speed/memory gains.[web:47]

---

### 🔁 Quantization-Aware Training (QAT)

QAT simulates quantization effects **during training**, allowing the model to adapt to quantization noise and recover accuracy.[web:47]

**Why QAT:**
- Significantly better accuracy than pure PTQ, especially at low bitwidths (INT4 and below).[web:47]
- Inserts fake-quantization ops in the graph to mimic real integer arithmetic.
- Allows training-time regularization toward quantization-friendly weights.

**Typical workflow:**
1. Start from a trained FP32/FP16 checkpoint.
2. Wrap modules with QAT stubs / fake-quant nodes.
3. Fine-tune under QAT for a few epochs.
4. Export to real quantized backend (e.g., ONNX, TensorRT, vendor runtimes).

> ✅ Recommended when you need **aggressive compression** but cannot afford large accuracy drops.

---

## 🧩 DeepSpeed Library

[DeepSpeed](https://github.com/microsoft/DeepSpeed) is a deep learning optimization library that provides **system-level and algorithmic optimizations** for training and inference of large models.[web:32]

**Core capabilities:**
- **ZeRO (Zero Redundancy Optimizer):** Shards optimizer states, gradients, and parameters across GPUs to scale to billions of parameters on limited memory.[web:32]
- **Pipeline + Tensor + Data Parallelism:** Full 3D parallelism for very large transformers.[web:32]
- **DeepSpeed-Inference:** Optimized inference kernels, quantization, and tensor-parallel runtimes for LLM serving.[web:32]
- **Memory & throughput optimizations:** Fused kernels, overlapping communication/computation, activation checkpointing.[web:32]

**Typical use cases:**
- Training multi‑billion parameter models on commodity clusters.
- Serving LLMs with **tensor-parallel + quantized** inference.
- Integrating with frameworks like Hugging Face Transformers for large-scale experiments.[web:32]

> ✅ Combine **DeepSpeed + FlashAttention + mixed precision + quantization** to get a production-grade stack for both training and serving large models.

---

## 📚 References
