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
