# Reading map

## 1. Topics in this knowledge base

```mermaid
mindmap
  root((Halo Neural Insights))
    Architectures
      Transformers and ViT
      Swin, DeiT
      CNN backbones
    Perception
      DETR, Deformable DETR
      BEVFormer, BEVDet
      3D vision: homography, DLT, RANSAC
    Self-supervised
      Contrastive: MoCo, SimCLR
      Non-contrastive: BYOL, Barlow Twins, VICReg
      Distillation: DINO, DINOv2
      Masked modeling: MAE
      JEPA family
    Foundation models
      VLM: CLIP, BLIP, DeepSeek
      VLA: π0.5, SmolVLA
      MoE: Switch, Mixtral
    Model engineering
      FlashAttention
      Mixed precision
      Pruning, PTQ, QAT
      DeepSpeed ZeRO, PiPPy
```

## 2. Self-supervised learning: how the families avoid collapse

```mermaid
flowchart TB
  SSL["Self-supervised representation learning"] --> C["Contrastive<br/>positives vs negatives"]
  SSL --> NC["Non-contrastive<br/>no negatives"]
  SSL --> MIM["Masked modeling<br/>reconstruct inputs"]
  SSL --> JEPA["Joint-embedding predictive<br/>predict latents"]
  C --> C1["MoCo: queue + momentum encoder"]
  C --> C2["SimCLR: large batch + augmentation"]
  NC --> N1["BYOL: predictor + momentum target"]
  NC --> N2["Barlow Twins / VICReg:<br/>decorrelate / variance terms"]
  NC --> N3["DINO / DINOv2: self-distillation,<br/>centering + sharpening"]
  MIM --> M1["MAE: mask 75% of patches, reconstruct pixels"]
  JEPA --> J1["I-JEPA (images) → V-JEPA (video) → V-JEPA 2 (world model + planning)"]
```

## 3. VLA design space (from the π0.5 and SmolVLA notes)

```mermaid
flowchart LR
  OBS["images + instruction + state"] --> VLM["VLM backbone<br/>(π0.5: SigLIP + Gemma; SmolVLA: SmolVLM2)"]
  VLM --> S2["System 2 (≈1 Hz):<br/>decompose task into subtasks"]
  S2 --> VLM
  VLM --> FAST["FAST tokens<br/>DCT + BPE of action chunks<br/>(fast to train, autoregressive decode)"]
  VLM --> FLOW["flow-matching action expert<br/>(continuous chunks, fast decode)"]
  KI["knowledge insulation:<br/>stop gradients from action expert into VLM"] -.-> FLOW
  FLOW --> RTC["real-time chunking:<br/>inpaint the next chunk while the current one executes"]
  RTC --> ROBOT["robot at ≈50 Hz"]
```

## 4. Efficiency stack (from the model-engineering notes)

```mermaid
flowchart TB
  subgraph Train
    FA["FlashAttention<br/>IO-aware exact attention"]
    MP["mixed precision<br/>bf16/fp16 + fp32 master weights"]
    Z["DeepSpeed ZeRO<br/>shard optimizer / grads / params"]
    PP["pipeline parallelism<br/>(PiPPy → torch.distributed.pipelining)"]
  end
  subgraph Deploy
    PR["pruning<br/>structured for real speed-ups"]
    PTQ["post-training quantization<br/>(static / dynamic)"]
    QAT["quantization-aware training<br/>for INT4 and below"]
  end
  Train --> CKPT["checkpoint"] --> Deploy
```
