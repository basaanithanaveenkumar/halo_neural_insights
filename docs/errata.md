# Errata

Problems found in the notes while writing the survey (September 2026). The notes themselves
have **not** been edited. Apply these corrections with the `insights-add-paper` skill after
checking each one against arXiv.

## Citation mismatches in `README.md`

| Entry as written | Problem | Suggested correction |
|---|---|---|
| **DreamSim**, "Derrick et al.", arXiv:2402.19277 | Wrong authors and id. DreamSim is Fu, Tamir, Sundaram et al., NeurIPS 2023 | *DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data*, [arXiv:2306.09344](https://arxiv.org/abs/2306.09344) |
| **DepthAnything**, "(Unleashing Unmanned Aerial Vehicle Potential …)" | The title is wrong. It is not about UAVs | *Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data*, Yang et al., CVPR 2024, arXiv:2401.10891 (the id is correct) |
| **PointContrast**, arXiv:2101.02691 | That id is DepthContrast (Zhang et al., *Self-Supervised Pretraining of 3D Features on any Point-Cloud*) | PointContrast: Xie et al., ECCV 2020, [arXiv:2007.10985](https://arxiv.org/abs/2007.10985) |
| **DepthContrast**, "(Self-Supervised Learning from Depth and Camera Motion)", arXiv:2104.02960 | The title and description don't match DepthContrast | Use arXiv:2101.02691 for DepthContrast (see above), and verify what 2104.02960 is before keeping it |
| **Contrastive Learning with Domain Knowledge**, Khosla et al., arXiv:2004.11362 | The paper is titled *Supervised Contrastive Learning* (NeurIPS 2020) | Rename to *Supervised Contrastive Learning (SupCon)* |
| **MAE**, year 2021 | arXiv 2021, but published at CVPR 2022 | Keep 2021 (arXiv) or write "CVPR 2022", consistently with the other rows |
| **Temporal Contrastive Learning**, Wang et al., arXiv:2008.02531 | Unverified: the title and authors look generic | Check on arXiv, then fix or remove |
| **VL-JEPA**, arXiv:2512.10942 | Unverified | Check on arXiv |

## Formatting artefacts

- `[web:NN]` markers (for example `[web:50]`, `[web:57]`) appear 25× in `README.md`, 30× in
  `halo_model_engineering/Readme.md` and 7× in `vlm/Deepseek.md`. `[attached_file:1]`
  appears 7× in `halo_vla/Readme.md`. These are chat-export citation stubs with no targets.
  Replace them with real references (e.g. Dao et al. 2022 for FlashAttention, Micikevicius et
  al. 2018 for mixed precision, Rajbhandari et al. 2020 for ZeRO).
- The model-engineering section is duplicated between `README.md` and
  `halo_model_engineering/Readme.md`. Keep it in one place and link to it.
- `vlm/Deepseek.md` is mostly template text ("Brief introduction to…", "[Add your own
  diagrams…]"). Fill it in or trim it.
