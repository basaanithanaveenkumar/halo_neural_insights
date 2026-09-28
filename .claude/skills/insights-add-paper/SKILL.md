---
name: insights-add-paper
description: Add a paper to the Halo Neural Insights reading tables (README.md and topic folders) with a verified title, authors, year, arXiv link and one-line contribution. Use when asked to add, fix or reorganise paper entries or citations in this knowledge base.
---

# Adding a paper entry

This repo is a curated knowledge base (Markdown only, no code). Paper tables use this format:

```markdown
| Paper | Authors | Year | ArXiv/Link | Key Contribution |
|-------|---------|------|------------|------------------|
| **Short name** (Full title) | First author et al. (Lab) | 2024 | [arxiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) | One sentence: what is new, and a headline number only if it is from the paper |
```

## Rules

1. **Verify before writing.** The arXiv id, title and first author must belong to the same
   paper. Check them against the arXiv abstract page. Previous entries had mismatches (see
   `docs/errata.md`), so never infer an id from memory.
2. Put the entry in the right section: Transformers & Attention, Object Detection, 3D
   Perception & BEV, Backbones, Self-Supervised Learning (core / related / multimodal /
   3D / JEPA), Mixture of Experts, VLA (`halo_vla/`), VLM (`vlm/`), Model engineering
   (`halo_model_engineering/`), 3D vision (`halo_3D/`).
3. Don't duplicate. Search first (`grep -n "arxiv:XXXX" -r .`). If a paper fits two
   sections, list it once and link to it from the other.
4. No chat-export artefacts: remove `[web:NN]`, `[attached_file:N]` and similar markers.
   Cite the real source instead.
5. Numbers (accuracy, speed-ups) must come from the paper's abstract or tables. Otherwise
   leave them out.
6. If the paper relates to a Hale/Halo repo (e.g. flow matching → HALE-WAM, MoE →
   HaleBlocks/HaloBlocks, VLA → Hale-VLA), add a "See also" link to that repo.

After adding entries, update `docs/reading-map.md` if a new topic appears, and check that
all Mermaid blocks still render.
