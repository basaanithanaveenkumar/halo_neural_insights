---
name: insights-topic-notes
description: Write or expand a topic note in Halo Neural Insights (e.g. a VLA deep dive like the π0.5 note, MoE, DeepSeek, model engineering, 3D vision) with a consistent structure, Mermaid diagram and verified references, and link it into the knowledge map. Use when asked to document what was learned about a model, method or tool.
---

# Topic notes

Topic folders: `halo_vla/`, `vlm/`, `moe/`, `halo_model_engineering/`, `halo_3D/`.
One `README.md` (or `Readme.md`) per folder. Add a new folder for a new area.

## Structure

```markdown
## <Topic>

### TL;DR
Three sentences: what it is, why it matters, what to remember.

### How it works
Mechanism first, with one Mermaid diagram (flowchart TB/LR). Quote labels that contain
brackets or <tokens>: A["&lt;image&gt; tokens"].

### Key ideas
- Idea: explanation (with a citation)

### Trade-offs and pitfalls
What it costs, when it fails, and caveats seen in practice.

### Connection to the Hale/Halo repos
Which repo implements or could use this, and where (file path).

### References
- Author et al., *Title*, venue year. [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
```

## Rules

- Replace placeholders ("Add your own diagrams here", "Brief introduction to…") with real
  content, or delete the section. Don't leave template text.
- No `[web:NN]` / `[attached_file:N]` artefacts. Every claim cites a real source.
- Keep speculation marked as such ("we expect", "untested").
- Link the new note from `README.md` and `docs/reading-map.md`.
