---
name: hale-publish
description: Update and rebuild this knowledge base's public material — the survey paper (paper/), the GitHub Pages hub page for the Hale/Halo ecosystem (project-page/), the docs (docs/), the Mermaid maps and the blog (docs/blog/). Use after adding notes or when asked to refresh, write up or publish them.
---

# Publishing

| Material | Location |
|---|---|
| Survey paper | `paper/main.tex`, `paper/references.bib` (`make -C paper`) |
| Ecosystem hub page | `project-page/index.html`, deployed by `.github/workflows/pages.yml` on push to `main` (Settings → Pages → Source: GitHub Actions) |
| Docs | `docs/README.md`, `docs/reading-map.md` (Mermaid), `docs/ecosystem.md`, `docs/errata.md` |
| Blog | `docs/blog/YYYY-MM-DD-slug.md`, indexed in `docs/blog/README.md` |

Rules: the paper is a survey, so every factual claim needs a verified citation, and there are
no experimental results. When a Hale/Halo repo changes its headline facts (parameter
counts, test status, measured results), update `docs/ecosystem.md`, the hub page table and
paper Section 6 together. Keep the Mermaid source identical between docs and the page.
