# Manuscript workspace

Start with [the manuscript draft](../docs/MANUSCRIPT_DRAFT.md) and [writing guide](../docs/WRITING_GUIDE.md). Editable DOCX and review PDF versions are under `docs/docx/` and `docs/pdf/`. Use the [figure catalog](../docs/FIGURE_CATALOG.md) for all 20 figures, captions and source evidence.

The [eight table sets](tables/) contain CSV evidence and formatted Markdown/LaTeX exports. Insert the LaTeX fragments into the chosen venue's template with the `booktabs` package; they are table fragments, not a compiled venue template. Figures have PNG, PDF and SVG exports. Figure 19's SVG contains the unchanged raster montage.

The [architecture guide](../docs/ARCHITECTURE.md) and [two-page vector plate](../docs/figures/architecture_plate.pdf) describe the exact implemented network and its fusion module. Editable SVGs and 400-dpi PNGs are in [docs/figures](../docs/figures/). The main manuscript includes the overview; keep the detailed module view in Methods or a supplement according to space. These are original explanatory drawings, separate from the 20 Kaggle result figures.

[references.bib](references.bib) contains verified primary references. [claim_evidence.csv](claim_evidence.csv) maps manuscript statements to supporting results and their limitations. Additional related methods are discussed in the existing related-work report; no external published score is presented as a reproduced result.

Regenerate tables with `python scripts/build_paper_tables.py` from the repository root. That command also refreshes the marked table blocks in the manuscript draft. Original experiment sources and results remain frozen; generating figures and tables does not run a model.

The draft uses a venue-neutral structure. Supply real authorship, institutional declarations and the selected venue's formatting before submission. The current results do not support a quality-conditioning benefit, and the manuscript preserves that finding.
