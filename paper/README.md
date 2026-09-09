# Manuscript workspace

Start with [the manuscript draft](../docs/MANUSCRIPT_DRAFT.md) and [writing guide](../docs/WRITING_GUIDE.md). Editable DOCX and review PDF versions are under `docs/docx/` and `docs/pdf/`. Use the [figure catalog](../docs/FIGURE_CATALOG.md) for all 20 figures, captions and source evidence.

The [eight table sets](tables/) contain CSV evidence and formatted Markdown/LaTeX exports. Insert the LaTeX fragments into the chosen venue's template with the `booktabs` package; they are table fragments, not a compiled venue template. Figures have PNG, PDF and SVG exports. Figure 19's SVG contains the unchanged raster montage.

[references.bib](references.bib) contains verified primary references. [claim_evidence.csv](claim_evidence.csv) maps manuscript statements to supporting results and their limitations. Additional related methods are discussed in the existing related-work report; no external published score is presented as a reproduced result.

Regenerate tables with `python scripts/build_paper_tables.py` from the repository root. That command also refreshes the marked table blocks in the manuscript draft. Original experiment sources and results remain frozen; generating figures and tables does not run a model.

The draft uses a venue-neutral structure. Supply real authorship, institutional declarations and the selected venue's formatting before submission. The current results do not support a quality-conditioning benefit, and the manuscript preserves that finding.
