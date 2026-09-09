#!/usr/bin/env python3
"""Index the verified Kaggle figures and assemble a PDF atlas without replotting."""
from pathlib import Path
import json
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]


def build():
    folder = ROOT / "results/conference_figures"
    manifest = json.loads((folder / "figure_manifest.json").read_text())
    verification = json.loads((ROOT / "audit/conference_visualizations_verification.json").read_text())
    assert manifest["figure_count"] == verification["figure_count"] == 20
    assert verification["status"] == "COMPLETE" and verification["executed_image_bytes_match_figure_files"]
    writer = PdfWriter()
    for record in manifest["figures"]:
        reader = PdfReader(folder / (record["stem"] + ".pdf")); assert len(reader.pages) == 1
        writer.add_page(reader.pages[0]); writer.add_outline_item(f"{record['number']:02d}: {record['title']}", record["number"] - 1)
    writer.add_metadata({"/Title": "MRI conference: 20 verified figures", "/Subject": "Atlas assembled from unchanged Kaggle PDF exports; full captions in FIGURE_CATALOG.md"})
    writer.write(folder / "conference_figure_atlas.pdf")
    lines = ["# Conference figure catalog", "",
        "All 20 figures were produced or preserved by [Kaggle CPU notebook version 1](https://www.kaggle.com/code/dasshovon/mri-conference-20-visualizations). The output bundle, plotted values, source files and all 20 image outputs in the executed notebook were independently checked. The original MRI montage's PNG/PDF bytes are unchanged.", "",
        "[Executed notebook](../results/conference_figures/03_conference_visualizations.executed.ipynb) · [20-page PDF atlas](../results/conference_figures/conference_figure_atlas.pdf) · [Source/output hashes](../results/conference_figures/figure_manifest.json) · [Kaggle verification](../audit/conference_visualizations_verification.json)", "",
        "The atlas is assembled locally from the downloaded PDF pages without replotting. This catalog supplies complete captions and interpretation limits. Chart PDF/SVG files have vector elements; Figure 19 is a preserved raster montage with [CC BY-SA 4.0 attribution](../results/conference_figures/ATTRIBUTION.md). Figure generation adds no training, inference or new hypothesis tests.", "",
        "| ID | Figure | Writing role |", "|---|---|---|"]
    for r in manifest["figures"]:
        lines.append(f"| {r['number']:02d} | {r['title']} | {r['suggested_placement']} |")
    for r in manifest["figures"]:
        stem = "../results/conference_figures/" + r["stem"]
        lines += ["", f"## Figure {r['number']:02d}. {r['title']}", "", f"![{r['title']}]({stem}.png)", "",
                  r["caption"], "", "**Interpretation:** " + r["interpretation"], "",
                  f"[PNG]({stem}.png) · [PDF]({stem}.pdf) · [SVG]({stem}.svg) · [Plotted CSV](../results/conference_figures/data/{r['stem']}.csv)", "",
                  "Sources: " + ", ".join(f"[{Path(p).name}](../{p})" for p in r["sources"]) + ".", ""]
    (ROOT / "docs/FIGURE_CATALOG.md").write_text("\n".join(lines))
    index = ["# Verified conference visualizations", "",
             "Twenty figures from Kaggle CPU notebook version 1. Start with the [full caption catalog](../../docs/FIGURE_CATALOG.md), [executed notebook](03_conference_visualizations.executed.ipynb) or [20-page atlas](conference_figure_atlas.pdf).", "",
             "The execution receipt and figure manifest preserve downloaded source/output hashes. The atlas and this index were assembled locally afterwards. Original outputs are retained, including the unchanged qualitative montage and its attribution.", "",
             "| Figure | PNG | PDF | SVG | Plotted values |", "|---|---|---|---|---|"]
    for r in manifest["figures"]:
        s = r["stem"]
        index.append(f"| {r['number']:02d}. {r['title']} | [PNG]({s}.png) | [PDF]({s}.pdf) | [SVG]({s}.svg) | [CSV](data/{s}.csv) |")
    (folder / "README.md").write_text("\n".join(index) + "\n")
    print(json.dumps({"catalog_figures": 20, "atlas_pages": len(PdfReader(folder / "conference_figure_atlas.pdf").pages)}))


if __name__ == "__main__": build()
