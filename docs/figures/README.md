# Architecture figures

Original vector documentation for the QMMF implementation used in the completed Kaggle experiments. These explanatory figures are separate from the [20 verified Kaggle result figures](../../results/conference_figures/).

| Plate | PNG preview | Editable SVG | Vector PDF |
|---|---|---|---|
| Network, fusion and training | [PNG](qmmf_architecture.png) | [SVG](qmmf_architecture.svg) | [PDF](qmmf_architecture.pdf) |
| Enlarged fusion detail | [PNG](qmmf_fusion_detail.png) | [SVG](qmmf_fusion_detail.svg) | [PDF](qmmf_fusion_detail.pdf) |

![Implemented QMMF architecture](qmmf_architecture.png)

[Two-page plate](architecture_plate.pdf) · [complete captions, source map and visual references](../ARCHITECTURE.md) · [source/shape/export manifest](architecture_manifest.json) · [editable renderer](../../scripts/build_architecture_diagrams.py).

The anatomy and region symbols are schematic vector drawings. No published figure, MRI raster or prediction image is embedded. Visual references are U-Net Figure 1, HeMIS Figure 1 and Sebastian Raschka's annotated architecture comparisons; links and the precise design conventions are documented in the methods guide. The artwork does not imply endorsement or architectural novelty.

To rebuild without modifying the approved exports:

```bash
python scripts/build_architecture_diagrams.py --output runtime/architecture_rebuild
```

The generator verifies the frozen main-study source/configuration and traces tensor shapes on an untrained synthetic CPU input. It performs no MRI inference or training. Files are exported as 400-dpi PNG, vector PDF and SVG with editable labels. Inspect font sizes after reduction to the intended journal layout.
