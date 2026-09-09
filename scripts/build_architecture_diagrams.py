#!/usr/bin/env python3
"""Draw original, editable architecture plates from the frozen Kaggle implementation.

This is documentation generation. A synthetic shape trace verifies the drawing;
no MRI is read and no training or performance measurement is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PALETTE = {
    "ink": "#193548", "muted": "#526979", "line": "#7A8D99",
    "blue": "#28699C", "blue_bg": "#EDF4FA", "teal": "#157D80",
    "teal_bg": "#EAF6F3", "purple": "#72578E", "purple_bg": "#F3EFF7",
    "amber": "#AA7025", "amber_bg": "#FCF5E9", "border": "#D0DBE3",
    "red": "#B74E5B", "red_bg": "#FCF0F1", "panel": "#FAFCFE",
}
REFERENCES = [
    {"title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
     "url": "https://arxiv.org/pdf/1505.04597", "figure": "Figure 1, PDF page 2",
     "use": "Scale annotations and explicit encoder-decoder skip connections; original artwork not reused."},
    {"title": "HeMIS: Hetero-Modal Image Segmentation",
     "url": "https://arxiv.org/pdf/1607.05194", "figure": "Figure 1, PDF page 3",
     "use": "Separate modality-feature and statistical-aggregation stages; original artwork not reused."},
    {"title": "Sebastian Raschka: The Big LLM Architecture Comparison",
     "url": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison",
     "figure": "Figure 1 and annotated component diagrams",
     "use": "Consistent module colors and annotated component detail; visual reference only, no LLM components or artwork reused."},
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trace_architecture():
    """Use the submitted configuration, verify source identity, trace actual tensors."""
    sys.path.insert(0, str(ROOT / "segmentation/src"))
    import torch
    from qmmf.config import ExperimentConfig
    from qmmf.models import build_model

    snapshot_path = ROOT / "audit/source_snapshots/segmentation_study_s42_v1.json"
    snapshot = json.loads(snapshot_path.read_text())
    sources = snapshot["sources"]
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    assert digest == snapshot["source_sha256"], "Frozen source digest differs"
    for rel, content in sources.items():
        assert (ROOT / rel).read_text() == content, f"Submitted source differs: {rel}"
    protocol_path = ROOT / "results/segmentation_study_s42/protocol_lock.json"
    resolved = json.loads(protocol_path.read_text())["configs"]["qmmf"]
    for seed in (43, 44):
        other = json.loads((ROOT / f"results/segmentation_study_s{seed}/protocol_lock.json").read_text())
        assert other["configs"]["qmmf"]["net"] == resolved["net"]
        assert other["configs"]["qmmf"]["loss"] == resolved["loss"]
    cfg = ExperimentConfig.from_dict(resolved)
    assert cfg.net.widths == (24, 48, 96, 160)
    assert cfg.data.context_slices == 5 and cfg.data.crop_size == (192, 192)
    assert cfg.net.quality_dim == 7 and cfg.net.modality_embed_dim == 16
    torch.set_num_threads(2)
    torch.manual_seed(20260909)
    model = build_model(cfg)
    traced = {}

    def hook(name):
        def record(module, args, out):
            value = out[0] if isinstance(out, tuple) else out
            traced[name] = list(value.shape)
        return record

    handles = []
    for name, module in [("shared_stem", model.stem), ("context", model.context),
                         ("head", model.head)]:
        handles.append(module.register_forward_hook(hook(name)))
    for prefix, modules in [("encoder", model.encoders), ("fusion", model.fusions),
                            ("decoder", model.decoders), ("aux", model.aux_heads)]:
        for i, module in enumerate(modules):
            handles.append(module.register_forward_hook(hook(f"{prefix}_{i}")))
    x = torch.zeros(1, 4, 5, 192, 192)
    a = torch.ones(1, 4)
    q = torch.zeros(1, 4, 7)
    with torch.no_grad():
        model.train()
        logits, aux = model(x, a, q, return_aux=True)
    for handle in handles:
        handle.remove()
    assert list(logits.shape) == [1, 3, 192, 192]
    assert [list(v.shape) for v in aux["deep_logits"]] == [[1, 3, 48, 48], [1, 3, 96, 96]]
    with torch.no_grad():
        model.eval()
        _, inference_aux = model(x, a, q, return_aux=True)
    assert inference_aux["deep_logits"] == []
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 1221465
    return {
        "submitted_source_sha256": digest, "source_files_verified": len(sources),
        "protocol": str(protocol_path.relative_to(ROOT)), "protocol_sha256": sha(protocol_path),
        "snapshot": str(snapshot_path.relative_to(ROOT)), "snapshot_sha256": sha(snapshot_path),
        "method": "Untrained synthetic CPU forward pass; no MRI, optimization or performance scores.",
        "synthetic_input": list(x.shape), "parameters": parameters,
        "tensor_trace_training": traced, "inference_auxiliary_heads": 0,
        "resolved_model_config": resolved["net"], "resolved_loss_config": resolved["loss"],
        "important_scope": "192 x 192 sizes describe training crops; inference preserves the full brain field of view.",
    }


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2, "text.color": PALETTE["ink"],
        "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "mri-qmmf-architecture-v1.2",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def canvas(width, height, inches):
    fig = plt.figure(figsize=inches)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width); ax.set_ylim(0, height); ax.set_axis_off()
    return fig, ax


def text(ax, x, y, value, size=8.2, color="ink", weight="normal", ha="center", va="center", **kwargs):
    return ax.text(x, y, value, fontsize=size, color=PALETTE.get(color, color),
                   weight=weight, ha=ha, va=va, linespacing=1.32, **kwargs)


def box(ax, x, y, w, h, label="", color="blue", size=8.2, dashed=False, weight="normal"):
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.085", linewidth=0.85,
        facecolor=PALETTE[f"{color}_bg"], edgecolor=PALETTE[color],
        linestyle=(0, (4, 2.5)) if dashed else "solid", zorder=3)
    ax.add_patch(patch)
    if label:
        item=text(ax, x+w/2, y+h/2, label, size=size, weight=weight, zorder=4)
        item._mri_container=patch
    return patch


def arrow(ax, start, end, color="ink", dashed=False, connection="arc3,rad=0", lw=1.0):
    p = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
        linewidth=lw, color=PALETTE.get(color,color),
        linestyle=(0,(4,2.5)) if dashed else "solid",
        connectionstyle=connection, shrinkA=1.5, shrinkB=1.5, zorder=2)
    ax.add_patch(p)


def route(ax, points, color="ink", dashed=False, lw=1.0):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0],end[0]], [start[1],end[1]], color=PALETTE[color],
                linewidth=lw, linestyle=(0,(4,2.5)) if dashed else "solid", zorder=2)
    arrow(ax, points[-2], points[-1], color, dashed, lw=lw)


def panel(ax, x, y, w, h, letter, title, subtitle=""):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=.14",
        facecolor=PALETTE["panel"], edgecolor=PALETTE["border"], linewidth=.8,zorder=0))
    text(ax,x+.24,y+h-.34,letter,10.2,weight="bold",ha="left")
    text(ax,x+.72,y+h-.34,title,9.1,weight="bold",ha="left")
    if subtitle:
        text(ax,x+.26,y+h-.72,subtitle,7.5,color="muted",ha="left")


def brain_icon(ax, cx, cy, radius=.38, regions=False):
    """Original vector icon: schematic anatomy, never an MRI or prediction."""
    for sign in (-1,1):
        vertices = np.array([[0,.92], [.43,1.10], [.9,.75], [.84,.18],
                             [1.05,-.28], [.56,-1.02], [.05,-.82],
                             [.0,-.12], [.03,.55], [0,.92]])
        vertices[:,0] *= sign
        vertices = vertices*radius+[cx,cy]
        codes=[MplPath.MOVETO]+[MplPath.CURVE4]*9
        ax.add_patch(PathPatch(MplPath(vertices,codes),facecolor="#DBE5EC",
                              edgecolor="#718694",linewidth=.65,zorder=5))
        for offset, sy in [(.25,.50),(.39,.02),(.30,-.47)]:
            verts=np.array([[.12,sy],[.27,sy+.13],[.55,sy+.15],[.66,sy-.06]])
            verts[:,0]*=sign
            ax.add_patch(PathPatch(MplPath(verts*radius+[cx,cy],
                [MplPath.MOVETO]+[MplPath.CURVE4]*3),fill=False,edgecolor="#A4B4C0",linewidth=.6,zorder=6))
    if regions:
        shapes={"WT":(.24,-.13,.75,.92,"teal"),"TC":(.28,-.15,.48,.64,"purple"),"ET":(.28,-.16,.23,.31,"amber")}
        for dx,dy,w,h,c in [shapes[regions]]:
            ax.add_patch(Ellipse((cx+dx*radius,cy+dy*radius),w*radius,h*radius,
                                facecolor=PALETTE[c],edgecolor="white",linewidth=.5,zorder=7))


def feature_stack(ax, x, y, w, h):
    for i,color in reversed(list(enumerate(["#DCEBF7","#E2F1EC","#ECE5F4","#F6ECD9"]))):
        ax.add_patch(FancyBboxPatch((x+i*.045,y+i*.045),w,h,
            boxstyle="round,pad=0,rounding_size=.045",facecolor=color,
            edgecolor="#8BA1B3",linewidth=.65,zorder=2))


def fusion_panel(ax, x, y, w, h, standalone=False):
    """Layout in a fixed 10.4 x 4.6 coordinate system."""
    # Main and enlarged exports have the same logical content and distinct scale.
    old = ax.transData
    # A child axes preserves reusable geometry without raster embedding.
    fig=ax.figure
    a=fig.add_axes([0,0,1,1],label=f"fusion-{len(fig.axes)}")
    # Use figure bounds obtained from parent data coordinates.
    lo=fig.transFigure.inverted().transform(old.transform((x,y)))
    hi=fig.transFigure.inverted().transform(old.transform((x+w,y+h)))
    a.set_position([lo[0],lo[1],hi[0]-lo[0],hi[1]-lo[1]])
    a.set_xlim(0,10.4);a.set_ylim(0,4.6);a.set_axis_off();a.patch.set_alpha(0)
    fs=9.6 if standalone else 7.6
    box(a,.28,3.08,2.13,.80,"Pooled feature\n"+r"$\mathrm{GAP}(F_m^s)$",size=fs)
    box(a,2.85,3.08,2.96,.80,r"$[\mathrm{GAP}(F_m^s),\,q_m,\,e_m]$"+"\nq: 7 values; e: 16 values",color="purple",size=fs)
    box(a,6.23,3.08,1.52,.80,"MLP gate\n32 → 1",color="purple",size=fs)
    box(a,8.20,3.08,1.91,.80,"Masked softmax\n"+r"$\alpha_m^s;\ a_m$",color="teal",size=fs)
    arrow(a,(2.41,3.48),(2.85,3.48));arrow(a,(5.81,3.48),(6.23,3.48))
    arrow(a,(7.75,3.48),(8.20,3.48))
    text(a,5.2,4.23,"GAP drives the weights; all statistics use the unpooled feature tensors F.",fs-.35,color="muted")
    for center in (1.75,5.22):
        route(a,[(9.15,3.08),(9.15,2.73),(center,2.73),(center,2.35)],color="teal")
    text(a,8.58,2.60,"a only",fs-.5,color="amber",ha="right")
    arrow(a,(8.85,2.62),(8.85,2.35),color="amber")
    box(a,.28,1.32,2.96,1.03,"Weighted mean\n"+r"$\mu=\sum_m\alpha_m F_m$",color="teal",size=fs)
    box(a,3.74,1.32,2.96,1.03,"Weighted dispersion\n"+r"$\sigma=\sqrt{\sum_m\alpha_m(F_m-\mu)^2+\epsilon}$",color="purple",size=fs-.5)
    box(a,7.19,1.32,2.92,1.03,"Available-feature max\n"+r"$r=\max_{m:a_m=1}\,F_m$",color="amber",size=fs-.35)
    for center in (1.75,5.22,8.65):
        route(a,[(center,1.32),(center,1.04),(5.22,1.04),(5.22,.84)],color="line")
    box(a,2.12,.15,6.20,.69,r"$[\mu,\sigma,r]$"+"  →  1×1 convolution  →  GroupNorm  →  GELU  →  "+r"$Z^s$",color="teal",size=fs-.3)
    return a


def overview(trace):
    fig, ax=canvas(18,13.7,(9,6.85))
    text(ax,.40,13.29,"QMMF-Net",14.3,weight="bold",ha="left")
    text(ax,3.60,13.29,"Quality-conditioned masked moment fusion for MRI segmentation",10.0,ha="left")
    text(ax,.42,12.87,"IMPLEMENTED ARCHITECTURE  •  2.5D context  •  1,221,465 parameters  •  source-verified tensor shapes",7.7,color="muted",ha="left")
    panel(ax,.32,5.88,17.36,6.57,"A","Four-scale segmentation network",
          "Shared encoder weights across T1, T1ce, T2 and FLAIR; fusion occurs independently at every scale.")
    # Column labels and four equal-height scales keep the skip topology explicit.
    for x,label in [(1.86,"MRI INPUT"),(5.17,"SHARED ENCODER"),(8.07,"MASKED FUSION"),(11.34,"DECODER"),(15.33,"REGION OUTPUTS")]:
        text(ax,x,11.23,label,7.8,color="muted",weight="bold")
    ys=[10.35,9.16,7.97,6.78]
    # Illustrative MRI cards; their origin is explicit in the diagram and caption.
    for i,(label,color) in enumerate([("T1","blue"),("T1ce","teal"),("T2","purple"),("FLAIR","amber")]):
        x=.64+(i%2)*1.37; y=9.0-(i//2)*1.43
        box(ax,x,y,1.12,1.21,color=color)
        brain_icon(ax,x+.56,y+.69,.37)
        text(ax,x+.56,y+.19,label,8.0,weight="bold",zorder=8)
    text(ax,1.86,10.67,"5 slices per modality",7.4)
    text(ax,.66,7.17,"Schematic\ninputs",6.9,color="muted",ha="left")
    box(ax,.65,6.19,2.41,.67,"All-four preprocessing\n→ availability mask a",color="teal",size=7.0)
    arrow(ax,(1.86,7.55),(1.86,6.87),color="teal")
    route(ax,[(3.08,6.53),(3.49,6.53),(3.49,10.35),(4.02,10.35)])
    for i,(cy,channels) in enumerate(zip(ys,[24,48,96,160])):
        size=192//2**i
        feature_stack(ax,4.02,cy-.42,2.19,.82)
        label="5×3×3 conv + FiLM\n3×3 refinement" if i==0 else "Average pool ↓2\n2 × 3×3 conv"
        text(ax,5.14,cy+.03,label,7.1,zorder=5)
        text(ax,5.16,cy-.55,f"4 × {channels} × {size} × {size}",7.0,color="blue")
        if i<3:
            route(ax,[(4.02,cy-.25),(3.77,cy-.25),(3.77,ys[i+1]+.22),(4.02,ys[i+1]+.22)],color="blue")
        arrow(ax,(6.37,cy),(7.17,cy),color="blue")
        box(ax,7.17,cy-.39,1.81,.78,f"QMMF  ·  scale {i+1}\n"+r"$\mu\,\oplus\,\sigma\,\oplus\,r$",color="teal",size=7.9)
        text(ax,8.07,cy-.57,f"Z{i+1}: {channels} × {size} × {size}",6.9,color="teal")
        if i<3:
            arrow(ax,(8.98,cy),(10.30,cy),color="teal")
            box(ax,10.30,cy-.39,2.12,.78,"↑2; concatenate skip\n"+f"2 × Conv → {channels} ch.",color="blue",size=7.0)
            if i<2:
                arrow(ax,(11.36,ys[i+1]+.42),(11.36,cy-.42),color="blue")
    box(ax,9.65,6.30,3.43,.92,"2 × residual context block\nDW 1×7 → 7×1; pointwise MLP",color="purple",size=7.4)
    arrow(ax,(8.98,ys[-1]),(9.65,ys[-1]),color="teal")
    arrow(ax,(11.36,7.22),(11.36,ys[2]-.42),color="purple")
    text(ax,9.65,6.01,"Fusion uses q[7], e[16], a (panel B).",6.9,color="muted",ha="left")
    # Three overlapping outputs and exact native-volume reconstruction.
    arrow(ax,(12.42,ys[0]),(13.25,ys[0]))
    box(ax,13.25,9.92,3.99,.86,"1×1 head → 3 logits → sigmoid\nReconstruct; project ET ⊆ TC ⊆ WT",color="teal",size=7.15)
    for cx,name,color in [(13.96,"WT","teal"),(15.27,"TC","purple"),(16.56,"ET","amber")]:
        brain_icon(ax,cx,9.02,.44,regions=name)
        text(ax,cx,8.42,name,8.0,color=color,weight="bold")
    arrow(ax,(15.25,9.92),(15.25,9.59),color="teal")
    text(ax,15.25,8.03,"Nested regions shown schematically\nNative-volume masks at threshold 0.5",7.0,color="muted")
    # Auxiliary heads are drawn separately and dashed, training only.
    route(ax,[(12.42,ys[1]),(12.85,ys[1]),(12.85,7.28),(13.58,7.28)],color="amber",dashed=True)
    route(ax,[(12.42,ys[2]),(12.66,ys[2]),(12.66,6.56),(13.58,6.56)],color="amber",dashed=True)
    box(ax,13.58,6.18,3.66,1.44,"TRAINING ONLY\n96² × 3 auxiliary head: 0.25\n48² × 3 auxiliary head: 0.50\nSee panel C for the objective",color="amber",size=7.35,dashed=True)
    panel(ax,.32,.70,10.51,4.82,"B","Fusion module at one encoder scale")
    fusion_panel(ax,.44,.81,10.26,3.89)
    panel(ax,11.12,.70,6.56,4.82,"C","Teacher–student training")
    box(ax,11.42,3.78,2.36,.83,"EMA teacher\nAll four modalities",color="purple",size=7.8,dashed=True)
    box(ax,15.0,3.78,2.36,.83,"Student θ\nSampled subset",color="blue",size=7.6)
    arrow(ax,(15.0,4.20),(13.78,4.20),color="amber",dashed=True)
    text(ax,14.39,4.61,"EMA update",6.8,color="amber")
    route(ax,[(12.6,3.78),(12.6,3.14),(13.23,3.14)],color="purple",dashed=True)
    route(ax,[(16.18,3.78),(16.18,3.14),(15.72,3.14)],color="blue")
    box(ax,13.23,2.70,2.49,.88,"Consistency MSE\nTeacher stop-gradient\nconfidence ≥ 0.90",color="purple",size=6.9)
    text(ax,11.48,2.53,"After 18 epochs",7.1,color="muted",ha="left")
    box(ax,11.44,1.09,5.91,1.19,"Supervised: 0.6 Dice + 0.4 BCE\n+ 0.10 boundary + 0.05 nesting\n+ 0.20 consistency + auxiliary supervision",color="amber",size=7.9)
    arrow(ax,(14.46,2.70),(14.46,2.28),color="purple",dashed=True)
    route(ax,[(17.36,4.16),(17.52,4.16),(17.52,1.70),(17.35,1.70)],color="blue")
    text(ax,14.40,.88,"Reference WT / TC / ET masks supervise the student.",6.6,color="muted")
    text(ax,.40,.32,"Training crop shapes: H = W = 192; inference retains the full brain field of view. Solid: prediction path. Dashed: training only.",7.1,color="muted",ha="left")
    return fig


def fusion_detail():
    fig, ax=canvas(12,7.1,(7.5,4.44))
    text(ax,.35,6.70,"QMMF fusion: descriptors, availability and feature statistics",12.0,weight="bold",ha="left")
    text(ax,.37,6.25,"One independent block per encoder scale; weights are shared across modalities within each gate.",8.9,color="muted",ha="left")
    fusion_panel(ax,.24,1.36,11.52,4.72,standalone=True)
    text(ax,.38,1.04,r"$\alpha_m = \frac{a_m\exp(\ell_m)}{\sum_j a_j\exp(\ell_j)}$"+r"    with    $\ell_m=g([\mathrm{GAP}(F_m),q_m,e_m])$"+r"    and    $\epsilon=10^{-6}$",10.0,ha="left")
    text(ax,.38,.51,"No quality: omit q only. Shuffled quality: use a different training group's descriptors.\nMatched moments: equal available weights, mean + dispersion, widened bottleneck; retains auxiliary heads.",8.8,color="muted",ha="left")
    return fig


def validate_text_bounds(fig):
    fig.canvas.draw()
    # Fit labels to their vector containers; reject layouts requiring tiny type.
    for ax in fig.axes:
        for item in ax.texts:
            patch=getattr(item,"_mri_container",None)
            if patch is None:continue
            container=patch.get_window_extent(fig.canvas.get_renderer())
            for _ in range(30):
                bb=item.get_window_extent(fig.canvas.get_renderer())
                if bb.width <= container.width*.95 and bb.height <= container.height*.89:
                    break
                item.set_fontsize(item.get_fontsize()-.1)
            if item.get_fontsize()<6.3:
                raise ValueError(f"Box label requires redesign: {item.get_text()}")
    fig.canvas.draw()
    bounds=fig.bbox
    violations=[]
    for ax in fig.axes:
        for item in ax.texts:
            b=item.get_window_extent(fig.canvas.get_renderer())
            if b.x0 < bounds.x0-1 or b.y0 < bounds.y0-1 or b.x1 > bounds.x1+1 or b.y1 > bounds.y1+1:
                violations.append(item.get_text())
    if violations:
        raise ValueError(f"Text extends beyond export canvas: {violations}")


def build(dest):
    dest.mkdir(parents=True,exist_ok=True)
    trace=trace_architecture();style()
    figures=[("qmmf_architecture",overview(trace)),("qmmf_fusion_detail",fusion_detail())]
    outputs=[]
    for stem,fig in figures:
        validate_text_bounds(fig)
        for extension in ("png","pdf","svg"):
            path=dest/f"{stem}.{extension}"
            metadata=({"CreationDate":None,"ModDate":None,"Creator":"MRI research architecture renderer"}
                      if extension=="pdf" else {"Date":None} if extension=="svg" else {})
            fig.savefig(path,dpi=400,metadata=metadata)
            if extension=="svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
            outputs.append({"file":path.name,"bytes":path.stat().st_size,"sha256":sha(path)})
    combined=dest/"architecture_plate.pdf"
    with PdfPages(combined,metadata={"Title":"QMMF architecture and fusion detail","CreationDate":None,"ModDate":None}) as pdf:
        for stem,fig in figures:pdf.savefig(fig)
    outputs.append({"file":combined.name,"bytes":combined.stat().st_size,"sha256":sha(combined)})
    record={"artifact_type":"Original explanatory vector schematics, separate from the 20 Kaggle result figures",
        "version":"1.2.0", "renderer":str(Path(__file__).relative_to(ROOT)),"renderer_sha256":sha(__file__),
        "trace":trace,"design_references":REFERENCES,
        "artwork":"Original geometry; schematic anatomy; no published artwork, MRI raster or prediction image reused.",
        "exports":outputs,"text_canvas_bounds":"passed"}
    (dest/"architecture_manifest.json").write_text(json.dumps(record,indent=2)+"\n")
    plt.close("all")
    print(json.dumps({"destination":str(dest),"exports":len(outputs),"source_identity":"verified",
                      "parameters":trace["parameters"],"training_auxiliary_heads":2,"inference_auxiliary_heads":0},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"docs/figures")
    build(parser.parse_args().output.resolve())
