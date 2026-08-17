"""ONNX export with a torch-vs-onnxruntime parity check.

Width is fixed (dynamic batch only): the research is clear that fixed/bucketed
widths export and run far more reliably under TensorRT than fully-dynamic width.
Bucket your inputs to a few fixed widths and export one model per bucket if needed.
"""
from __future__ import annotations

from pathlib import Path

import torch


def export_onnx(net, img_h: int, img_w: int, output: str = "svtrv2.onnx",
                opset: int = 18, check: bool = True) -> str:
    net = net.cpu().eval()
    dummy = torch.randn(1, 3, img_h, img_w)
    torch.onnx.export(
        net, dummy, str(output),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )
    size_mb = Path(output).stat().st_size / 1024 / 1024
    with torch.no_grad():
        out = net(dummy)
    print(f"  exported: {output} ({size_mb:.1f} MB) | input 1x3x{img_h}x{img_w} "
          f"| output 1x{out.shape[1]}x{out.shape[2]}")

    if check:
        try:
            import numpy as np
            import onnxruntime as ort
            sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
            ort_out = sess.run(None, {"image": dummy.numpy()})[0]
            diff = float(np.max(np.abs(out.numpy() - ort_out)))
            print(f"  parity:   torch vs onnxruntime max|diff| = {diff:.2e} "
                  f"[{'ok' if diff < 1e-3 else 'WARN'}]")
        except Exception as exc:  # pragma: no cover
            print(f"  parity:   skipped ({exc})")
    return str(output)


def run_export(ckpt_path: str, output: str = None, opset: int = 18,
               bin_name: str = "medium") -> str:
    """Export one MSR canvas.

    MSR means the model accepts three canvas sizes, but a fixed-width ONNX graph
    is what exports and runs reliably under TensorRT, so pick the bin you serve
    and export that one (or export all three and route by aspect ratio).
    """
    import torch

    from .config import MSR_BINS
    from .engine import load_checkpoint

    net, _ckpt = load_checkpoint(ckpt_path, torch.device("cpu"))
    bins = {str(b["name"]): b for b in MSR_BINS}
    if bin_name not in bins:
        raise ValueError(f"unknown MSR bin {bin_name!r}; expected one of {sorted(bins)}")
    b = bins[bin_name]
    output = output or f"{Path(ckpt_path).stem}-{bin_name}.onnx"
    return export_onnx(net, int(b["height"]), int(b["width"]), output, opset)
