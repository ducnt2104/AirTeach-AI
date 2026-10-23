"""
AirTeach AI - System Performance & Algorithm Benchmark Suite
=============================================================
Runs empirical profiling and measurements required by the project dossier:
  1. 1€ Filter throughput & per-point latency
  2. Ramer-Douglas-Peucker (RDP) point reduction ratio & latency
  3. Cubic Bézier spline (C1) fitting latency
  4. Geometric Shape Snapping latency & classification accuracy
  5. RMSE Jitter reduction efficiency (raw vs smoothed)
  6. Face detection (YuNet) & recognition (SFace) latency on CPU
  7. Gesture FSM state transition latency & debounce timing
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.stroke_beautifier import (
    OneEuroFilter2D,
    ramer_douglas_peucker,
    fit_cubic_bezier_spline,
    GeometricShapeSnapper,
    ShapeType,
    compute_stroke_rmse,
)
from src.core.hand_gesture_engine import HandGestureFSM, GestureState, LM


def benchmark_1euro_filter(iterations: int = 10_000) -> dict:
    """Benchmark 1€ Filter throughput and per-point latency."""
    filt = OneEuroFilter2D(min_cutoff=1.2, beta=0.35, d_cutoff=1.0)
    rng = np.random.default_rng(42)
    pts = rng.uniform(100.0, 900.0, size=(iterations, 2))

    t0 = time.perf_counter()
    for i in range(iterations):
        filt.filter(pts[i, 0], pts[i, 1], float(i * 16.66))
    elapsed_s = time.perf_counter() - t0

    latency_us = (elapsed_s / iterations) * 1e6
    throughput_pts_sec = iterations / elapsed_s
    return {
        "iterations": iterations,
        "elapsed_sec": round(elapsed_s, 4),
        "per_point_latency_us": round(latency_us, 2),
        "throughput_pts_sec": round(throughput_pts_sec, 0),
    }


def benchmark_rdp_simplification() -> dict:
    """Benchmark RDP point reduction ratio and execution time."""
    results = {}
    for n_pts in [100, 500, 1000]:
        t = np.linspace(0, 4 * np.pi, n_pts)
        pts = np.column_stack([t * 100, np.sin(t) * 100])
        # Add high-frequency noise
        rng = np.random.default_rng(42)
        pts += rng.normal(0, 1.5, size=pts.shape)

        t0 = time.perf_counter()
        simplified = ramer_douglas_peucker(pts, epsilon=4.0)
        elapsed_ms = (time.perf_counter() - t0) * 1e3

        reduction_pct = (1.0 - len(simplified) / n_pts) * 100
        results[f"{n_pts}_points"] = {
            "input_points": n_pts,
            "output_points": len(simplified),
            "reduction_percent": round(reduction_pct, 1),
            "latency_ms": round(elapsed_ms, 3),
        }
    return results


def benchmark_bezier_spline() -> dict:
    """Benchmark cubic Bézier spline fitting latency."""
    results = {}
    for n_knots in [50, 150, 300]:
        t = np.linspace(0, 2 * np.pi, n_knots)
        pts = np.column_stack([t * 150, np.cos(t) * 150])
        v = np.full(n_knots, 0.5)

        t0 = time.perf_counter()
        segments = fit_cubic_bezier_spline(pts, velocities=v)
        elapsed_ms = (time.perf_counter() - t0) * 1e3

        results[f"{n_knots}_knots"] = {
            "knots": n_knots,
            "segments_generated": len(segments),
            "latency_ms": round(elapsed_ms, 3),
        }
    return results


def benchmark_shape_snapper() -> dict:
    """Benchmark GeometricShapeSnapper classification accuracy and latency."""
    snapper = GeometricShapeSnapper()
    results = {}

    # 1. Straight Line
    line_pts = np.column_stack([np.linspace(100, 800, 120), np.full(120, 300)])
    t0 = time.perf_counter()
    line_snap = snapper.snap(line_pts)
    line_ms = (time.perf_counter() - t0) * 1e3
    results["line"] = {
        "detected": line_snap.shape_type.name,
        "expected": "LINE",
        "confidence": round(line_snap.confidence, 3),
        "angle_snapped": line_snap.params.get("angle_snapped", False),
        "latency_ms": round(line_ms, 3),
    }

    # 2. Circle
    theta = np.linspace(0, 2 * np.pi, 160)
    circle_pts = np.column_stack([500 + 150 * np.cos(theta), 500 + 150 * np.sin(theta)])
    t0 = time.perf_counter()
    circle_snap = snapper.snap(circle_pts)
    circle_ms = (time.perf_counter() - t0) * 1e3
    results["circle"] = {
        "detected": circle_snap.shape_type.name,
        "expected": "CIRCLE",
        "confidence": round(circle_snap.confidence, 3),
        "latency_ms": round(circle_ms, 3),
    }

    # 3. Rectangle
    p1 = np.column_stack([np.linspace(100, 400, 40), np.full(40, 100)])
    p2 = np.column_stack([np.full(40, 400), np.linspace(100, 300, 40)])
    p3 = np.column_stack([np.linspace(400, 100, 40), np.full(40, 300)])
    p4 = np.column_stack([np.full(40, 100), np.linspace(300, 100, 40)])
    rect_pts = np.vstack([p1, p2, p3, p4])
    t0 = time.perf_counter()
    rect_snap = snapper.snap(rect_pts)
    rect_ms = (time.perf_counter() - t0) * 1e3
    results["rectangle"] = {
        "detected": rect_snap.shape_type.name,
        "expected": "RECTANGLE",
        "confidence": round(rect_snap.confidence, 3),
        "latency_ms": round(rect_ms, 3),
    }

    return results


def benchmark_jitter_reduction_rmse() -> dict:
    """Benchmark RMSE jitter reduction comparing raw noisy stroke vs 1€ Filter + Bézier."""
    t = np.linspace(0, 3 * np.pi, 250)
    clean_curve = np.column_stack([t * 120 + 100, np.sin(t) * 80 + 300])

    # Synthesize realistic human hand tremor noise (Gaussian + high-freq jitter)
    rng = np.random.default_rng(101)
    noise = rng.normal(0, 4.5, size=clean_curve.shape)
    noisy_curve = clean_curve + noise

    # Filter with 1€ Filter
    filt = OneEuroFilter2D(min_cutoff=1.2, beta=0.35, d_cutoff=1.0)
    filtered_pts = []
    for i in range(len(noisy_curve)):
        fx, fy = filt.filter(noisy_curve[i, 0], noisy_curve[i, 1], float(i * 16.66))
        filtered_pts.append([fx, fy])
    filtered_pts = np.array(filtered_pts)

    # Simplify + fit Bézier
    simplified = ramer_douglas_peucker(filtered_pts, epsilon=2.5)
    segments = fit_cubic_bezier_spline(simplified)

    # Sample Bézier curve points
    bezier_pts = []
    for seg in segments:
        for u in np.linspace(0, 1, 10, endpoint=False):
            bezier_pts.append(seg.evaluate(u))
    bezier_pts.append(segments[-1].evaluate(1.0))
    bezier_pts = np.array(bezier_pts)

    raw_rmse = compute_stroke_rmse(clean_curve, noisy_curve)
    filtered_rmse = compute_stroke_rmse(clean_curve, filtered_pts)
    bezier_rmse = compute_stroke_rmse(clean_curve, bezier_pts)
    jitter_reduction_pct = (1.0 - filtered_rmse / raw_rmse) * 100

    return {
        "raw_noisy_rmse_px": round(raw_rmse, 3),
        "filtered_1euro_rmse_px": round(filtered_rmse, 3),
        "bezier_spline_rmse_px": round(bezier_rmse, 3),
        "jitter_reduction_percent": round(jitter_reduction_pct, 1),
    }


def benchmark_face_models() -> dict:
    """Benchmark OpenCV DNN YuNet & SFace models on CPU."""
    models_dir = _ROOT / "models"
    yunet_path = models_dir / "face_detection_yunet_2023mar.onnx"
    sface_path = models_dir / "face_recognition_sface_2021dec.onnx"

    results = {}
    import cv2
    if yunet_path.exists():
        detector = cv2.FaceDetectorYN.create(
            str(yunet_path), "", (320, 240), 0.7, 0.3, 5000, cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
        )
        fake_frame = np.full((240, 320, 3), 180, dtype=np.uint8)
        # Warmup
        detector.detect(fake_frame)
        t0 = time.perf_counter()
        iters = 20
        for _ in range(iters):
            detector.detect(fake_frame)
        elapsed_ms = ((time.perf_counter() - t0) / iters) * 1e3
        results["yunet_detection_ms"] = round(elapsed_ms, 2)
        results["yunet_fps"] = round(1000.0 / elapsed_ms, 1)

    if sface_path.exists():
        recognizer = cv2.FaceRecognizerSF.create(
            str(sface_path), "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
        )
        fake_face = np.full((112, 112, 3), 150, dtype=np.uint8)
        # Warmup
        recognizer.feature(fake_face)
        t0 = time.perf_counter()
        iters = 20
        for _ in range(iters):
            recognizer.feature(fake_face)
        elapsed_ms = ((time.perf_counter() - t0) / iters) * 1e3
        results["sface_feature_extraction_ms"] = round(elapsed_ms, 2)

    return results


def run_all_benchmarks() -> dict:
    """Execute complete benchmarking suite and return structured report."""
    print("=" * 70)
    print("   AIRTEACH AI - COMPREHENSIVE PERFORMANCE BENCHMARK SUITE")
    print("=" * 70)

    print("\n[1/6] Benchmarking 1€ Filter adaptive smoothing...")
    one_euro = benchmark_1euro_filter()
    print(f"  ✓ Latency: {one_euro['per_point_latency_us']} µs/point | Throughput: {one_euro['throughput_pts_sec']:,.0f} pts/sec")

    print("\n[2/6] Benchmarking RDP point simplification...")
    rdp = benchmark_rdp_simplification()
    for k, v in rdp.items():
        print(f"  ✓ {k}: {v['reduction_percent']}% reduction in {v['latency_ms']} ms")

    print("\n[3/6] Benchmarking C¹ Cubic Bézier Spline fitting...")
    bezier = benchmark_bezier_spline()
    for k, v in bezier.items():
        print(f"  ✓ {k}: {v['segments_generated']} segments fitted in {v['latency_ms']} ms")

    print("\n[4/6] Benchmarking Geometric Shape Snapper (Convex Hull & Circularity)...")
    snapper = benchmark_shape_snapper()
    for k, v in snapper.items():
        print(f"  ✓ {k.capitalize()}: detected as {v['detected']} (confidence {v['confidence']}) in {v['latency_ms']} ms")

    print("\n[5/6] Benchmarking RMSE Jitter Reduction Efficiency...")
    rmse = benchmark_jitter_reduction_rmse()
    print(f"  ✓ Raw RMSE: {rmse['raw_noisy_rmse_px']} px → Filtered RMSE: {rmse['filtered_1euro_rmse_px']} px")
    print(f"  ✓ Tremor Jitter Reduction: {rmse['jitter_reduction_percent']}%")

    print("\n[6/6] Benchmarking Face Detection (YuNet) & Recognition (SFace) on CPU...")
    face = benchmark_face_models()
    for k, v in face.items():
        print(f"  ✓ {k}: {v}")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "one_euro_filter": one_euro,
        "rdp_simplification": rdp,
        "cubic_bezier_spline": bezier,
        "geometric_shape_snapper": snapper,
        "jitter_reduction_rmse": rmse,
        "face_models": face,
    }

    # Save to data/benchmark_results.json
    out_file = _ROOT / "data" / "benchmark_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SUCCESS] Benchmark data written to {out_file}")

    return report


if __name__ == "__main__":
    run_all_benchmarks()
