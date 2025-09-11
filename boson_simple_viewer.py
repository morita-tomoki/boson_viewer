#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Very simple Boson viewer (OpenCV only, no pyuvc)

- Opens a camera with OpenCV (`cv2.VideoCapture`).
- If the frame is 16-bit (Y16), linearly scales to 8-bit for display.
- Optional colormap. Press 'q' to quit.

Note:
- This relies on your OpenCV backend (macOS: AVFoundation) being able to
  deliver frames from the Boson. Some environments expose Boson as Y16
  which OpenCV may or may not decode directly. If it fails or looks wrong,
  use the original `boson_uvc_viewer.py` which uses pyuvc to access Y16.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np


COLORMAPS = {
    "none": None,
    "inferno": cv2.COLORMAP_INFERNO,
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "hot": cv2.COLORMAP_HOT,
}


def to_u8_auto(img: np.ndarray) -> np.ndarray:
    """Convert any 1- or 3-channel image to displayable 8-bit.

    - uint16: auto min-max scale to 0..255
    - float:  clip [0,1] then *255
    - uint8:  pass through
    - 3ch color: assumes already 8-bit BGR
    """
    if img is None:
        raise ValueError("Empty frame")

    if img.dtype == np.uint16:
        # Scale per-frame for visibility. Simple and effective for viewing.
        lo = int(np.min(img))
        hi = int(np.max(img))
        if hi <= lo:
            hi = lo + 1
        scaled = (np.clip(img, lo, hi) - lo) / float(hi - lo)
        return (scaled * 255.0).astype(np.uint8)

    if np.issubdtype(img.dtype, np.floating):
        scaled = np.clip(img, 0.0, 1.0) * 255.0
        return scaled.astype(np.uint8)

    if img.dtype == np.uint8:
        return img

    # Fallback: try to normalize to uint8
    img_f = img.astype(np.float32)
    img_f -= img_f.min()
    m = img_f.max()
    if m > 0:
        img_f /= m
    return (img_f * 255.0).astype(np.uint8)


def apply_colormap(gray_u8: np.ndarray, cmap_name: str) -> np.ndarray:
    cmap = COLORMAPS.get(cmap_name)
    if cmap is None:
        return gray_u8
    # If single-channel, apply cmap; if already BGR, just return
    if gray_u8.ndim == 2 or (gray_u8.ndim == 3 and gray_u8.shape[2] == 1):
        return cv2.applyColorMap(gray_u8, cmap)
    return gray_u8


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Boson viewer (OpenCV only)")
    parser.add_argument("--device", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--title", type=str, default="Boson (Simple)", help="Window title")
    parser.add_argument(
        "--colormap",
        type=str,
        default="none",
        choices=list(COLORMAPS.keys()),
        help="Colormap for display",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        # Try explicitly with AVFoundation on macOS (harmless elsewhere)
        cap.release()
        cap = cv2.VideoCapture(args.device, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("[ERR] Failed to open camera. Try a different --device index.", file=sys.stderr)
        sys.exit(1)

    # Optional: ask backend not to convert to RGB (may be ignored)
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    except Exception:
        pass

    cv2.namedWindow(args.title, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Empty frame; continuing...")
                continue

            # If frame is single-channel (e.g., Y16/Y8), scale to 8-bit.
            if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
                gray = frame.squeeze()
                img8 = to_u8_auto(gray)
                vis = apply_colormap(img8, args.colormap)
            else:
                # Multi-channel frame (likely BGR). Preserve color by default.
                if args.colormap != "none":
                    # Convert to gray only to apply a colormap.
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    img8 = to_u8_auto(gray)
                    vis = apply_colormap(img8, args.colormap)
                else:
                    vis = frame  # keep original color (e.g., already pseudo-colored)

            cv2.imshow(args.title, vis)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
