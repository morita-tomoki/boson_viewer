#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boson Tuner Viewer: unified live viewer + on-the-fly camera tuning via SDK.

Controls high-impact settings for night maritime imaging:
  - Gain mode (HIGH/LOW/AUTO/DUAL)
  - FFC (Run, Mode: MANUAL/AUTO)
  - AGC mode + interactively set AGC ROI
  - Color LUT enable/disable and palette selection

Requirements:
  - OpenCV (`pip install opencv-python`)
  - pyserial (`pip install pyserial`)
  - SDK at ./SDK_USER_PERMISSIONS (or override with --sdk-path)

Usage examples:
  - python3 boson_tuner_viewer.py --device 0 --port /dev/cu.usbmodemXXXX
  - python3 boson_tuner_viewer.py --device 0 --port COM7 --title "Boson Tuner"

Keys:
  - q: quit

  Gain:
    - 1: HIGH  2: LOW  3: AUTO  4: DUAL  (cycle with 'g')

  FFC:
    - f: Run FFC now
    - m: Toggle FFC mode (AUTO/MANUAL)

  AGC:
    - a: Cycle AGC mode (NORMAL → AUTO_LINEAR → MANUAL → HOLD → THRESHOLD)
    - r: Start ROI selection (click-drag to set). ESC/right-click to cancel.
    - c: Clear ROI (full frame)

  LUT:
    - e: enable LUT    d: disable LUT
    - w: WHITEHOT  b: BLACKHOT  i: IRONBOW  r: RAINBOW
    - g: GRADEDFIRE h: HOTTEST  a: ARCTIC   l: LAVA   o: GLOBOW

Notes:
  - If your UVC stream is raw IR16 and colorized on the PC, camera-side LUT
    changes may not be visible. If the camera outputs colorized frames, changes
    will appear immediately.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import os
import time
from datetime import datetime


def open_capture(device_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device_index)
    if cap.isOpened():
        return cap
    # Fallback for macOS AVFoundation
    cap.release()
    cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
    return cap


def enum_simple_name(enum_val) -> str:
    try:
        name = enum_val.name
    except Exception:
        return str(enum_val)
    for prefix in ("FLR_COLORLUT_", "FLR_BOSON_", "FLR_AGC_", "FLR_SYSCTRL_", "FLR_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


class RoiSelector:
    def __init__(self, window: str):
        self.window = window
        self.dragging = False
        self.start = (0, 0)
        self.end = (0, 0)
        self.active = False
        self.last_selection: Optional[Tuple[int, int, int, int]] = None  # x0, y0, x1, y1

    def start_selection(self):
        self.active = True
        self.dragging = False
        self.last_selection = None

    def cancel(self):
        self.active = False
        self.dragging = False
        self.last_selection = None

    def get_rect(self) -> Optional[Tuple[int, int, int, int]]:
        if not self.last_selection:
            return None
        x0, y0, x1, y1 = self.last_selection
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def on_mouse(self, event, x, y, flags, param):
        if not self.active:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.end = (x, y)
            x0, y0 = self.start
            x1, y1 = self.end
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            self.last_selection = (x0, y0, x1, y1)
            self.active = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Boson tuner viewer (pySerial SDK)")
    parser.add_argument("--device", type=int, default=0, help="Camera index for OpenCV (default: 0)")
    parser.add_argument("--title", type=str, default="Boson Tuner", help="Window title")
    parser.add_argument("--port", type=str, required=True, help="Serial port for SDK (e.g., /dev/cu.usbmodemXXXX | COM7)")
    parser.add_argument("--baud", type=int, default=None, help="Baud rate (default per SDK: 921600)")
    parser.add_argument("--sdk-path", type=str, default="SDK_USER_PERMISSIONS", help="Path to SDK_USER_PERMISSIONS package")
    parser.add_argument("--no-overlay", action="store_true", help="Disable on-screen overlay text")
    parser.add_argument(
        "--no-camera-overlay",
        action="store_true",
        help="Disable camera-drawn overlays (spot meter, symbology, telemetry, isotherm) at startup",
    )
    parser.add_argument("--record-overlay", choices=["on","off"], default="on", help="Burn on-screen overlay into recording")
    parser.add_argument("--record-out", type=str, default=None, help="Output video path (default: captures/<timestamp>_dev<idx>.mp4)")
    parser.add_argument("--codec", type=str, default="mp4v", help="FourCC codec (default: mp4v; fallback MJPG if open fails)")
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback FPS if CAP reports 0")
    parser.add_argument("--record-size", type=str, default=None, help="Force recording size WxH (e.g., 640x512). Default: first raw frame size")
    args = parser.parse_args()

    # Import SDK
    sys.path.append(args.sdk_path)
    try:
        from SDK_USER_PERMISSIONS import CamAPI
        from SDK_USER_PERMISSIONS import (
            FLR_ENABLE_E,
            FLR_COLORLUT_ID_E,
            FLR_BOSON_GAINMODE_E,
            FLR_BOSON_FFCMODE_E,
            FLR_AGC_MODE_E,
            FLR_ROI_T,
        )
        from SDK_USER_PERMISSIONS import FLR_RESULT
    except Exception:
        print("[ERR] Failed to import SDK from:", args.sdk_path, file=sys.stderr)
        raise

    # Init SDK client over pySerial
    try:
        cam = CamAPI.pyClient(manualport=args.port, manualbaud=args.baud, useDll=False, ex=True)
    except Exception as e:
        print(f"[ERR] Failed to open SDK serial port '{args.port}': {e}", file=sys.stderr)
        sys.exit(1)

    # Optionally disable camera-side overlays that draw shapes/text (e.g., center box, scale bar)
    if args.no_camera_overlay:
        try:
            cam.spotMeterSetEnable(FLR_ENABLE_E.FLR_DISABLE)
            print("[INFO] Spot meter overlay disabled")
        except Exception:
            pass
        try:
            cam.symbologySetEnable(FLR_ENABLE_E.FLR_DISABLE)
            print("[INFO] Symbology overlay disabled")
        except Exception:
            pass
        try:
            cam.telemetrySetState(FLR_ENABLE_E.FLR_DISABLE)
            print("[INFO] Telemetry overlay disabled")
        except Exception:
            pass
        try:
            cam.isothermSetEnable(FLR_ENABLE_E.FLR_DISABLE)
            print("[INFO] Isotherm overlay disabled")
        except Exception:
            pass

    # Print camera identification (PN/SN) for debugging
    def _pn_to_str(pn_obj) -> str:
        try:
            arr = getattr(pn_obj, 'value', None)
            if arr is None:
                return str(pn_obj)
            b = bytes([(x or 0) & 0xFF for x in arr])
            b = b.split(b"\x00", 1)[0]
            return b.decode('ascii', errors='ignore').strip()
        except Exception:
            return str(pn_obj)

    try:
        cam_pn_obj = cam.bosonGetCameraPN()
        cam_sn = cam.bosonGetCameraSN()
        msg = f"[INFO] Camera PN: {_pn_to_str(cam_pn_obj)}  SN: {cam_sn}"
        try:
            sensor_pn_obj = cam.bosonGetSensorPN()
            msg += f"  SensorPN: {_pn_to_str(sensor_pn_obj)}"
        except Exception:
            pass
        try:
            # Best-effort: report DVO video size (may not reflect UVC)
            vals = cam.dvoGetClockInfo()  # ex=True → returns tuple of outputs (no return code)
            vcols = vals[8]
            vrows = vals[11]
            fps_hz = vals[18]
            msg += f"  DVO:{vcols}x{vrows}@{fps_hz:.2f}"
        except Exception:
            pass
        print(msg)
    except Exception:
        print("[WARN] Could not read camera PN/SN")

    # State cache
    gain_mode = None
    ffc_mode = None
    agc_mode = None
    lut_enabled = None
    lut_id = None
    roi_rect: Optional[Tuple[int, int, int, int]] = None  # x0,y0,x1,y1 in image coords
    # Native camera ROI extent (rows/cols) from camera coordinates
    native_roi = None
    native_w = None  # columns
    native_h = None  # rows

    # Query initial states (best-effort)
    try:
        gain_mode = cam.bosonGetGainMode()
    except Exception:
        pass
    try:
        ffc_mode = cam.bosonGetFFCMode()
    except Exception:
        pass
    try:
        agc_mode = cam.agcGetMode()
    except Exception:
        pass
    try:
        # Ensure LUT control is enabled to allow palette changes; ignore errors.
        cam.colorLutSetControl(FLR_ENABLE_E.FLR_ENABLE)
        lut_id = cam.colorLutGetId()
        lut_enabled = True
    except Exception:
        lut_enabled = None
        lut_id = None

    # Get current AGC ROI to learn native dimensions
    try:
        native_roi = cam.agcGetROI()
        native_h = int(native_roi.rowStop - native_roi.rowStart + 1)
        native_w = int(native_roi.colStop - native_roi.colStart + 1)
    except Exception:
        native_roi = None
        native_h = None
        native_w = None

    # Open camera capture
    cap = open_capture(args.device)
    if not cap.isOpened():
        print("[ERR] Failed to open camera. Try a different --device index.", file=sys.stderr)
        try:
            cam.Close()
        except Exception:
            pass
        sys.exit(1)

    cv2.namedWindow(args.title, cv2.WINDOW_NORMAL)
    selector = RoiSelector(args.title)
    cv2.setMouseCallback(args.title, selector.on_mouse)

    # Recording state
    recording = False
    rec_writer: Optional[cv2.VideoWriter] = None
    rec_start_time: Optional[float] = None
    burn_overlay = args.record_overlay == "on"
    record_size: Optional[Tuple[int, int]] = None  # (w,h)

    # Parse forced record size if provided
    if args.record_size:
        try:
            parts = args.record_size.lower().split('x')
            if len(parts) == 2:
                rw = int(parts[0]); rh = int(parts[1])
                if rw > 0 and rh > 0:
                    record_size = (rw, rh)
        except Exception:
            print("[WARN] Invalid --record-size; ignoring")

    def ensure_captures_dir():
        out_dir = os.path.join(os.getcwd(), "captures")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def to_bgr_u8(img):
        if img is None:
            return None
        if img.dtype == np.uint16:
            lo = int(np.min(img))
            hi = int(np.max(img))
            if hi <= lo:
                hi = lo + 1
            scaled = (np.clip(img, lo, hi) - lo) / float(hi - lo)
            u8 = (scaled * 255.0).astype(np.uint8)
            if u8.ndim == 2 or (u8.ndim == 3 and u8.shape[2] == 1):
                return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
            return u8
        if np.issubdtype(img.dtype, np.floating):
            u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            if u8.ndim == 2 or (u8.ndim == 3 and u8.shape[2] == 1):
                return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
            return u8
        # uint8 or others
        if img.dtype != np.uint8:
            img_f = img.astype(np.float32)
            img_f -= img_f.min()
            m = img_f.max()
            if m > 0:
                img_f /= m
            u8 = (img_f * 255.0).astype(np.uint8)
        else:
            u8 = img
        if u8.ndim == 2 or (u8.ndim == 3 and (u8.shape[2] == 1)):
            return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        return u8

    def start_recording(sample_frame):
        nonlocal recording, rec_writer, rec_start_time
        if recording:
            return True
        # Decide output path and codec
        out_path = args.record_out
        if not out_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = ensure_captures_dir()
            out_path = os.path.join(out_dir, f"{ts}_dev{args.device}.mp4")
        # FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = float(args.fps)
        # Decide size (lock to first raw frame unless forced)
        h_s, w_s = sample_frame.shape[:2]
        if record_size is None:
            record_size_local = (w_s, h_s)
        else:
            record_size_local = record_size
        # Try primary codec
        fourcc = cv2.VideoWriter_fourcc(*args.codec)
        writer = cv2.VideoWriter(out_path, fourcc, fps, record_size_local)
        if not writer.isOpened():
            # Fallback to MJPG AVI
            root, ext = os.path.splitext(out_path)
            out_path = root + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(out_path, fourcc, fps, record_size_local)
            if not writer.isOpened():
                print("[ERR] Failed to open VideoWriter for both primary and fallback codecs.")
                return False
            else:
                print(f"[WARN] Primary codec failed; recording with MJPG: {out_path}")
        else:
            print(f"[INFO] Recording to {out_path} @ {fps:.2f} FPS, size={record_size_local[0]}x{record_size_local[1]}")
        rec_writer = writer
        rec_start_time = time.time()
        recording = True
        return True

    def stop_recording():
        nonlocal recording, rec_writer, rec_start_time
        if rec_writer is not None:
            try:
                rec_writer.release()
            except Exception:
                pass
        rec_writer = None
        rec_start_time = None
        if recording:
            print("[INFO] Recording stopped")
        recording = False

    def apply_agc_roi_from_selection(frame_shape):
        nonlocal roi_rect
        if selector.get_rect() is None:
            return
        x0, y0, x1, y1 = selector.get_rect()
        h, w = frame_shape[:2]
        # Clamp to display bounds
        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w - 1))
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h - 1))
        if x1 <= x0 or y1 <= y0:
            return
        roi_rect = (x0, y0, x1, y1)
        # Map display coords to camera-native ROI if available
        try:
            if native_w is not None and native_h is not None and native_roi is not None:
                colStart_native = int(round(native_roi.colStart + (x0 / float(w)) * native_w))
                colStop_native  = int(round(native_roi.colStart + (x1 / float(w)) * native_w))
                rowStart_native = int(round(native_roi.rowStart + (y0 / float(h)) * native_h))
                rowStop_native  = int(round(native_roi.rowStart + (y1 / float(h)) * native_h))
                # Clamp to native bounds and ensure valid ordering
                colStart_native = max(int(native_roi.colStart), min(colStart_native, int(native_roi.colStop)))
                colStop_native  = max(int(native_roi.colStart), min(colStop_native,  int(native_roi.colStop)))
                rowStart_native = max(int(native_roi.rowStart), min(rowStart_native, int(native_roi.rowStop)))
                rowStop_native  = max(int(native_roi.rowStart), min(rowStop_native,  int(native_roi.rowStop)))
                if colStop_native <= colStart_native:
                    colStop_native = min(colStart_native + 1, int(native_roi.colStop))
                if rowStop_native <= rowStart_native:
                    rowStop_native = min(rowStart_native + 1, int(native_roi.rowStop))
                roi = FLR_ROI_T()
                roi.rowStart = rowStart_native
                roi.rowStop = rowStop_native
                roi.colStart = colStart_native
                roi.colStop = colStop_native
                cam.agcSetROI(roi)
                print(f"[INFO] AGC ROI set (native) rows[{roi.rowStart},{roi.rowStop}] cols[{roi.colStart},{roi.colStop}]")
            else:
                # Fallback: try using display coords directly
                roi = FLR_ROI_T()
                roi.rowStart = int(y0)
                roi.rowStop = int(y1)
                roi.colStart = int(x0)
                roi.colStop = int(x1)
                cam.agcSetROI(roi)
                print(f"[INFO] AGC ROI set (display) x[{x0},{x1}] y[{y0},{y1}]")
        except Exception as e:
            print("[ERR] agcSetROI failed:", e, file=sys.stderr)

    def clear_agc_roi(frame_shape):
        nonlocal roi_rect
        h, w = frame_shape[:2]
        try:
            roi = FLR_ROI_T()
            if native_roi is not None:
                roi.rowStart = int(native_roi.rowStart)
                roi.rowStop = int(native_roi.rowStop)
                roi.colStart = int(native_roi.colStart)
                roi.colStop = int(native_roi.colStop)
            else:
                roi.rowStart = 0
                roi.rowStop = int(h - 1)
                roi.colStart = 0
                roi.colStop = int(w - 1)
            cam.agcSetROI(roi)
            roi_rect = None
            print("[INFO] AGC ROI cleared (full frame)")
        except Exception as e:
            print("[ERR] clear ROI failed:", e, file=sys.stderr)

    def draw_overlay(img):
        if args.no_overlay:
            return
        y = 20
        dy = 18
        def put(line):
            nonlocal y
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            y += dy

        gm = enum_simple_name(gain_mode) if gain_mode is not None else "?"
        ffcm = enum_simple_name(ffc_mode) if ffc_mode is not None else "?"
        agcm = enum_simple_name(agc_mode) if agc_mode is not None else "?"
        lutn = enum_simple_name(lut_id) if lut_id is not None else "?"
        luts = "EN" if lut_enabled else ("DIS" if lut_enabled is not None else "?")

        put(f"Gain: {gm}   FFC: {ffcm}")
        put(f"AGC: {agcm}   LUT: {lutn} ({luts})")
        put("Keys: [q]quit [1/2/3/4]gain [g]cycle  [f]FFC [m]FFCmode  [a]AGCmode [r]ROI [c]clearROI  LUT:[e/d] + w,b,i,h,l,o,R,G,A  REC:[v] snap:[s]")

        # Draw ROI rectangle
        rect = selector.get_rect() or roi_rect
        if rect is not None:
            x0, y0, x1, y1 = rect
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 1)

        # Draw live selection rectangle if dragging
        if selector.active and selector.dragging:
            x0, y0 = selector.start
            x1, y1 = selector.end
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), 1)

        # Recording indicator
        if recording and rec_start_time is not None:
            elapsed = int(time.time() - rec_start_time)
            mm = elapsed // 60
            ss = elapsed % 60
            rec_text = f"[REC {mm:02d}:{ss:02d}]"
            cv2.putText(img, rec_text, (10, y + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Helper: set functions with state update
    def set_gain(mode):
        nonlocal gain_mode
        try:
            cam.bosonSetGainMode(mode)
            gain_mode = cam.bosonGetGainMode()
            print(f"[INFO] Gain mode set to {enum_simple_name(gain_mode)}")
        except Exception as e:
            print("[ERR] set gain failed:", e, file=sys.stderr)

    def cycle_gain():
        order = [
            FLR_BOSON_GAINMODE_E.FLR_BOSON_HIGH_GAIN,
            FLR_BOSON_GAINMODE_E.FLR_BOSON_LOW_GAIN,
            FLR_BOSON_GAINMODE_E.FLR_BOSON_AUTO_GAIN,
            FLR_BOSON_GAINMODE_E.FLR_BOSON_DUAL_GAIN,
        ]
        try:
            current = gain_mode
            if current in order:
                idx = (order.index(current) + 1) % len(order)
            else:
                idx = 0
            set_gain(order[idx])
        except Exception as e:
            print("[ERR] cycle gain failed:", e, file=sys.stderr)

    def run_ffc():
        try:
            cam.bosonRunFFC()
            print("[INFO] FFC triggered")
        except Exception as e:
            print("[ERR] FFC run failed:", e, file=sys.stderr)

    def toggle_ffc_mode():
        nonlocal ffc_mode
        try:
            # Toggle AUTO <-> MANUAL
            target = FLR_BOSON_FFCMODE_E.FLR_BOSON_AUTO_FFC
            if ffc_mode == FLR_BOSON_FFCMODE_E.FLR_BOSON_AUTO_FFC:
                target = FLR_BOSON_FFCMODE_E.FLR_BOSON_MANUAL_FFC
            cam.bosonSetFFCMode(target)
            ffc_mode = cam.bosonGetFFCMode()
            print(f"[INFO] FFC mode set to {enum_simple_name(ffc_mode)}")
        except Exception as e:
            print("[ERR] toggle FFC mode failed:", e, file=sys.stderr)

    def cycle_agc_mode():
        nonlocal agc_mode
        order = [
            FLR_AGC_MODE_E.FLR_AGC_MODE_NORMAL,
            FLR_AGC_MODE_E.FLR_AGC_MODE_AUTO_LINEAR,
            FLR_AGC_MODE_E.FLR_AGC_MODE_MANUAL,
            FLR_AGC_MODE_E.FLR_AGC_MODE_HOLD,
            FLR_AGC_MODE_E.FLR_AGC_MODE_THRESHOLD,
        ]
        current = agc_mode
        if current in order:
            idx = (order.index(current) + 1) % len(order)
        else:
            idx = 0
        attempted = 0
        while attempted < len(order):
            target = order[idx]
            try:
                cam.agcSetMode(target)
                agc_mode = cam.agcGetMode()
                print(f"[INFO] AGC mode set to {enum_simple_name(agc_mode)}")
                return
            except Exception as e:
                # If feature not enabled/supported, try next mode
                if getattr(e, 'args', None) and e.args and e.args[0] == FLR_RESULT.R_CAM_FEATURE_NOT_ENABLED:
                    idx = (idx + 1) % len(order)
                    attempted += 1
                    continue
                print("[ERR] cycle AGC mode failed:", e, file=sys.stderr)
                return
        print("[WARN] No supported AGC mode change available on this device.")

    def set_lut_enabled(en: bool):
        nonlocal lut_enabled
        try:
            cam.colorLutSetControl(FLR_ENABLE_E.FLR_ENABLE if en else FLR_ENABLE_E.FLR_DISABLE)
            lut_enabled = en
            print(f"[INFO] LUT {'ENABLED' if en else 'DISABLED'}")
        except Exception as e:
            print("[ERR] set LUT control failed:", e, file=sys.stderr)

    def set_lut(lut):
        nonlocal lut_id
        try:
            cam.colorLutSetId(lut)
            lut_id = cam.colorLutGetId()
            print(f"[INFO] LUT set to {enum_simple_name(lut_id)}")
        except Exception as e:
            print("[ERR] set LUT failed:", e, file=sys.stderr)

    key_help_short = (
        "[q]quit [1/2/3/4]gain [g]cycle [f]FFC [m]FFCmode [a]AGCmode [r]ROI [c]clearROI LUT:[e/d+w,b,i,h,l,o,R,G,A] REC:[v] snap:[s]"
    )

    # Debug: report captured frame size on first frame / change
    last_frame_report = None  # (w,h,channels,str(dtype))

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Empty frame; continuing...")
                continue

            # Debug: print capture size when changed
            try:
                h0, w0 = frame.shape[:2]
                c0 = 1 if frame.ndim == 2 else frame.shape[2]
                info = (w0, h0, c0, str(frame.dtype))
                if info != last_frame_report:
                    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap_fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"[DBG] Captured frame size: {w0}x{h0}, channels={c0}, dtype={frame.dtype}; CAP reports {cap_w}x{cap_h} @ {cap_fps:.2f} fps")
                    last_frame_report = info
            except Exception:
                pass

            raw = frame.copy()
            vis = frame

            # If selection just ended, apply ROI to SDK
            if selector.get_rect() is not None and selector.last_selection is not None:
                apply_agc_roi_from_selection(vis.shape)
                # Consume this selection so we don't reapply constantly
                selector.last_selection = None

            # Draw overlays and selection rectangle
            if not args.no_overlay:
                draw_overlay(vis)
                # Also show short help at bottom-left
                cv2.putText(
                    vis,
                    key_help_short,
                    (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

            # Write frame if recording
            if recording and rec_writer is not None:
                rec_src = vis if burn_overlay else raw
                rec_bgr = to_bgr_u8(rec_src)
                if rec_bgr is not None:
                    # Resize to locked record_size if needed
                    if record_size is not None:
                        target_sz = record_size
                    else:
                        target_sz = (rec_bgr.shape[1], rec_bgr.shape[0])
                    if (rec_bgr.shape[1], rec_bgr.shape[0]) != target_sz:
                        rec_bgr = cv2.resize(rec_bgr, target_sz, interpolation=cv2.INTER_AREA)
                    rec_writer.write(rec_bgr)

            cv2.imshow(args.title, vis)

            k = cv2.waitKey(1) & 0xFF
            if k == 255:  # no key
                continue

            # Quit
            if k == ord('q'):
                break

            # Gain direct selection
            if k in (ord('1'), ord('2'), ord('3'), ord('4')):
                mapping = {
                    ord('1'): FLR_BOSON_GAINMODE_E.FLR_BOSON_HIGH_GAIN,
                    ord('2'): FLR_BOSON_GAINMODE_E.FLR_BOSON_LOW_GAIN,
                    ord('3'): FLR_BOSON_GAINMODE_E.FLR_BOSON_AUTO_GAIN,
                    ord('4'): FLR_BOSON_GAINMODE_E.FLR_BOSON_DUAL_GAIN,
                }
                set_gain(mapping[k])
                continue
            if k == ord('g'):
                cycle_gain()
                continue

            # FFC
            if k == ord('f'):
                run_ffc()
                continue
            if k == ord('m'):
                toggle_ffc_mode()
                continue

            # AGC
            if k == ord('a'):
                cycle_agc_mode()
                continue
            if k == ord('r'):
                # Start interactive ROI selection
                selector.start_selection()
                print("[INFO] Draw ROI: click-drag, release to set. ESC/right-click to cancel.")
                continue
            if k == 27:  # ESC to cancel ROI selection
                if selector.active:
                    selector.cancel()
                    print("[INFO] ROI selection canceled")
                continue
            if k == ord('c'):
                clear_agc_roi(vis.shape)
                continue

            # LUT control
            if k == ord('e'):
                set_lut_enabled(True)
                continue
            if k == ord('d'):
                set_lut_enabled(False)
                continue
            if k in (ord('w'), ord('b'), ord('i'), ord('R'), ord('G'), ord('h'), ord('A'), ord('l'), ord('o')):
                lut_map = {
                    ord('w'): FLR_COLORLUT_ID_E.FLR_COLORLUT_WHITEHOT,
                    ord('b'): FLR_COLORLUT_ID_E.FLR_COLORLUT_BLACKHOT,
                    ord('i'): FLR_COLORLUT_ID_E.FLR_COLORLUT_IRONBOW,
                    ord('R'): FLR_COLORLUT_ID_E.FLR_COLORLUT_RAINBOW,
                    ord('G'): FLR_COLORLUT_ID_E.FLR_COLORLUT_GRADEDFIRE,
                    ord('h'): FLR_COLORLUT_ID_E.FLR_COLORLUT_HOTTEST,
                    ord('A'): FLR_COLORLUT_ID_E.FLR_COLORLUT_ARCTIC,
                    ord('l'): FLR_COLORLUT_ID_E.FLR_COLORLUT_LAVA,
                    ord('o'): FLR_COLORLUT_ID_E.FLR_COLORLUT_GLOBOW,
                }
                set_lut(lut_map[k])
                continue

            # Recording toggle and snapshot
            if k == ord('v'):
                if not recording:
                    # Initialize with a sample frame size
                    sample_bgr = to_bgr_u8(vis if burn_overlay else raw)
                    if sample_bgr is None:
                        print("[ERR] Cannot start recording: invalid frame.")
                    elif start_recording(sample_bgr):
                        # Replace writer size based on sample_bgr size if needed
                        pass
                else:
                    stop_recording()
                continue
            if k == ord('s'):
                out_dir = ensure_captures_dir()
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                path = os.path.join(out_dir, f"snap_{ts}_dev{args.device}.png")
                # Save what is shown on screen (with overlay)
                snap_bgr = to_bgr_u8(vis)
                ok_save = cv2.imwrite(path, snap_bgr)
                if ok_save:
                    print(f"[INFO] Snapshot saved: {path}")
                else:
                    print("[ERR] Failed to save snapshot")
                continue

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if recording:
            stop_recording()
        try:
            cam.Close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
