#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boson 640 (UVC / Y16) viewer for macOS
- libuvc + pyuvc(uvc) で UVC デバイスを列挙・接続
- 16bit(Y16) を 8bit にレンジ圧縮して OpenCV 表示
- 自動コントラスト (パーセンタイル) / 固定レンジ 切替
- FPS 表示、フレームタイムアウト時の再接続
- PEP8 準拠
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional, Tuple, List

import cv2
import numpy as np

import uvc  # pyuvc (libuvc binding)


# ------------------------------
# 設定とユーティリティ
# ------------------------------

COLORMAPS = {
    "none": None,
    "inferno": cv2.COLORMAP_INFERNO,
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "hot": cv2.COLORMAP_HOT,
}


def list_devices() -> List[dict]:
    """UVC デバイス一覧を取得。"""
    try:
        return uvc.device_list()
    except Exception as exc:
        print(f"[ERR] Failed to enumerate UVC devices: {exc}", file=sys.stderr)
        return []


def pick_y16_device(
    devices: List[dict],
    preferred_uid: Optional[str] = None,
) -> Optional[str]:
    """
    Y16 をサポートするデバイスから 1 台選ぶ。
    preferred_uid が指定されていればそれを優先。
    戻り値: device uid or None
    """
    if not devices:
        return None

    # まず preferred_uid が有効ならそれを返す
    if preferred_uid:
        for d in devices:
            if d.get("uid") == preferred_uid:
                return preferred_uid

    # Y16 が選べるかどうかを確認するため、一旦順に開いてモードを見る
    for d in devices:
        uid = d.get("uid")
        try:
            cap = uvc.Capture(uid)
            try:
                modes = cap.available_modes
                # available_modes は (w, h, fps, FOURCC) のタプルリスト想定
                for m in modes:
                    if len(m) >= 4 and str(m[3]).upper().startswith("Y16"):
                        cap.close()
                        return uid
            finally:
                # ここに来るときは cap を閉じておく
                try:
                    cap.close()
                except Exception:
                    pass
        except Exception:
            # 開けないデバイスはスキップ
            continue

    # 見つからなければ最初のデバイスの uid を返す（最後の手段）
    return devices[0].get("uid")


def percentile_auto_range(
    img16: np.ndarray,
    low_p: float = 2.0,
    high_p: float = 98.0,
) -> Tuple[int, int]:
    """16bit画像のパーセンタイルに基づくレンジを返す。"""
    lo = int(np.percentile(img16, low_p))
    hi = int(np.percentile(img16, high_p))
    if hi <= lo:
        hi = lo + 1
    return lo, hi


def y16_to_u8(
    img16: np.ndarray,
    clip_min: int,
    clip_max: int,
) -> np.ndarray:
    """
    16bit(Y16) → 8bit のレンジ圧縮。
    clip_min～clip_max の間を 0～255 に正規化。
    """
    img = np.clip(img16, clip_min, clip_max).astype(np.float32)
    img = (img - clip_min) / max(1.0, float(clip_max - clip_min))
    img = (img * 255.0).astype(np.uint8)
    return img


def apply_colormap(gray_u8: np.ndarray, cmap_name: str) -> np.ndarray:
    """OpenCV のカラーマップを適用。"""
    cmap = COLORMAPS.get(cmap_name, None)
    if cmap is None:
        return gray_u8
    return cv2.applyColorMap(gray_u8, cmap)


# ------------------------------
# キャプチャクラス
# ------------------------------

class BosonUvcCapture:
    """Boson の UVC（Y16）を pyuvc で掴んでフレームを取り出す。"""

    def __init__(self, device_uid: Optional[str] = None):
        self.device_uid = device_uid
        self.cap: Optional[uvc.Capture] = None

    def open(self) -> None:
        """デバイスを列挙して接続。Y16 モードを選ぶ。"""
        devices = list_devices()
        if not devices:
            raise RuntimeError("UVCデバイスが見つかりません。Boson を接続してください。")

        uid = pick_y16_device(devices, self.device_uid)
        if uid is None:
            raise RuntimeError("Y16対応のUVCデバイスが見つかりません。")

        self.cap = uvc.Capture(uid)
        print(f"[INFO] Opened device uid={uid}")

        # モード一覧を出す
        modes = getattr(self.cap, "available_modes", [])
        if modes:
            print("[INFO] Available modes (w, h, fps, fourcc):")
            for m in modes:
                print("  ", m)

        # Y16 のモードに設定（最初に見つかったもの）
        y16_modes = [m for m in modes if len(m) >= 4 and str(m[3]).upper().startswith("Y16")]
        if not y16_modes:
            # 仕方ないので最初のモード
            print("[WARN] Y16 モードが見つからないため、既定モードを使用します。")
        else:
            # Boson 640 は 640x512 を優先
            preferred = None
            for m in y16_modes:
                if m[0] == 640 and m[1] == 512:
                    preferred = m
                    break
            if preferred is None:
                preferred = y16_modes[0]

            print(f"[INFO] Select mode: {preferred}")
            self.cap.frame_mode = preferred

        # 自動で色変換等をしない（念のため）
        try:
            self.cap.auto_exposure = False  # 無効化できない場合もある
        except Exception:
            pass

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.close()
            except Exception:
                pass
            self.cap = None

    def get_frame(self, timeout_s: float = 2.0) -> np.ndarray:
        """
        1フレーム取得。戻り値は dtype=uint16 の 2D 配列想定。
        タイムアウトやエラー時は例外。
        """
        if self.cap is None:
            raise RuntimeError("デバイスがオープンされていません。")

        start = time.time()
        while True:
            try:
                frame = self.cap.get_frame_robust()  # 例外を投げてくれる安定版
                img16 = frame.img  # numpy.ndarray (uint16)
                if img16 is None or img16.size == 0:
                    raise RuntimeError("空フレームを受信。")
                return img16
            except Exception as exc:
                if time.time() - start > timeout_s:
                    raise TimeoutError(f"フレーム取得タイムアウト: {exc}") from exc
                # 少し待って再試行
                time.sleep(0.01)


# ------------------------------
# メイン表示ループ
# ------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FLIR Boson 640 UVC(Y16) viewer for macOS"
    )
    parser.add_argument("--device-uid", type=str, default=None,
                        help="接続したい UVC デバイスの UID（未指定なら自動選択）")
    parser.add_argument("--auto-contrast", type=int, default=1,
                        help="自動コントラスト有効(1)/無効(0) ※パーセンタイル")
    parser.add_argument("--p-low", type=float, default=2.0,
                        help="自動コントラストの下側パーセンタイル（%）")
    parser.add_argument("--p-high", type=float, default=98.0,
                        help="自動コントラストの上側パーセンタイル（%）")
    parser.add_argument("--clip-min", type=int, default=3000,
                        help="固定レンジ使用時の最小値")
    parser.add_argument("--clip-max", type=int, default=10000,
                        help="固定レンジ使用時の最大値")
    parser.add_argument("--colormap", type=str, default="none",
                        choices=list(COLORMAPS.keys()),
                        help="表示カラーマップ")
    parser.add_argument("--title", type=str, default="Boson 640 (UVC/Y16)",
                        help="ウィンドウタイトル")
    parser.add_argument("--reconnect-wait", type=float, default=2.0,
                        help="再接続試行までの待ち時間[秒]")

    args = parser.parse_args()

    # OpenCV の HighGUI が macOS で前面に出ない場合は、フルディスクアクセスや
    # カメラ権限を確認してください。
    cv2.namedWindow(args.title, cv2.WINDOW_NORMAL)

    cap = BosonUvcCapture(device_uid=args.device_uid)

    def ensure_open() -> None:
        tries = 0
        while True:
            try:
                cap.open()
                return
            except Exception as exc:
                tries += 1
                print(f"[WARN] Open failed (try {tries}): {exc}")
                time.sleep(args.reconnect_wait)

    ensure_open()

    last_t = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            try:
                img16 = cap.get_frame(timeout_s=2.0)

                # FPS 計測
                frame_count += 1
                now = time.time()
                if now - last_t >= 1.0:
                    fps = frame_count / (now - last_t)
                    frame_count = 0
                    last_t = now

                # レンジ決定
                if args.auto_contrast:
                    lo, hi = percentile_auto_range(img16, args.p_low, args.p_high)
                else:
                    lo, hi = args.clip_min, args.clip_max

                # 8bit 化
                img8 = y16_to_u8(img16, lo, hi)

                # カラーマップ
                vis = apply_colormap(img8, args.colormap)

                # OSD（レンジ・FPS）
                text = f"{lo}-{hi} (16bit)  |  FPS: {fps:.1f}"
                cv2.putText(
                    vis,
                    text,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    lineType=cv2.LINE_AA,
                )

                cv2.imshow(args.title, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            except TimeoutError as exc:
                print(f"[WARN] {exc} -> 再接続を試みます...")
                cap.close()
                time.sleep(args.reconnect_wait)
                ensure_open()
            except Exception as exc:
                print(f"[ERR] 予期しないエラー: {exc}")
                cap.close()
                time.sleep(args.reconnect_wait)
                ensure_open()

    finally:
        cap.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()