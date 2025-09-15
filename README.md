# Boson Tuner Viewer

FLIR Boson 用のライブビューア兼チューニングツールです。UVC から取得した映像を表示しながら、SDK 経由で以下の設定をその場で操作できます。

- ゲインモード: HIGH / LOW / AUTO / DUAL
- FFC: 即時実行、AUTO/MANUAL の切替
- AGC モード切替 + 画面上ドラッグで AGC ROI 指定
- カラーパレット（LUT）の有効/無効・種類切替
- 録画（オーバーレイの焼き込み ON/OFF）とスナップショット保存

起動時にカメラの PN/SN（可能ならセンサ PN）を取得して表示します。

> 注意: UVC が IR16（グレースケール）を吐き、PC 側で擬似カラー化している構成では、カメラ側の LUT 変更は見た目に反映されない場合があります。カメラがカラー化済みのフレームを出力している場合は即時に反映されます。

---

## 必要環境

- Python 3.x
- パッケージ
  - OpenCV: `pip install opencv-python`
  - pyserial: `pip install pyserial`
  - NumPy: `pip install numpy`
- FLIR Boson Python SDK（本リポジトリ直下の `SDK_USER_PERMISSIONS` ディレクトリ。場所を変える場合は `--sdk-path` で指定）
- カメラの映像入力（UVC）と制御用シリアルポートが OS に認識されていること

プラットフォーム補足:
- macOS: OpenCV のバックエンドは通常の `VideoCapture(index)` に失敗した場合、`CAP_AVFOUNDATION` で再試行します。
- Windows: シリアルポートは `COM7` のように指定します。
- Linux: シリアルポートは `/dev/ttyACM0` 等を指定します。

---

## 使い方（基本）

1) 依存パッケージをインストール

```bash
pip install opencv-python pyserial numpy
```

2) シリアルポート名を確認

- macOS: `ls /dev/cu.usb*`
- いずれの OS でも: `python -m serial.tools.list_ports`

3) 実行

```bash
python3 boson_tuner_viewer.py --device 0 --port /dev/cu.usbmodemXXXX
# 例（Windows）
python3 boson_tuner_viewer.py --device 0 --port COM7 --title "Boson Tuner"
```

- `--device`: OpenCV のカメラインデックス（既定: 0）
- `--port`: SDK 用シリアルポート（必須）

---

## コマンドライン引数

- `--device <int>`: UVC デバイスインデックス（既定: 0）
- `--title <str>`: ウィンドウタイトル（既定: "Boson Tuner"）
- `--port <str>`: シリアルポート（例: `/dev/cu.usbmodemXXXX`, `COM7`）
- `--baud <int>`: ボーレート（未指定時は SDK 既定 921600）
- `--sdk-path <path>`: SDK パッケージのパス（既定: `SDK_USER_PERMISSIONS`）
- `--no-overlay`:
  - 画面へのテキスト/矩形などのオーバーレイ描画を無効化
- `--no-camera-overlay`:
  - 起動時にカメラ側オーバーレイ（スポットメータ、シンボロジ、テレメトリ、アイソサーム）を無効化
- `--record-overlay {on,off}`: 録画に画面オーバーレイを焼き込むか（既定: `on`）
- `--record-out <path>`: 録画ファイルの出力先。未指定なら `captures/<timestamp>_dev<idx>.mp4`
- `--codec <fourcc>`: FourCC（既定: `mp4v`。失敗時は MJPG にフォールバック）
- `--fps <float>`: CAP が 0 を返す場合のフォールバック FPS（既定: 30.0）
- `--record-size WxH`:
  - 録画サイズを明示指定（例: `640x512`）。未指定時は最初に得たフレームサイズで固定
- `--crop-telemetry {on,off}`:
  - UVC フレーム末尾に付く余分なテレメトリ行を検出したら 2 行分カット（既定: on, 例: 514→512）

---

## キーボード操作

- 終了: `q`

- ゲイン
  - `1`: HIGH, `2`: LOW, `3`: AUTO, `4`: DUAL
  - `g`: 上記モードを順次切替

- FFC（フラットフィールド補正）
  - `f`: 即時 FFC 実行
  - `m`: FFC モード切替（AUTO ↔ MANUAL）

- AGC（自動ゲイン制御）
  - `a`: モード切替（NORMAL → AUTO_LINEAR → MANUAL → HOLD → THRESHOLD）
  - `r`: ROI 選択開始（左ドラッグで矩形選択。右クリック/ESC でキャンセル）
  - `c`: ROI クリア（フルフレーム）

- LUT（カラーパレット）
  - `e`: 有効化, `d`: 無効化
  - `w`: WHITEHOT, `b`: BLACKHOT, `i`: IRONBOW, `R`: RAINBOW
  - `G`: GRADEDFIRE, `h`: HOTTEST, `A`: ARCTIC, `l`: LAVA, `o`: GLOBOW

- 録画 / スナップショット
  - `v`: 録画開始/停止（`captures/` に保存。初回フレームのサイズ/FPSで固定。必要に応じて `--record-size`, `--codec` を使用）
  - `s`: 表示中のフレームを PNG で保存（オーバーレイは画面表示に準拠）

画面左上に現在の状態（Gain/FFC/AGC/LUT）とショートカットヘルプが表示されます（`--no-overlay` 指定時は非表示）。

---

## 録画動作の詳細

- 出力先: 既定はリポジトリ直下の `captures/` ディレクトリに `YYYYmmdd_HHMMSS_dev<idx>.mp4`。
- コーデック: 既定 `mp4v`。オープンに失敗した場合は `.avi` + `MJPG` に自動フォールバックします。
- 解像度: 明示指定が無ければ「最初に取得したフレームのサイズ」で固定します。
- FPS: CAP から取得。0 のときは `--fps` の値を使用します（既定 30fps）。
- オーバーレイ焼き込み: `--record-overlay on` のときは画面表示そのままを書き出します。`off` のときは「生フレーム」を書き出します。

---

## ROI 指定の挙動

- 画面上で選択した矩形を、可能ならカメラの「ネイティブ座標」に線形マッピングして `agcSetROI` を呼び出します。
- ネイティブ解像度が取得できない場合は、画面座標そのままで `agcSetROI` を試行します。
- クリア時はフルフレームに戻します。

---

## トラブルシュート

- `Failed to import SDK ...`:
  - `--sdk-path` が正しいか、`SDK_USER_PERMISSIONS` が存在するか確認してください。
- `Failed to open SDK serial port`:
  - シリアルポート名、ボーレート、アクセス権限（Linux/macOS での権限やグループ）を確認してください。
- カメラ映像が開けない / 黒画面:
  - `--device` のインデックスを変えて試してください。macOS ではバックエンド切替が自動で行われます。
- LUT 変更が反映されない:
  - UVC がグレースケール（IR16）で PC 側カラー化の場合は見た目が変わらないことがあります。
- フレーム高が 514 などになる:
  - 既定で末尾 2 行をカットします（`--crop-telemetry on`）。不要なら `off` にしてください。

---

## ライセンス / クレジット

- FLIR Boson SDK に関する権利は各社に帰属します。
- 本ツールは `boson_tuner_viewer.py` を用いた操作・確認用のサンプル/ユーティリティです。

---

## 開発者向けメモ

- 依存パッケージや OS 設定に依存する問題がある場合、`print` ログ（起動時の PN/SN、CAP の幅/高さ/FPS など）がデバッグの参考になります。
- macOS で UVC が複数存在する場合、`--device` を切り替えて挙動を確認してください。
- 必要に応じて機能追加・変更（例: PC 側カラー化、温度/輝度の表示等）を行ってください。

