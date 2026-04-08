# FloorSet Solution — Floorplan Transformer

用於 VLSI Floorplan的自回歸 Transformer（ICCAD 2026 Contest）。

---

## 快速開始

```powershell
# 1. 建立訓練快取（首次執行，或修改 block 排序後需重跑）
python solution/data/build_cache.py

# 2. 訓練
python solution/train.py

# 3. 官方評估（100 個驗證案例）
cd iccad2026contest
python iccad2026_evaluate.py --evaluate my_optimizer.py

# 4. 詳細評估（逐案輸出分數）
python evaluate_detail.py --evaluate my_optimizer.py
python evaluate_detail.py --evaluate my_optimizer.py --test-id 94            # 單一案例
python evaluate_detail.py --evaluate my_optimizer.py --test-id 94 --log-blocks  # 輸出每個 block 的 x,y,w,h
python evaluate_detail.py --evaluate my_optimizer.py --viz                   # 同時輸出視覺化圖片
```

---

## 目錄結構

```
solution/
├── config.py                  # 所有超參數（改設定從這裡開始）
├── train.py                   # 訓練入口
├── inference.py               # 自回歸推論工具函式
├── viz_utils.py               # Floorplan視覺化
│
├── model/
│   ├── transformer_floorplan.py   # 完整模型（整合各子模組）
│   ├── encoder.py                 # Transformer Encoder + NetlistBiasAttention
│   ├── decoder.py                 # Transformer Decoder（teacher-forcing / 自回歸）
│   ├── regression_head.py         # ContinuousRegressionHead + 無重疊搜尋
│   └── token_feature.py           # TokenFeatureProjection
│
├── data/
│   ├── floorset_loader.py         # preprocess_sample()、sort_block_indices()、DataLoader
│   └── build_cache.py             # 預計算並分片存放訓練資料
│
├── loss/
│   ├── wirelength_loss.py         # HPWL（b2b + p2b）
│   ├── area_loss.py               # 整體 bounding box 面積
│   └── violation_loss.py          # Overlap / Grouping / MIB / boundary 約束違反
│
├── checkpoints/
│   ├── best.pt                    # 最佳 checkpoint（依 Avg Cost）
│   └── latest.pt                  # 最新 checkpoint
│
└── logs/
    ├── loss_curve.png
    ├── val_log.txt
    └── viz/                       # 訓練過程中儲存的Floorplan圖
```

---

## 模型架構

### 整體流程

```
token_features [B, k, 21]
      │
      ▼
TokenFeatureProjection          Linear(21, d_model) + LayerNorm
      │
      ▼
FloorplanEncoder                6 層 Transformer Encoder
  + NetlistBiasAttention        attention_score += α × w_int（線網權重偏置）
      │  [B, k, d_model]
      ▼
FloorplanDecoder                6 層 Transformer Decoder（自回歸）
  訓練時（teacher-forcing）     平行計算，使用 causal mask，餵入 GT 位置
  推論時（autoregressive）      逐步生成，以自身預測作為下一步輸入
      │  [B, k, d_model]
      ▼
ContinuousRegressionHead
  shared MLP                    Linear(d_model, 64) → ReLU
  xy_head                       Linear(64, 2) → sigmoid → (x, y) ∈ (0, 1)
  ratio_head                    Linear(66, 1) → clamp(log r, log 0.2, log 5.0) → ratio
  尺寸解碼                      w = sqrt(area × ratio)，h = sqrt(area / ratio)
      │  [B, k, 4]
      ▼
pred_positions (x, y, w, h)，正規化至 [0, 1]
```

### 關鍵參數

| 參數 | 數值 |
|---|---|
| d_model | 256 |
| Encoder 層數 | 6 |
| Decoder 層數 | 6 |
| Attention heads | 8 |
| FFN hidden dim | 1024 |
| 面積守恆 | `w × h == area_target`，由數學恆等式保證 |
| 長寬比限制 | w/h ∈ [0.2, 5.0] |

---

## Token Features（21 維）

| 索引 | 特徵 |
|---|---|
| 0 | area_target（除以 canvas² 正規化） |
| 1 | is_soft |
| 2 | is_fixed_shape |
| 3 | is_preplaced |
| 4–5 | target_w、target_h（正規化；soft block 為 0）|
| 6–7 | target_x、target_y（正規化；非 preplaced 為 0）|
| 8–15 | boundary_type one-hot（8 個方向）|
| 16 | is_mib |
| 17–20 | cluster_group one-hot（group 1–4；全零表示無群組）|

---

## Block 擺放順序（自回歸序列）

Block 在送入 Decoder 前依以下優先級排序：

```
preplaced → boundary → cluster（依 group ID 排列）→ fixed-shape → mib → normal
```

每個 block 只歸入最高優先級的類別（互斥）。此順序讓 Decoder 先看到有結構約束的 block，在擺放較自由的 soft block 時擁有更完整的上下文。

修改排序後需重建快取：`python solution/data/build_cache.py`

---

## Loss 函式

| Loss | 權重 | 說明 |
|---|---|---|
| `L_wirelength` | 0.3 | b2b + p2b 連線的 HPWL |
| `L_area` | 0.3 | 所有 block 的 bounding box 面積 |
| `V_overlap` | 50.0 | 兩兩重疊面積（硬懲罰）|
| `V_grouping` | 0.1 | Cluster 群心距離懲罰 |
| `V_mib` | 0.1 | MIB 尺寸偏差 |
| `V_boundary` | 0.1 | Boundary 擺放間距懲罰 |

---

## 推論時的無重疊保證

1. **預先填入**：自回歸迴圈開始前，先將所有 preplaced block 的位置加入 `placed_blocks`，確保非 preplaced block 在任何步驟都能避開它們。

2. **邊界候選搜尋**：對每個非 preplaced block，`find_nonoverlap_position` 在以下候選集合中找最近的不重疊位置：
   ```
   x_cands = {0} ∪ {px + pw + ε  for each placed block}
   y_cands = {0} ∪ {py + ph + ε  for each placed block}
   ```

3. **事後 Legalization**：推論結束後，以 greedy sweep 解決剩餘重疊，非固定 block 最多移動 30 輪直到無重疊。

4. **精確座標還原**：Preplaced block 直接使用 litelabel 的原始像素整數座標，避免 normalize→denormalize 的浮點誤差（~8e-6 px）觸發比賽門檻（1e-6 px）。

---

## 訓練指令

```powershell
# Smoke test（2 batch × 2 epoch，快速確認流程正確）
python solution/train.py --smoke-test

# 完整訓練
python solution/train.py

# 從指定 checkpoint 繼續訓練
python solution/train.py --checkpoint best.pt

# 只使用前 N 個 cache shard 訓練（快速實驗用，例如只用 1 個 shard）
python solution/train.py --num-shards 1
```

每 `VALIDATE_EVERY=5` 個 epoch 自動呼叫官方評估器（subprocess），Avg Cost 最佳的 checkpoint 存至 `checkpoints/best.pt`。

---

## 主要設定參數（`solution/config.py`）

| 參數 | 預設值 | 說明 |
|---|---|---|
| `D_MODEL` | 256 | Embedding 維度 |
| `BATCH_SIZE` | 128 | 訓練 batch size |
| `MAX_EPOCHS` | 100 | 最大訓練 epoch 數 |
| `LEARNING_RATE` | 1e-4 | 峰值學習率 |
| `WARMUP_STEPS` | 4000 | 線性 warmup 步數 |
| `VALIDATE_EVERY` | 5 | 每幾個 epoch 執行官方評估 |
| `MIN_RATIO` | 0.2 | w/h 最小長寬比 |
| `MAX_RATIO` | 5.0 | w/h 最大長寬比 |
| `CACHE_PRELOAD` | False | 是否將完整快取載入 RAM |
