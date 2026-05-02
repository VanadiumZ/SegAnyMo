# SegAnyMotion 数据输入输出与 Shape 备忘

本文按实际代码链路梳理 SegAnyMotion 的三阶段流程：

1. 预处理（depth / tracks / dino）
2. moseg 推理（`inference.py`）
3. SAM2 后处理（`sam2/run_sam2.py`）

目标是明确：每阶段输入是什么、输出是什么、核心张量 shape 是什么、哪些参数会影响 shape（尤其 `T/H/W/N`）。

---

## 0. 总流程（从 `run_inference.py` 出发）

统一入口：`core/utils/run_inference.py`

- 预处理阶段可调起：
  - `core/utils/run_depth.py`
  - `preproc/run_tapir.py`（Bootstapir）
  - `core/utils/dino_feat.py`
- moseg 阶段调起：
  - `inference.py`
- SAM2 阶段调起：
  - `sam2/run_sam2.py`

关键入口参数（最常用）：

- 输入相关：`--video_path` / `--data_dir`
- 采样相关：`--step`、`--e`
- moseg 输出根目录：`--motin_seg_dir`
- sam2 输出根目录：`--sam2dir`

备注：使用 `--video_path` 时，脚本会先抽帧到 `images/<seq_name>/`，并只处理该序列（已修复过滤逻辑）。

---

## 1. 阶段A：预处理（Depth / Tracks / DINO）

## 1.1 输入与目录约定

输入来源：

- 视频模式：`--video_path /path/to/xxx.mp4`
  - 先执行抽帧，输出 `.../images/<seq_name>/00000.png ...`
- 图像序列模式：`--data_dir /path/to/images_root`
  - 其中每个子目录是一个序列，如 `images_root/7128`、`images_root/26525`

图像文件支持（常见）：

- 抽帧输出是 `.png`
- 后续读取通常支持 `png/jpg/jpeg`

---

## 1.2 抽帧与 resize 对 shape 的影响

### 抽帧（`video_to_images`）

- 原始帧：每帧 `H x W x 3`（OpenCV BGR）
- 输出命名：`{saved_frame_count:05d}.png`
- `--e`（efficiency）开启时：
  - 帧数 `T` 最多 100（均匀抽样）
  - `frame_interval = total_frames // target_frames`

### resize（`resize_images`）

- 若 `max(H, W) > 1000`，按比例缩放到长边 1000
- 影响后续所有阶段的空间分辨率：`H/W` 变小

---

## 1.3 Depth（`core/utils/run_depth.py`）

输入：

- `--img_dir`: 序列帧目录
- `--out_raw_dir`: 深度输出目录

输出：

- 每帧一个深度/视差图：`out_raw_dir/<frame>.png`（`uint16`）
- 尺寸与输入图像帧一致（同名匹配）

shape：

- 单帧深度：`H x W`（2D）
- 序列深度文件集合：`T` 张 PNG

注意：

- `run_depth.py` 中虽有 `--step` 参数，但当前实现中实际保存逻辑未按 step 采样（参数保留但未实用到核心遍历）。

---

## 1.4 Tracks（`preproc/run_tapir.py`，Bootstapir）

输入：

- `--image_dir`: 序列帧目录
- `--out_dir`: 轨迹输出目录
- `--step`: query 帧间隔（影响 query 时间点数量）

内部 shape（代码可见）：

- 模型输入帧（resize 后）近似为：`[1, T, resize_h, resize_w, 3]`
- query 时刻：`q_ts = range(0, T, step)`

输出文件：

- 命名：`<name_t>_<name_j>.npy`
- 每个文件保存某个 query 时刻到所有目标帧的结果，单文件 shape 为：
  - `[#targets, 4]`
  - 4 维含义：`x, y, occlusion, expected_dist`

注：轨迹是“拆分存文件”格式，不是一次性一个大 `N x T x ...` 文件。

---

## 1.5 DINO（`core/utils/dino_feat.py`）

输入：

- `--image_dir`: 序列帧目录
- `--step`: 只对 `q_ts` 采样帧提特征

内部 shape（核心）：

- patch 特征 reshape 后：`[H_p, W_p, C]`
  - `H_p/W_p` 由 patch 网格决定
  - `C` 由模型类型决定

输出：

- 每个采样帧一个 `.npy`（`float16`）：
  - `dinos/<seq>/<frame>.npy`
  - 内容 shape：`[H_p, W_p, C]`

---

## 2. 阶段B：moseg 推理（`inference.py`）

## 2.1 输入

命令关键参数：

- `--imgs_dir`: RGB 帧目录
- `--depths_dir`: 深度目录
- `--track_dir`: 轨迹目录（Bootstapir 或 CoTracker）
- `--save_dir`: moseg 输出根目录（实际按 seq 再分子目录）

输入契约（必须对齐）：

- RGB、Depth、Tracks 最终都按同一序列同一帧时序对齐
- 帧名一致性非常重要（同名匹配）

---

## 2.2 核心内部张量 shape

### 图像与深度

- `imgs`: `[B, T, C, H, W]`（代码注释明确）
- `depths`: `[B, 1, H, W, L]`（实际 `L == T`）

### 轨迹与可见性（Bootstapir 分支）

- 聚合后 `track`: `[B, 2, N, L]`
  - 2 维是 `(x, y)`
- `visible_mask`: `[N, L]`
- 传入模型的 `mask = (~visible_mask).unsqueeze(0).unsqueeze(0)`：
  - shape `[B, 1, N, L]`

### 模型输入约定（源码注释）

- `traj: [B, 2, N, L]`
- `depth: [B, 1, H, W, L]`
- `mask: [B, 1, N, L]`

### 模型输出 `pred`

- 通过后续 `view(-1)` 与 `d_mask` 使用方式可知与点维 `N` 对齐
- 精确高维形状在代码中未直接打印，文档中按“与 N 对齐”理解即可

---

## 2.3 moseg 输出（最重要）

输出目录：

- `<save_dir>/<seq_name>/`

核心文件与 shape：

- `dynamic_traj.npy`: `[2, N_dyn, T]`
- `dynamic_visibility.npy`: `[N_dyn, T]`
- `dynamic_confidences.npy`: `[N_dyn, T]`

额外可视化（可能生成）：

- `original.mp4`
- `dynamic.mp4`

其中 `N_dyn` 是筛选后的动态点数量（不一定等于原始 `N`）。

---

## 3. 阶段C：SAM2（`sam2/run_sam2.py`）

## 3.1 输入

命令关键参数：

- `--video_dir`: RGB 帧目录
- `--dynamic_dir`: moseg 输出目录（上一阶段产物）
- `--output_mask_dir`: SAM2 输出根目录
- `--extract_only` / `--extract_frame`: 导出指定帧及掩码

从 `dynamic_dir` 读取的三件套（硬契约）：

- `dynamic_traj.npy`: `[2, N, T]`
- `dynamic_visibility.npy`: `[N, T]`
- `dynamic_confidences.npy`: `[N, T]`

---

## 3.2 核心 mask 形状

按帧 mask：

- 单帧动态/静态掩码：`H x W`（bool 或 uint8）
- 多帧堆叠后常见为：`[T, H, W]`

传播与 merge 后，最终会得到：

- 每帧实例分割 palette PNG
- 动态 union mask（D）
- 静态参考 mask（static_ref）

---

## 3.3 SAM2 输出目录与文件

输出路径会因 `output_mask_dir` 是否含 `"baseline"` 而分支，常见结构：

- 初始预测（非 baseline）：
  - `<output_mask_dir>/initial_preds/<video>/<frame>.png`
- 最终结果：
  - `<output_mask_dir>/final_res/<video>/...`

`extract_only`/`extract_frame` 时常见导出（`extracted_frames/...`）：

- `frame.png` / `frame.npy`（RGB）
- `dynamic_mask.png` / `dynamic_mask.npy`
- `mask0.png` / `mask0.npy`（静态）
- `mask1...png` / `mask1...npy`（动态对象）

---

## 3.4 static/moving mask 的具体实现逻辑（你当前这版）

你当前仓库里的 static/moving mask 不是“模型直接端到端输出”，而是 **moseg 动态点 + SAM2 分割结果 + 规则后处理** 的组合逻辑。

### Step 1：读取 moseg 输出（点级动态信息）

`run_sam2.py` 的 `load_data(dynamic_dir)` 会读取：

- `dynamic_traj.npy`：`[2, N, T]`
- `dynamic_visibility.npy`：`[N, T]`
- `dynamic_confidences.npy`：`[N, T]`

这些文件只表达“哪些轨迹点是动态点”，还不是像素级 mask。

### Step 2：基于动态点驱动 SAM2，得到每帧动态区域 `D_t`

SAM2 传播后会产生每帧实例掩码（对象级）。代码把这些对象掩码并成 union 动态掩码：

- `dyn_segments[t]`：第 `t` 帧动态区域（布尔 `H x W`）
- 形成方式是对每个对象 mask 做按位或：`dyn_segments[t] |= dyn_m`

因此 moving mask 的本体就是每帧 `D_t`。

### Step 3：构造全局静态参考 `static_ref`

代码先在参考帧定义一个静态参考（当前实现优先取第 3 帧，索引 2）：

- `W`：该帧可用的总前景/工作区域（来自 SAM2 输出拼合）
- `D_ref`：参考帧动态区域
- `static_ref = W_ref & ~D_ref`

直观上：参考帧里“不是动态”的部分被当作静态参考模板（帧数不足 3 时回退到最后可用帧）。

### Step 4：每一帧静态 mask 由 `static_ref` 与当前动态区相减得到

对任意帧 `t`，代码使用：

- `static_t = static_ref - (static_ref ∩ D_t)`
- 等价布尔表达：`static_t = static_ref & ~D_t`

这就是 `mask0` 的来源（静态）。

### Step 5：导出规则（`extract_frame`）

在 `--extract_only` 或 `--extract_frame` 时，导出指定帧：

- `dynamic_mask`：`D_t`（完整 moving 区域）
- `mask0`：`static_ref & ~D_t`（静态）
- `mask1...`：从 `D_t` 连通域/对象拆分的动态对象掩码
  - 若只得到一个动态对象，至少会有 `mask1 = D_t`

### 这个设计的语义总结

- **moving mask**：当前帧动态 union 区域（`D_t`）
- **static mask (`mask0`)**：参考帧静态参考 `static_ref` 去掉当前帧动态区域后的剩余部分

所以你的 static/moving 定义是“**参考帧静态参考（默认第3帧） + 每帧动态扣除**”，不是逐帧独立重新估计 static。

---

## 4. 一页版 Shape 对照表

- 视频抽帧单帧：`H x W x 3`（PNG）
- 深度单帧：`H x W`（PNG, uint16）
- Bootstapir 单文件：`[#targets, 4]`（`x,y,occ,dist`）
- DINO 单帧：`[H_p, W_p, C]`（float16）
- moseg 输入 `imgs`：`[B,T,C,H,W]`
- moseg 输入 `depths`：`[B,1,H,W,T]`
- moseg 输入 `traj`：`[B,2,N,T]`
- moseg 输入 `mask`：`[B,1,N,T]`
- moseg 输出 `dynamic_traj`：`[2,N_dyn,T]`
- moseg 输出 `dynamic_visibility`：`[N_dyn,T]`
- moseg 输出 `dynamic_confidences`：`[N_dyn,T]`
- SAM2 每帧 mask：`H x W`
- SAM2 堆叠 mask（常见）：`[T,H,W]`

---

## 5. 影响 shape 的关键参数

- `--e`
  - `T` 上限 100
  - 长边限制 1000（影响 `H/W`）
- `--step`
  - 影响 TAPIR query 时间点密度（影响轨迹覆盖与 `N_dyn`）
  - 影响 DINO 提特征帧集合
- SAM2 内部还有独立采样策略（与上游 step 不完全绑定）

---

## 6. 实验常见坑（建议每次开跑前自检）

- **序列混跑**：`data_dir` 下多个序列时，需确认是否只跑目标序列（`video_path` 已自动限制，`data_dir` 可配 `--seq_names`）。
- **帧名不对齐**：RGB/Depth/Track 不同步会导致下游 shape 或索引错误。
- **路径分支差异**：`output_mask_dir` 是否包含 `"baseline"` 会改变 SAM2 输出层级。
- **`--e` 的副作用**：不仅加速，还改变时空分辨率；实验对比时要保持一致。
- **历史结果覆盖**：阶段间有跳过逻辑，但路径不同可能被误判为“未完成”，建议固定命名规范。

---

## 7. 推荐命名规范（防止实验遗忘）

建议每个实验统一一个 `SEQ_ID`，并让所有阶段路径都包含它：

- moseg：`./result/moseg_<SEQ_ID>/`
- sam2：`./result/sam2_<SEQ_ID>/`

并在实验记录里固定写下：

- 输入视频路径
- 是否 `--e`
- `--step`
- config 文件与模型权重

这样后续回看结果时可以快速定位每个产物对应的参数设置。
