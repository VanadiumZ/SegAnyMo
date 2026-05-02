#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/SegAnyMo

# =========================
# 可配置参数（按需修改）
# 目标：仅导出指定帧的 frame + masks，不保留视频
# =========================
VIDEO_PATH="${VIDEO_PATH:-/root/autodl-tmp/SegAnyMo/frame.mp4}"  # 指定要处理的目标视频
GPU_ID="${GPU_ID:-0}"
STEP="${STEP:-10}"
CONFIG_FILE="${CONFIG_FILE:-./configs/example.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./result}"
USE_EFFICIENCY="${USE_EFFICIENCY:-0}"   # 1=开启 --e, 0=关闭
EXTRACT_FRAME="${EXTRACT_FRAME:-0.5}"   # 0~1, 指定导出帧位置
CLEAN_VIDEO_OUTPUTS="${CLEAN_VIDEO_OUTPUTS:-1}"  # 1=清理流程产生的 mp4

SEQ_NAME="$(basename "${VIDEO_PATH}")"
SEQ_NAME="${SEQ_NAME%.*}"

MOSEG_DIR="${OUTPUT_ROOT}/moseg_${SEQ_NAME}"
SAM2_DIR="${OUTPUT_ROOT}/sam2_${SEQ_NAME}"

EFF_FLAG=""
if [[ "${USE_EFFICIENCY}" == "1" ]]; then
  EFF_FLAG="--e"
fi

echo "======================================"
echo "[SegAnyMo] 目标序列: ${SEQ_NAME}"
echo "[SegAnyMo] 输入视频: ${VIDEO_PATH}"
echo "[SegAnyMo] 输出根目录: ${OUTPUT_ROOT}"
echo "[SegAnyMo] 导出帧比例: ${EXTRACT_FRAME}"
echo "======================================"

# 1) 预处理：深度 + TAPIR 轨迹 + DINO
python core/utils/run_inference.py \
  --video_path "${VIDEO_PATH}" \
  --gpus "${GPU_ID}" \
  --step "${STEP}" \
  --depths \
  --tracks \
  --dinos \
  ${EFF_FLAG}

# 2) Moseg 推理（config 中需配置正确的 resume_path）
python core/utils/run_inference.py \
  --video_path "${VIDEO_PATH}" \
  --motin_seg_dir "${MOSEG_DIR}" \
  --config_file "${CONFIG_FILE}" \
  --gpus "${GPU_ID}" \
  --step "${STEP}" \
  --motion_seg_infer \
  ${EFF_FLAG}

# 3) SAM2 阶段（使用上一步 moseg 结果）
python core/utils/run_inference.py \
  --video_path "${VIDEO_PATH}" \
  --sam2dir "${SAM2_DIR}" \
  --motin_seg_dir "${MOSEG_DIR}" \
  --gpus "${GPU_ID}" \
  --sam2 \
  ${EFF_FLAG} \
  --extract_only \
  --extract_frame "${EXTRACT_FRAME}"

# 4) 可选清理视频文件（仅保留 frame/mask 结果）
if [[ "${CLEAN_VIDEO_OUTPUTS}" == "1" ]]; then
  rm -f "${MOSEG_DIR}/${SEQ_NAME}/original.mp4" \
        "${MOSEG_DIR}/${SEQ_NAME}/dynamic.mp4"
  rm -f "${SAM2_DIR}/final_res/${SEQ_NAME}/video/"*.mp4 2>/dev/null || true
  rm -f "${SAM2_DIR}/final_res/${SEQ_NAME}/combined_mask_video/"*.mp4 2>/dev/null || true
fi

echo
echo "========== 结果路径 =========="
echo "导出目录: ${SAM2_DIR}/extracted_frames/${SEQ_NAME}"
echo "RGB 帧: ${SAM2_DIR}/extracted_frames/${SEQ_NAME}/frame.png (.npy)"
echo "动态总掩码: ${SAM2_DIR}/extracted_frames/${SEQ_NAME}/dynamic_mask.png (.npy)"
echo "静态掩码: ${SAM2_DIR}/extracted_frames/${SEQ_NAME}/mask0.png (.npy)"
echo "动态对象掩码: ${SAM2_DIR}/extracted_frames/${SEQ_NAME}/mask1*.png (.npy)"
echo "=============================="