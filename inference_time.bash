#!/bin/bash
set -o xtrace

COMMON_ARGS=(--dataset nyuv2 --tasks dense-visual-embedding --raw-depth --dense-visual-embedding-decoder-n-channels-out 512)

MODELS=("fullres" "reducedres")
MODEL_EXTRA_ARGS=(
    "--weights-filepath ./trained_models/mixed/dveformer_fullres_mixed.pth"
    "--weights-filepath ./trained_models/mixed/dveformer_reducedres_mixed.pth --dense-visual-embedding-decoder-n-upsamplings 0"
)

ARGS_EXPORT=(--no-time-pytorch --trt-onnx-export-only --model-onnx-filepath ./model_tensorrt.onnx)
ARGS_TIME_TRT32=(--n-runs-warmup 20 --n-runs 80 --no-time-pytorch)
ARGS_TIME_TRT16=(--n-runs-warmup 20 --n-runs 80 --no-time-pytorch --trt-floatx 16)

RESULTS_FILE='./results.csv'
echo "Model,TensorRT FP32 FPS,TensorRT FP16 FPS" > "${RESULTS_FILE}"

for idx in "${!MODELS[@]}"; do
    model_name=${MODELS[$idx]}
    extras=${MODEL_EXTRA_ARGS[$idx]}

    read -r -a extras_array <<<"${extras}"

    model_args=("${COMMON_ARGS[@]}" "${extras_array[@]}")

    python3 inference_time_whole_model.py "${ARGS_EXPORT[@]}" "${model_args[@]}"

    fps_fp32=$(python3 inference_time_whole_model.py "${ARGS_TIME_TRT32[@]}" "${model_args[@]}" | sed -n 's/.*fps tensorrt ([^)]*): \([0-9.]*\).*/\1/p')
    if [[ -z "${fps_fp32}" ]]; then
        fps_fp32="NA"
    fi

    fps_fp16=$(python3 inference_time_whole_model.py "${ARGS_TIME_TRT16[@]}" "${model_args[@]}" | sed -n 's/.*fps tensorrt ([^)]*): \([0-9.]*\).*/\1/p')
    if [[ -z "${fps_fp16}" ]]; then
        fps_fp16="NA"
    fi
    echo "${model_name},${fps_fp32},${fps_fp16}" >> "${RESULTS_FILE}"
done
