#!/bin/bash

ONNX_MODEL="results/crack_detector/weights/best.onnx"
OUTPUT_ENGINE="model_fp16.plan"

echo "Building TensorRT Engine..."

trtexec \
    --onnx=$ONNX_MODEL \
    --saveEngine=$OUTPUT_ENGINE \
    --fp16 \
    --shapes=input:1x3x720x1280 \
    --workspace=2048 \
    | tee trt_build.log

echo ""
echo "Build complete: $OUTPUT_ENGINE"
echo "Log saved: trt_build.log"
echo ""
echo "Parse log for:"
echo "  - mean: XX.XX ms"
echo "  - FPS = 1000 / mean_ms"