CUDA_VISIBLE_DEVICES=3 python adversarial.py \
    --model_path checkpoint/model.t7 \
    --p inf \
    --xi 0.5 \
    --delta 0.1 \
    --len-train-x 100 \
    --len-wm-x 50
