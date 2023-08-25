
BIN_DATA_DIR=/work/yukuanfu88/24k_bin
CHECKPOINT_DIR=/work/yukuanfu88/dgslm_results/encodec
# PRETRAINED_PATH=/home/yukuanfu88/iven/fairseq_dgslm/examples/textless_nlp/ckpt/speech_dlm_base.pt

CUDA_VISIBLE_DEVICE='0,1,2,3,4,5,6,7' fairseq-train $BIN_DATA_DIR \
    --save-dir $CHECKPOINT_DIR \
    --tensorboard-logdir $CHECKPOINT_DIR \
    --task speech_dlm_task \
    --unit-channels unitA,unitB \
    --next-unit-prediction "False" --edge-unit-prediction "True" \
    --duration-prediction "True" --delayed-duration-target "True" \
    --criterion speech_dlm_criterion \
    --arch speech_dlm --decoder-cross-layers 4 \
    --dropout 0.1 --attention-dropout 0.1 \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 1.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
    --max-tokens 12288 --tokens-per-sample 6144 --sample-break-mode eos \
    --update-freq 6 --num-workers 32 --skip-invalid-size-inputs-valid-test \
    --max-update 250000 --warmup-updates 20000 \
    --save-interval-updates 10000 --keep-last-epochs 1 --no-epoch-checkpoints \
    --log-interval 50 --seed 100501 \
    --fp16 --checkpoint-activations --num-code 8