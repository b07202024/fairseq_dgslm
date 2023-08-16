
BIN_DATA_DIR=/home/yukuanfu88/iven/fairseq_dgslm/examples/textless_nlp/bin_data
CHECKPOINT_DIR=/home/yukuanfu88/iven/fairseq_dgslm/examples/textless_nlp/sep_results
# PRETRAINED_PATH=/home/yukuanfu88/iven/fairseq_dgslm/examples/textless_nlp/new_results/checkpoint_best.pt

CUDA_VISIBLE_DEVICE=0 fairseq-train $BIN_DATA_DIR \
    --save-dir $CHECKPOINT_DIR \
    --tensorboard-logdir $CHECKPOINT_DIR \
    --task speech_dlm_task --channels unitB,unitA \
    --next-unit-prediction "False" --edge-unit-prediction "True" \
    --duration-prediction "True" --delayed-duration-target "True" \
    --ctc-prediction "True" --ctc-loss-weight 1 \
    --criterion speech_dlm_criterion \
    --arch speech_dlm --decoder-cross-layers 4 \
    --share-decoder-input-output-embed \
    --dropout 0.1 --attention-dropout 0.1 \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 1.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 \
    --max-tokens 18432 --tokens-per-sample 6144 --sample-break-mode eos \
    --update-freq 16 --num-workers 4 --skip-invalid-size-inputs-valid-test \
    --max-update 250000 --warmup-updates 20000 \
    --save-interval-updates 10000 --keep-last-epochs 1 --no-epoch-checkpoints \
    --log-interval 50 --seed 100501 \
    --fp16 --checkpoint-activations \
    # --restore-file $PRETRAINED_PATH \