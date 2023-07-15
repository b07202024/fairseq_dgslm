source ~/.bashrc
conda activate dgslm

EXTENSION=wav
for splt in "test" "valid"; do
    MANIFEST_FILE="/home/iven/fairseq/examples/wav2vec/dgslm/$splt.tsv"
    OUTPUT_FILE="/home/iven/fairseq/examples/textless_nlp/data/$splt"
    for CHANNEL_ID in 1 2; do
        python /home/iven/fairseq/examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans_on_the_fly.py \
            --feature_type hubert \
            --kmeans_model_path /home/iven/fairseq/examples/textless_nlp/ckpt/hubert_fisher_km_500.bin \
            --acoustic_model_path /home/iven/fairseq/examples/textless_nlp/ckpt/hubert_fisher.pt \
            --layer 12 \
            --manifest_path $MANIFEST_FILE \
            --out_quantized_file_path ${OUTPUT_FILE} \
            --extension $EXTENSION \
            --channel_id $CHANNEL_ID \
            --hide-fname
    done
done