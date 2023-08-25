# source ~/.bashrc
# conda activate dgslm

EXTENSION=flac
for splt in "test"; do
    MANIFEST_FILE="/home/yukuanfu88/iven/fairseq_dgslm/examples/wav2vec/fisher_16k_sliced/$splt.tsv"
    OUTPUT_FILE="/home/yukuanfu88/iven/fairseq_dgslm/examples/textless_nlp/dgslm/16k_data/$splt"
    for CHANNEL_ID in 1 2; do
        python /home/yukuanfu88/iven/fairseq_dgslm/examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans_on_the_fly.py \
            --feature_type hubert \
            --kmeans_model_path /home/yukuanfu88/iven/fairseq_dgslm/examples/textless_nlp/ckpt/hubert_fisher_km_500.bin \
            --acoustic_model_path /home/yukuanfu88/iven/fairseq_dgslm/examples/textless_nlp/ckpt/hubert_fisher.pt \
            --layer 12 \
            --manifest_path $MANIFEST_FILE \
            --out_quantized_file_path ${OUTPUT_FILE} \
            --extension $EXTENSION \
            --hide-fname \
            --channel_id $CHANNEL_ID
    done
done