# source ~/.bashrc
# conda activate dgslm

EXTENSION=flac
for splt in "train" "valid" "test"; do
    MANIFEST_FILE="/home/yukuanfu88/iven/fairseq_dgslm/examples/wav2vec/fisher_24k_sliced/$splt.tsv"
    OUTPUT_FILE="/work/yukuanfu88/24k_data/$splt"
    for CHANNEL_ID in 1 2; do
        python /home/yukuanfu88/iven/fairseq_dgslm/examples/textless_nlp/gslm/speech2unit/clustering/code_generate.py \
            --manifest_path $MANIFEST_FILE \
            --out_quantized_file_path ${OUTPUT_FILE} \
            --extension $EXTENSION \
            --hide-fname \
            --channel_id $CHANNEL_ID
    done
done