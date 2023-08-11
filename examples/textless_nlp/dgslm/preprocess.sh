RAW_DATA_DIR=/home/yukuanfu88/iven/fairseq_dgslm/examples/textless_nlp/data   # path to text and unit file generated from unit_generate.sh
BIN_DATA_DIR=/home/yukuanfu88/iven/fairseq_dgslm/examples/textless_nlp/bin_data   # path to put the binary results

for lang in "unitA" "unitB"; do
    fairseq-preprocess --source-lang $lang \
        --srcdict $BIN_DATA_DIR/dict.unitA.txt \
        --only-source \
        --trainpref $RAW_DATA_DIR/train \
        --validpref $RAW_DATA_DIR/valid \
        --destdir $BIN_DATA_DIR \
        --workers 20
done

for lang in "textA" "textB"; do
    fairseq-preprocess --source-lang $lang \
        --srcdict $BIN_DATA_DIR/dict.textA.txt \
        --only-source \
        --trainpref $RAW_DATA_DIR/train \
        --validpref $RAW_DATA_DIR/valid \
        --destdir $BIN_DATA_DIR \
        --workers 20
done

for channel in "unitA" "textA" "unitB" "textB"; do
  for split in "train" "valid"; do
    mv $BIN_DATA_DIR/${split}.${channel}-None.${channel}.bin $BIN_DATA_DIR/${split}.${channel}.bin
    mv $BIN_DATA_DIR/${split}.${channel}-None.${channel}.idx $BIN_DATA_DIR/${split}.${channel}.idx
  done
done

for channel in "timeA" "timeB"; do
  for split in "train" "valid"; do
    cp $RAW_DATA_DIR/${split}.${channel} $BIN_DATA_DIR/${split}.${channel}
  done
done