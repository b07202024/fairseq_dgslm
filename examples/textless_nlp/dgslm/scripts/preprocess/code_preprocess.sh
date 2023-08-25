RAW_DATA_DIR=/work/yukuanfu88/24k_data    # path to text and unit file generated from unit_generate.sh
BIN_DATA_DIR=/work/yukuanfu88/24k_bin     # path to put the binary results
DICT_DIR=/home/yukuanfu88/iven/fairseq_dgslm/examples/textless_nlp/dgslm/dict
NUM_CODE=8

for lang in "unitA" "unitB"; do
  for (( i=0; i<$NUM_CODE; i++ )); do
    fairseq-preprocess --source-lang $lang \
      --srcdict $DICT_DIR/codec.txt \
      --only-source \
      --trainpref $RAW_DATA_DIR/train-$i \
      --validpref $RAW_DATA_DIR/valid-$i \
      --testpref $RAW_DATA_DIR/test-$i \
      --destdir $BIN_DATA_DIR \
      --workers 20
    for split in "train" "valid" "test"; do
      mv $BIN_DATA_DIR/${split}.${lang}-None.${lang}.bin $BIN_DATA_DIR/${split}-${i}.${lang}.bin
      mv $BIN_DATA_DIR/${split}.${lang}-None.${lang}.idx $BIN_DATA_DIR/${split}-${i}.${lang}.idx
    done
  done
done

for lang in "textA" "textB"; do
  fairseq-preprocess --source-lang $lang \
    --srcdict $DICT_DIR/dict.textA.txt \
    --only-source \
    --trainpref $RAW_DATA_DIR/train \
    --validpref $RAW_DATA_DIR/valid \
    --testpref $RAW_DATA_DIR/test \
    --destdir $BIN_DATA_DIR \
    --workers 20
  for split in "train" "valid" "test"; do
    mv $BIN_DATA_DIR/${split}.${lang}-None.${lang}.bin $BIN_DATA_DIR/${split}.${lang}.bin
    mv $BIN_DATA_DIR/${split}.${lang}-None.${lang}.idx $BIN_DATA_DIR/${split}.${lang}.idx
  done
done

for channel in "timeA" "timeB"; do
  for split in "train" "valid" "test"; do
    cp $RAW_DATA_DIR/${split}.${channel} $BIN_DATA_DIR/${split}.${channel}
  done
done