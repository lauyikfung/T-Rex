if [ $# -eq 0 ]; then
    FOLD=1
    DEVICE=0
elif [ $# -eq 1 ]; then
    FOLD=$1
    DEVICE=0
elif [ $# -eq 2 ]; then
    FOLD=$1
    DEVICE=$2

BATCHSIZE=20

DATADIR=data/candidate_generation

SLEEP_TIME=0.14

python ChatGPT_for_reranking.py \
    --fold $FOLD \
    --batch_size $BATCHSIZE \
    --data_dir $DATADIR \
    --sleep_time $SLEEP_TIME \