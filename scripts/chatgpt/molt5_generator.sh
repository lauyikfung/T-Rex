if [ $# -eq 0 ]; then
    FOLD=1
    DEVICE=0
elif [ $# -eq 1 ]; then
    FOLD=$1
    DEVICE=0
elif [ $# -eq 2 ]; then
    FOLD=$1
    DEVICE=$2

BATCHSIZE=32

DATADIR=data/candidate_generation


python molt5_generator.py \
    --fold $FOLD \
    --batch_size $BATCHSIZE \
    --data_dir $DATADIR \
    --device $DEVICE