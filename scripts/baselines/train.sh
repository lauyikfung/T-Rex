if [ $# -eq 0 ]; then
    FOLD=1
    DEVICE=0
elif [ $# -eq 1 ]; then
    FOLD=$1
    DEVICE=0
else
    FOLD=$1
    DEVICE=$2
fi

REACTION_LR=1e-3
REACTION_EPOCHS=50
REACTION_BATCHSIZE=32
SYNTHON_LR=1e-3
SYNTHON_EPOCHS=10
SYNTHON_BATCHSIZE=128

DATADIR=data/candidate_generation/


SAVEDIR=../chem_results/G2Gs/fold$FOLD/g2gs-lr-$REACTION_LR-$SYNTHON_LR-batch_size-$REACTION_BATCHSIZE-$SYNTHON_BATCHSIZE



python baselines/G2Gs/g2gs.py \
    --fold $FOLD \
    --reaction_lr $REACTION_LR \
    --reaction_epoch $REACTION_EPOCHS \
    --reaction_batch_size $REACTION_BATCHSIZE \
    --synthon_lr $SYNTHON_LR \
    --synthon_epoch $SYNTHON_EPOCHS \
    --synthon_batch_size $SYNTHON_BATCHSIZE \
    --data_dir $DATADIR \
    --save_dir $SAVEDIR \
    --device $DEVICE