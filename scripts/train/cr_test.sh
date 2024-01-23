if [ $# -eq 0 ]; then
    FOLD=1
    DEVICE=0
    NAME=Candidate_Generation
elif [ $# -eq 1 ]; then
    FOLD=$1
    DEVICE=0
    NAME=Candidate_Generation
elif [ $# -eq 2 ]; then
    FOLD=$1
    DEVICE=$2
    NAME=Candidate_Generation
else
    FOLD=$1
    DEVICE=$2
    NAME=$3
fi

REACTION_LR=1e-3
REACTION_EPOCHS=50
REACTION_BATCHSIZE=32
REACTIONWARMUPS=2000
SYNTHON_LR=1e-3
SYNTHON_EPOCHS=10
SYNTHON_BATCHSIZE=128


WANDB_KEY=YOUR_WANDB_KEY
REACTION_ARCH=CenterIdentificationTruncate

PLM=prajjwal1/bert-small

DATADIR=data/candidate_generation

SAVEDIR=../chem_results/$NAME/fold$FOLD/model-$REACTION_ARCH-lr-$REACTION_LR-$SYNTHON_LR-batch_size-$REACTION_BATCHSIZE-$SYNTHON_BATCHSIZE

python cr_test.py \
    --fold $FOLD \
    --PLM $PLM \
    --PLM_d 512 \
    --reaction_arch $REACTION_ARCH \
    --reaction_lr $REACTION_LR \
    --reaction_epoch $REACTION_EPOCHS \
    --reaction_batch_size $REACTION_BATCHSIZE \
    --reaction_warmups $REACTIONWARMUPS \
    --synthon_lr $SYNTHON_LR \
    --synthon_epoch $SYNTHON_EPOCHS \
    --synthon_batch_size $SYNTHON_BATCHSIZE \
    --data_dir $DATADIR \
    --save_dir $SAVEDIR \
    --device $DEVICE \
    --max_len 512
    