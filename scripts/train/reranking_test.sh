if [ $# -eq 0 ]; then
    FOLD=1
    DEVICE=0
    NAME=logs-ckpt/reranking_fold1.ckpt
elif [ $# -eq 1 ]; then
    FOLD=$1
    DEVICE=0
    NAME=logs-ckpt/reranking_fold1.ckpt
elif [ $# -eq 2 ]; then
    FOLD=$1
    DEVICE=$2
    NAME=logs-ckpt/reranking_fold1.ckpt
else
    FOLD=$1
    DEVICE=$2
    NAME=$3
fi

LR=1e-5
BATCHSIZE=9
ALPHA=0.2
EPOCH=10
GRAPH_MODEL_DIR=pretrained_GCN/reaction_model_fold1.pth

PLM=microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL

DATADIR=data/candidate_generation/

SAVEDIR=$NAME

python reranking_test.py \
    --fold $FOLD \
    --PLM $PLM \
    --lr $LR \
    --max_epochs $EPOCH \
    --alpha $ALPHA \
    --batch_size $BATCHSIZE \
    --data_dir $DATADIR \
    --device $DEVICE \
    --graph_model_dir $GRAPH_MODEL_DIR \
    --checkpoint $SAVEDIR \

    