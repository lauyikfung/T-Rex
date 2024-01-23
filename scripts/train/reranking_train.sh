if [ $# -eq 0 ]; then
    FOLD=1
    DEVICE=0
    NAME=reranking
elif [ $# -eq 1 ]; then
    FOLD=$1
    DEVICE=0
    NAME=reranking
elif [ $# -eq 2 ]; then
    FOLD=$1
    DEVICE=$2
    NAME=reranking
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
WANDB_KEY=YOUR_WANDB_KEY

PLM=microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL

DATADIR=data/candidate_generation/

SAVEDIR=$NAME
PROJECT_NAME=$NAME
EXP_NAME=$NAME

python reranking_train.py \
    --fold $FOLD \
    --PLM $PLM \
    --lr $LR \
    --max_epochs $EPOCH \
    --alpha $ALPHA \
    --batch_size $BATCHSIZE \
    --data_dir $DATADIR \
    --device $DEVICE \
    --project_name $PROJECT_NAME \
    --exp_name $EXP_NAME \
    --save_file_name $SAVEDIR \
    --graph_model_dir $GRAPH_MODEL_DIR \
    --wandb_key $WANDB_KEY
    