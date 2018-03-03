export REPO_DIR=/Users/lucaslingle/git/memn2n

export TASK_ID=$1
export number_of_hops=3
export TRIAL_DIRNAME=trial_${TRIAL_ID}__bAbI_joint_adj_${number_of_hops}hop_pe_ls_rn

export CHECKPOINT_DIR=${REPO_DIR}/checkpoints/${TRIAL_DIRNAME}
export VOCAB_DIR=${REPO_DIR}/vocab/${TRIAL_DIRNAME}

if [ -d "$REPO_DIR" ]; then
    echo "using $REPO_DIR as REPO_DIR"
    else echo "ERROR:"; echo "  $REPO_DIR does not exist!"; echo "  Please change script to indicate correct REPO_DIR"; return
fi

mkdir -p $CHECKPOINT_DIR
mkdir -p $VOCAB_DIR


# Training with linear start. Linear phase with official configs. 

python main.py \
  --dataset_selector=babi \
  --data_dir=datasets/bAbI/tasks_1-20_v1-2/en/ \
  --babi_joint=True \
  --weight_tying_scheme=adj \
  --number_of_hops=$number_of_hops \
  --embedding_dim=50 \
  --position_encoding=True \
  --linear_start=False \
  --epochs=1 \
  --initial_learning_rate=0.005 \
  --random_noise=False \
  --checkpoint_dir=$CHECKPOINT_DIR \
  --vocab_dir=$VOCAB_DIR \
  --mode=viz \
  --load=True