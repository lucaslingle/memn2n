export REPO_DIR=/Users/lucaslingle/git/memn2n

export TASK_ID=$1
export number_of_hops=3
export TRIAL_DIRNAME=trial_${TRIAL_ID}__bAbI_singletask${TASK_ID}_adj_${number_of_hops}hop_pe_ls_rn

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
  --babi_joint=False \
  --babi_task_id=$TASK_ID \
  --weight_tying_scheme=adj \
  --number_of_hops=$number_of_hops \
  --embedding_dim=20 \
  --position_encoding=True \
  --linear_start=True \
  --epochs=20 \
  --anneal_epochs=21 \
  --initial_learning_rate=0.005 \
  --random_noise=True \
  --gradient_clip=40 \
  --gradient_noise_scale=0.007 \
  --model_name=MemN2N_bAbI_singletask${TASK_ID}_adj_${number_of_hops}hop_pe_ls_rn \
  --checkpoint_dir=$CHECKPOINT_DIR \
  --vocab_dir=$VOCAB_DIR \
  --mode=train \
  --load=False

# Training with linear start. Softmax phase with official configs. 

python main.py \
  --dataset_selector=babi \
  --data_dir=datasets/bAbI/tasks_1-20_v1-2/en/ \
  --babi_joint=False \
  --babi_task_id=$TASK_ID \
  --weight_tying_scheme=adj \
  --number_of_hops=3 \
  --embedding_dim=20 \
  --position_encoding=True \
  --linear_start=False \
  --epochs=100 \
  --anneal_epochs=25 \
  --initial_learning_rate=0.005 \
  --random_noise=True \
  --gradient_clip=40 \
  --gradient_noise_scale=0.001 \
  --model_name=MemN2N_bAbI_singletask${TASK_ID}_adj_${number_of_hops}hop_pe_ls_rn \
  --checkpoint_dir=$CHECKPOINT_DIR \
  --vocab_dir=$VOCAB_DIR \
  --mode=train \
  --load=True

# Test the trained model:

python main.py \
  --dataset_selector=babi \
  --data_dir=datasets/bAbI/tasks_1-20_v1-2/en/ \
  --babi_joint=False \
  --babi_task_id=$TASK_ID \
  --weight_tying_scheme=adj \
  --number_of_hops=3 \
  --embedding_dim=20 \
  --position_encoding=True \
  --linear_start=False \
  --random_noise=False \
  --model_name=MemN2N_bAbI_singletask${TASK_ID}_adj_${number_of_hops}hop_pe_ls_rn \
  --checkpoint_dir=$CHECKPOINT_DIR \
  --vocab_dir=$VOCAB_DIR \
  --mode=test \
  --load=True