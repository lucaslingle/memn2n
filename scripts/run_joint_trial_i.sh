export REPO_DIR=/Users/lucaslingle/git/memn2n

export number_of_hops=3
export TRIAL_DIRNAME=trial_${TRIAL_ID}__bAbI_joint_adj_${number_of_hops}hop_pe_ls_rn

export CHECKPOINT_DIR=${REPO_DIR}/checkpoints/${TRIAL_DIRNAME}
export VOCAB_DIR=${REPO_DIR}/vocab/${TRIAL_DIRNAME}

if [ -d "$REPO_DIR" ]; then
    echo "using $REPO_DIR as REPO_DIR"
    else echo "ERROR:"; 
         echo "  $REPO_DIR does not exist!"; 
         echo "  Please change script to indicate correct REPO_DIR"; 
         return
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "creating trial CHECKPOINT_DIR: $CHECKPOINT_DIR";
    mkdir -p "$CHECKPOINT_DIR";
    else echo "ERROR:"; 
         echo "  $CHECKPOINT_DIR already exists!"; 
         echo "  This script trains a model from scratch.";
         echo "  To avoid overwriting checkpoints in a previous trial's CHECKPOINT_DIR,"; 
         echo "  please set a new value for the TRIAL_ID environment variable.";
         return
fi

if [ ! -d "$VOCAB_DIR" ]; then
    echo "creating trial VOCAB_DIR: $VOCAB_DIR"; 
    mkdir -p "$VOCAB_DIR";
    else echo "ERROR:"; 
         echo "  $VOCAB_DIR already exists!";
         echo "  This script trains a model from scratch, and computes a random train/validation split each time."; 
         echo "  This could result in a different vocab dictionary.";
         echo "  To avoid overwriting the persisted vocab dictionary in a previous trial's VOCAB_DIR,";
         echo "  please set a new value for the TRIAL_ID environment variable.";
         return
fi


# Training with linear start. Linear phase with official configs.

python main.py \
  --dataset_selector=babi \
  --data_dir=datasets/bAbI/tasks_1-20_v1-2/en/ \
  --babi_joint=True \
  --weight_tying_scheme=adj \
  --number_of_hops=$number_of_hops \
  --embedding_dim=50 \
  --position_encoding=True \
  --linear_start=True \
  --epochs=30 \
  --anneal_epochs=31 \
  --initial_learning_rate=0.005 \
  --random_noise=True \
  --gradient_clip=40 \
  --gradient_noise_scale=0.000 \
  --model_name=MemN2N_bAbI_joint_adj_${number_of_hops}hop_pe_ls_rn \
  --checkpoint_dir=$CHECKPOINT_DIR \
  --vocab_dir=$VOCAB_DIR \
  --mode=train \
  --load=False

# Training with linear start. Softmax phase with official configs. 

python main.py \
  --dataset_selector=babi \
  --data_dir=datasets/bAbI/tasks_1-20_v1-2/en/ \
  --babi_joint=True \
  --weight_tying_scheme=adj \
  --number_of_hops=$number_of_hops \
  --embedding_dim=50 \
  --position_encoding=True \
  --linear_start=False \
  --epochs=60 \
  --anneal_epochs=15 \
  --initial_learning_rate=0.005 \
  --random_noise=True \
  --gradient_clip=40 \
  --gradient_noise_scale=0.000 \
  --model_name=MemN2N_bAbI_joint_adj_${number_of_hops}hop_pe_ls_rn \
  --checkpoint_dir=$CHECKPOINT_DIR \
  --vocab_dir=$VOCAB_DIR \
  --mode=train \
  --load=True

# Test the trained model:

python main.py \
  --dataset_selector=babi \
  --data_dir=datasets/bAbI/tasks_1-20_v1-2/en/ \
  --babi_joint=True \
  --weight_tying_scheme=adj \
  --number_of_hops=$number_of_hops \
  --embedding_dim=50 \
  --position_encoding=True \
  --linear_start=False \
  --random_noise=False \
  --model_name=MemN2N_bAbI_joint_adj_${number_of_hops}hop_pe_ls_rn \
  --checkpoint_dir=$CHECKPOINT_DIR \
  --vocab_dir=$VOCAB_DIR \
  --mode=test \
  --load=True