export TRIAL_DIR=trial_${TRIAL_ID}__bAbI_joint_adj_3hop_pe_ls_rn

# Training with linear start. Linear phase with official configs. 

python main.py \
  --dataset_selector=babi \
  --data_dir=datasets/bAbI/tasks_1-20_v1-2/en/ \
  --babi_joint=True \
  --weight_tying_scheme=adj \
  --number_of_hops=3 \
  --embedding_dim=50 \
  --position_encoding=True \
  --linear_start=True \
  --epochs=20 \
  --anneal_epochs=21 \
  --initial_learning_rate=0.005 \
  --random_noise=True \
  --gradient_clip=40 \
  --gradient_noise_scale=0.00025 \
  --word_emb_initializer=xavier_normal_initializer \
  --temporal_emb_initializer=xavier_normal_initializer \
  --model_name=MemN2N_bAbI_joint_adj_3hop_pe_ls_rn \
  --checkpoint_dir=/Users/lucaslingle/git/memn2n/checkpoints/$TRIAL_DIR \
  --mode=train \
  --load=False

# Training with linear start. Softmax phase with official configs. 

python main.py \
  --dataset_selector=babi \
  --data_dir=datasets/bAbI/tasks_1-20_v1-2/en/ \
  --babi_joint=True \
  --weight_tying_scheme=adj \
  --number_of_hops=3 \
  --embedding_dim=50 \
  --position_encoding=True \
  --linear_start=False \
  --epochs=60 \
  --anneal_epochs=15 \
  --initial_learning_rate=0.005 \
  --random_noise=True \
  --gradient_clip=40 \
  --gradient_noise_scale=0.00025 \
  --word_emb_initializer=xavier_normal_initializer \
  --temporal_emb_initializer=xavier_normal_initializer \
  --model_name=MemN2N_bAbI_joint_adj_3hop_pe_ls_rn \
  --checkpoint_dir=/Users/lucaslingle/git/memn2n/checkpoints/$TRIAL_DIR \
  --mode=train \
  --load=True

# Test the trained model:

python main.py \
  --dataset_selector=babi \
  --data_dir=datasets/bAbI/tasks_1-20_v1-2/en/ \
  --babi_joint=True \
  --weight_tying_scheme=adj \
  --number_of_hops=3 \
  --embedding_dim=50 \
  --position_encoding=True \
  --linear_start=False \
  --random_noise=False \
  --model_name=MemN2N_bAbI_joint_adj_3hop_pe_ls_rn \
  --checkpoint_dir=/Users/lucaslingle/git/memn2n/checkpoints/$TRIAL_DIR \
  --mode=test \
  --load=True