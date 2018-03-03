export REPO_DIR=/Users/lucaslingle/git/memn2n

export number_of_hops=3
export TRIAL_DIRNAME=trial_${TRIAL_ID}__bAbI_joint_adj_${number_of_hops}hop_pe_ls_rn

export CHECKPOINT_DIR=${REPO_DIR}/checkpoints/${TRIAL_DIRNAME}
export VOCAB_DIR=${REPO_DIR}/vocab/${TRIAL_DIRNAME}

for ((i=1; i<=20; i++)); do
   export result=$(python main.py \
    --dataset_selector=babi \
    --data_dir=datasets/bAbI/tasks_1-20_v1-2/en/ \
    --babi_joint=False \
    --babi_task_id=$i \
    --position_encoding=True \
    --linear_start=False \
    --random_noise=False \
    --embedding_dim=50 \
    --mode=test \
    --load=True \
    --model_name=MemN2N_bAbI_joint_adj_${number_of_hops}hop_pe_ls_rn \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --vocab_dir=$VOCAB_DIR \
    --vocab_filename=vocab_babi_en_joint.pkl \
    --max_sentence_len_filename=max_sentence_len_babi_en_joint.pkl \
   2> /dev/null \
   | fgrep error_rate);
  echo task $i result: $result
done