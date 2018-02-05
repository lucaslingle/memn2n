export TRIAL_DIR=trial_${TRIAL_ID}__bAbI_joint_adj_3hop_pe_ls_rn

for ((i=1; i<=20; i++)); do
   python main.py \
    --dataset_selector=babi \
    --data_dir=datasets/bAbI/tasks_1-20_v1-2/en/ \
    --babi_joint=False \
    --babi_task_id=$i \
    --position_encoding=True \
    --linear_start=False \
    --random_noise=False \
    --embedding_dim=50 \
    --model_name=MemN2N_bAbI_joint_adj_3hop_pe_ls_rn \
    --checkpoint_dir=/Users/lucaslingle/git/memn2n/checkpoints/$TRIAL_DIR \
    --mode=test \
    --load=True \
    --vocab_filename=vocab_babi_en_joint.pkl \
   2> /dev/null \
   | fgrep error_rate 
done