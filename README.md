# memn2n
Tensorflow implementation of [End-to-End Memory Networks](https://arxiv.org/abs/1503.08895)

Current implementation includes:
   Section 2: End-to-End Memory Network model
   Section 4: Synthetic Question Answering experiments using the Facebook bAbI dataset. 

Coming soon:
   Section 5: Language Modeling experiments on the Penn Treebank dataset.

```
usage: main.py [-h] [--dataset_selector DATASET_SELECTOR]
               [--data_dir DATA_DIR] [--babi_joint [BABI_JOINT]]
               [--babi_task_id BABI_TASK_ID]
               [--validation_frac VALIDATION_FRAC] [--vocab_dir VOCAB_DIR]
               [--checkpoint_dir CHECKPOINT_DIR] [--model_name MODEL_NAME]
               [--mode MODE] [--load [LOAD]]
               [--save_freq_epochs SAVE_FREQ_EPOCHS] [--batch_size BATCH_SIZE]
               [--epochs EPOCHS]
               [--initial_learning_rate INITIAL_LEARNING_RATE]
               [--gradient_clip GRADIENT_CLIP] [--anneal_const ANNEAL_CONST]
               [--anneal_epochs ANNEAL_EPOCHS]
               [--number_of_memories NUMBER_OF_MEMORIES]
               [--embedding_dim EMBEDDING_DIM]
               [--number_of_hops NUMBER_OF_HOPS]
               [--linear_start [LINEAR_START]]
               [--position_encoding [POSITION_ENCODING]]
               [--weight_tying_scheme WEIGHT_TYING_SCHEME]
               [--random_noise [RANDOM_NOISE]]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_selector DATASET_SELECTOR
                        dataset selector: 'babi' or 'penn' [babi]
  --data_dir DATA_DIR   Data directory [datasets/bAbI/tasks_1-20_v1-2/en/]
  --babi_joint [BABI_JOINT]
                        run jointly on all bAbI tasks, if applicable [False]
  --babi_task_id BABI_TASK_ID
                        bAbI task to train on, if applicable [1]
  --validation_frac VALIDATION_FRAC
                        train-validation split [0.1]
  --vocab_dir VOCAB_DIR
                        directory to persist vocab-int dictionary [vocab/]
  --checkpoint_dir CHECKPOINT_DIR
                        checkpoints path
                        [/Users/lucaslingle/git/memn2n/checkpoints/]
  --model_name MODEL_NAME
                        a filename prefix for checkpoints [MemN2N]
  --mode MODE           train or test [train]
  --load [LOAD]         load from latest checkpoint [False]
  --save_freq_epochs SAVE_FREQ_EPOCHS
                        number of epochs between checkpoints [5]
  --batch_size BATCH_SIZE
                        batch size [32]
  --epochs EPOCHS       number of epochs [100]
  --initial_learning_rate INITIAL_LEARNING_RATE
                        initial learning rate [0.01]
  --gradient_clip GRADIENT_CLIP
                        maximum gradient norm [40]
  --anneal_const ANNEAL_CONST
                        annealing constant [0.5]
  --anneal_epochs ANNEAL_EPOCHS
                        number of epochs per annealing [25]
  --number_of_memories NUMBER_OF_MEMORIES
                        memory size [50]
  --embedding_dim EMBEDDING_DIM
                        word embedding dimension [20]
  --number_of_hops NUMBER_OF_HOPS
                        number of hops [3]
  --linear_start [LINEAR_START] start with linear attention (as opposed to softmaxed) [False]
  --position_encoding [POSITION_ENCODING] position encoding [True]
  --weight_tying_scheme WEIGHT_TYING_SCHEME
                        weight tying scheme: 'adj' or 'rnnlike' [adj]
  --random_noise [RANDOM_NOISE]
                        random noise (insert empty memories to regularize
                        temporal embedding) [False]

```