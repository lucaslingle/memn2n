End-to-End Memory Networks in Tensorflow
========================================
Tensorflow implementation of [End-to-End Memory Networks](https://arxiv.org/abs/1503.08895).<br>
The original code from Facebook using Matlab and Lua can be found [here](https://github.com/facebook/MemNN).<br>  

![alt tag](assets/memn2n_small.png?raw=true)

Current implementation:<br>  

| Section       | Description                                                                    | Status  |
| ------------- |--------------------------------------------------------------------------------| --------|
| Section 2     | End-to-End Memory Network model                                                | Done    |
| Section 4     | Synthetic Question Answering experiments using the Facebook bAbI dataset.      | Done    |
| Section 5     | Language Modeling experiments on the Penn Treebank dataset.                    | Soon!   |

<br>

Getting started
---------------

Dependencies
------------
Install Anaconda 3, if you don't have it already.<br>
Create a new conda environment using the dependencies listed in memn2n_env.yml:

```
$ conda env create -f memn2n_env.yml
```

And activate the environment:
```
$ source activate memn2n_env
```

Data
--------
Create a directory for the data:
```
mkdir datasets/bAbI/
```

Download the bAbI dataset to that directory, using the link provided on the following page:<br>
https://research.fb.com/downloads/babi/
<br>

Then unzip the dataset:
```
$ cd datasets/bAbI/
$ tar -zxvf tasks_1-20_v1-2.tar.gz
```

Usage
-----

To train the memory network with 3 hops and memory size of 50, run the following:  
```
$ python main.py \
    --dataset_selector=babi \
    --babi_joint=True \
    --number_of_hops=3 \
    --number_of_memories=50 \
    --mode=train

```

To see all configuration options, run:  

```
$ python main.py --help
```

And you'll see some options like this:
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

Replicating the results in the paper
------------------------------------

For the bAbI tasks, the best performing model on the 1k dataset, in terms of mean error percentage, used the following configuration:  

- Adjacent weight tying scheme
- 3 hops
- Position Encoding (PE)
- Linear Start (LS)
- Random Noise (RN)
- joint training on all 20 bAbI tasks 
- 60 epochs
- 15 epochs per annealing
- 0.5 annealing constant
- LS training: the linear phase ends when the validation error rate stops falling 
- LS training: 0.005 initial learning rate during the linear phase
- LS training: 0.01 initial learning rate during the softmax phase
<br>

Below are our error rates on the test set, when using this configuration.

NOTE: 
  Paper results on the test set are based on the model with the best results on the training set, out of 10 random initializations.  

  Our results on the test set are based on only training one model, so the error rate tends to be at least a bit higher, and sometimes significantly so. 
  In particular, the error rates on tasks 15 and 16 differ dramatically from what was reported in the paper. 

| Task                     | Paper Result  | Our Result  |
|--------------------------|---------------|-------------|
| 1: 1 supporting fact     |           0.0 |         0.0 |
| 2: 2 supporting facts    |          11.4 |        14.6 |
| 3: 3 supporting facts    |          21.9 |        30.2 |
| 4: 2 argument relations  |          13.4 |         5.9 |
| 5: 3 argument relations  |          14.4 |        13.6 |
| 6: yes/no questions      |           2.8 |         2.9 |
| 7: counting              |          18.3 |        15.8 |
| 8: lists/sets            |           9.3 |         9.4 |
| 9: simple negation       |           1.9 |         2.3 |
| 10: indefinite knowledge |           6.5 |         6.0 |
| 11: basic coreference    |           0.3 |         1.2 |
| 12: conjunction          |           0.1 |         0.1 |
| 13: compound coreference |           0.2 |         1.1 |
| 14: time reasoning       |           6.9 |         6.7 |
| 15: basic deduction      |           0.0 |        51.9 |
| 16: basic induction      |           2.7 |         2.1 |
| 17: positional reasoning |          40.4 |        44.2 |
| 18: size reasoning       |           9.4 |        10.4 |
| 19: path finding         |          88.0 |        89.8 |
| 20: agent's motivation   |           0.0 |         0.0 |
|--------------------------|---------------|-------------|
| Mean Error (%)           |          12.4 |        15.4 |
| Failed tasks (err. > 5%) |            11 |          12 |

A script to train a model with this configuration is provided. 
You may run it as follows:
```
export TRIAL_ID=my_trial_1
source scripts/run_joint_trial_i.sh
```

And when it is done, you can test your results by running:
```
source scripts/test_each_task_for_trial_i.sh
```



