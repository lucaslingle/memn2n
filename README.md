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
<details>
  <summary>click to expand view</summary>

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
</details>


Best performing model in the paper
------------------------------------

In the paper's results, the best performance on the bAbI 1k dataset, as measured by the test error rate averaged across all tasks, used the following configuration:  

- One model, trained jointly on all 20 bAbI tasks
- Adjacent weight tying scheme
- 3 hops
- Position Encoding (PE)
- Linear Start (LS)
- Random Noise (RN)
- Annealing constant: 0.5
- LS linear phase initial learning rate: 0.005 
- LS linear phase epochs: 30
- LS linear phase annealing period: 31
- LS softmax phase initial learning rate: 0.005
- LS softmax phase epochs: 60
- LS softmax phase annealing period: 15
<br>

Some of the details in the paper for LS training are presented ambiguously, and the ambiguity was resolved by deferring to the hyperparameters used in Facebook's matlab implementation of LS training.

Replicating the results of the best performing model
-----------------------------------------------------

Paper results:
- Results were computed by the authors by running 10 trials. 
- In each trial, the given model architecture is trained according to the given training regime.
- In each trial, the weights are initialized randomly, and the data is shuffled randomly. 
- The instance of the trained model with the lowest training error is then evaluated on the test set, and its results were reported.
- For comparison, these results are shown below.

Our results:
- Results computed for my implementation using the most up-to-date code in this repo.
- In contrast to the paper, these results were obtained via a single trial using the most up-to-date code.

| Task                     | Paper Result  | Our Result  |
|:-------------------------|--------------:|------------:|
| 1: 1 supporting fact     |           0.0 |         0.3 |
| 2: 2 supporting facts    |          11.4 |         8.8 |
| 3: 3 supporting facts    |          21.9 |        24.6 |
| 4: 2 argument relations  |          13.4 |         2.8 |
| 5: 3 argument relations  |          14.4 |        13.9 |
| 6: yes/no questions      |           2.8 |         4.0 |
| 7: counting              |          18.3 |        17.4 |
| 8: lists/sets            |           9.3 |        11.2 |
| 9: simple negation       |           1.9 |         3.3 |
| 10: indefinite knowledge |           6.5 |         6.7 |
| 11: basic coreference    |           0.3 |         1.0 |
| 12: conjunction          |           0.1 |         0.8 |
| 13: compound coreference |           0.2 |         0.4 |
| 14: time reasoning       |           6.9 |         7.3 |
| 15: basic deduction      |           0.0 |         0.0 |
| 16: basic induction      |           2.7 |         3.5 |
| 17: positional reasoning |          40.4 |        42.1 |
| 18: size reasoning       |           9.4 |         9.4 |
| 19: path finding         |          88.0 |        91.3 |
| 20: agent's motivation   |           0.0 |         0.5 |
|--------------------------|---------------|-------------|
| Mean Error (%)           |          12.4 |        12.5 |
| Failed tasks (err. > 5%) |            11 |          10 |

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

