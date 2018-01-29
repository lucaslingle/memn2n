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
- jointly trained on all 20 bAbI tasks. 
- 60 epochs
- 15 epochs per annealing
- 0.5 annealing constant
- linear phase of LS training ends when the validation error rate stops falling 
<br>

The script to implement this configuration can be found below:  
<br>

<details>
  <summary>Click to expand</summary>

```
# linear start component

python main.py \
  --dataset_selector=babi \
  --data_dir=datasets/bAbI/tasks_1-20_v1-2/en/ \
  --babi_joint=True \
  --position_encoding=True \
  --linear_start=True \
  --initial_learning_rate=0.005 \
  --random_noise=True \
  --epochs=20 \
  --embedding_dim=50 \
  --anneal_epochs=15 \
  --model_name=MemN2N_bAbI_joint_adj_3hop_pe_ls_rn \
  --mode=train \
  --load=False

# after this runs, run the training with softmaxes reintroduced:

python main.py \
  --dataset_selector=babi \
  --data_dir=datasets/bAbI/tasks_1-20_v1-2/en/ \
  --babi_joint=True \
  --position_encoding=True \
  --linear_start=False \
  --initial_learning_rate=0.01 \
  --random_noise=True \
  --epochs=40 \
  --embedding_dim=50 \
  --anneal_epochs=15 \
  --model_name=MemN2N_bAbI_joint_adj_3hop_pe_ls_rn \
  --mode=train \
  --load=True

# and test on the joint bAbI tasks by running

python main.py \
  --dataset_selector=babi \
  --data_dir=datasets/bAbI/tasks_1-20_v1-2/en/ \
  --babi_joint=True \
  --position_encoding=True \
  --linear_start=False \
  --random_noise=False \
  --embedding_dim=50 \
  --model_name=MemN2N_bAbI_joint_adj_3hop_pe_ls_rn \
  --mode=test \
  --load=True
```
</details>
<br>


In addition, there were some ambiguities in the paper, and the script above resolves them, in what I believe to be the correct way. 
You can find the details of my thinking below.  
<br>

<details>
  <summary>Click to expand</summary>

```

1. - Ambiguity: 
     What is the frequency that the validation error rate should be checked, when deciding when to end the linear phase. 

   - Resolution: 
     I checked after every epoch. Empirically, I have found that 20 epochs works fairly well for the linear phase of LS training.

2. - Ambiguity: 
     The paper asserts, without any caveats, that they use an initial learning rate of 0.01. 
     The paper asserts later on that during the linear training phase of LS training, the initial learning rate is 0.005. 
     What learning rate is used during the softmax phase of LS training? 
     Is it the annealed value from the linear training phase, or is it 0.01?

   - Resolution: 
     Empirically, I have found that the softmax training phase of LS training requires a higher learning rate than the annealed rate from the linear training phase.
     In addition, Facebook's official implementation appears to use a separate learning rate for the softmax training phase of LS training.
     I have therefore opted to use an initial learning rate of 0.01 during the softmax training phase of LS training.

3. - Ambiguity:
     Regarding random noise training, what is meant by "10% of empty memories"? 
   - Resolution:
     Based on Facebook's official implementation, it appears to mean that the number of empty memories to be interspersed should be 10% of the number of nonempty memories. 
     It does not mean 10% of the total number of empty memories. 

4. - Ambiguity:
     The random noise must be interspersed uniformly throughout the nonempty memories, but must be capped at exactly 10% of the number of nonempties. 
     How best to do this? 
   - Resolution:
     Facebook's official implementation achieves this by using a random permutation to generate the target memory locations of the nonempty memories. 
     This can be done in a manner that preserves the original order of the nonempty memories. This implementation follows the same approach. 
```
</details>

