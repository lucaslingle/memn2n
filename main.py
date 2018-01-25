import tensorflow as tf
from MemoryNetwork import MemoryNetwork
import babi_dataset_utils as bb
import os
import sys
import errno

import matplotlib.pyplot as plt
import seaborn as sns

flags = tf.app.flags

# dataset configs
flags.DEFINE_string("dataset_selector", "babi", "dataset selector: 'babi' or 'penn' [babi]")
flags.DEFINE_string("data_dir", 'datasets/bAbI/tasks_1-20_v1-2/en/', "Data directory [datasets/bAbI/tasks_1-20_v1-2/en/]")
flags.DEFINE_boolean("babi_joint", False, "run jointly on all bAbI tasks, if applicable [False]")
flags.DEFINE_integer("babi_task_id", 1, "bAbI task to train on, if applicable [1]")
flags.DEFINE_float("validation_frac", 0.1, "train-validation split [0.1]")
flags.DEFINE_string("vocab_dir", 'vocab/', "directory to persist vocab-int dictionary [vocab/]")

# checkpoint configs
flags.DEFINE_string("checkpoint_dir", "/Users/lucaslingle/git/memn2n/checkpoints/", "checkpoints path [/Users/lucaslingle/git/memn2n/checkpoints/]")
flags.DEFINE_string("model_name", "MemN2N", "a filename prefix for checkpoints [MemN2N]")
flags.DEFINE_string("mode", 'train', "train or test [train]")
flags.DEFINE_boolean("load", False, "load from latest checkpoint [False]")
flags.DEFINE_integer("save_freq_epochs", 5, "number of epochs between checkpoints [5]")

# training configs
flags.DEFINE_integer("batch_size", 32, "batch size [32]")
flags.DEFINE_integer("epochs", 100, "number of epochs [100]")
flags.DEFINE_float("initial_learning_rate", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("gradient_clip", 40, "maximum gradient norm [40]")
flags.DEFINE_float("anneal_const", 0.5, "annealing constant [0.5]")
flags.DEFINE_integer("anneal_epochs", 25, "number of epochs per annealing [25]")

# model configs
flags.DEFINE_integer("number_of_memories", 50, "memory size [50]")
flags.DEFINE_integer("embedding_dim", 20, "word embedding dimension [20]")
flags.DEFINE_integer("number_of_hops", 3, "number of hops [3]")
flags.DEFINE_boolean("linear_start", False, "start with linear attention (as opposed to softmaxed) [False]")
flags.DEFINE_boolean("position_encoding", True, "position encoding [True]")
flags.DEFINE_string("weight_tying_scheme", 'adj', "weight tying scheme: 'adj' or 'rnnlike' [adj]")
flags.DEFINE_boolean("random_noise", False, "random noise (insert empty memories to regularize temporal embedding) [False]")

FLAGS = flags.FLAGS

def compute_and_save_babi_vocab(data_dir, save_fp):

    babi = bb.bAbI()

    # compute and save a vocab dict that covers all bAbI tasks.
    _, _, _ = babi.prepare_data_for_joint_tasks(data_dir=data_dir, validation_frac=0.0, vocab_dict=None)

    babi.save_vocab_dict_to_file(vocab_dict=babi.vocab_dict, vocab_fp=save_fp)

    # load our vocab dictionary
    vocab_fp_exists = os.path.exists(save_fp)
    if not vocab_fp_exists:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), vocab_fp_exists)

def main():

    if FLAGS.dataset_selector == 'babi':

        babi = bb.bAbI()
        learning_rate = FLAGS.initial_learning_rate

        candidate_vocab_fp = os.path.join(FLAGS.vocab_dir, 'vocab_{}.pkl'.format(FLAGS.dataset_selector))
        vocab_fp_exists = os.path.exists(candidate_vocab_fp)

        # prepare vocab if it doesn't exist
        if not vocab_fp_exists:
            if FLAGS.load:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), vocab_fp_exists)

            compute_and_save_babi_vocab(FLAGS.data_dir, candidate_vocab_fp)

        with tf.Graph().as_default() as graph:

            # load our vocab dictionary
            vocab_dict = babi.load_vocab_dict_from_file(candidate_vocab_fp)

            # prepare the data, and store the max sentence length
            if FLAGS.babi_joint:
                train, val, test = babi.prepare_data_for_joint_tasks(
                    FLAGS.data_dir, FLAGS.validation_frac, vocab_dict=vocab_dict)
            else:
                train, val, test = babi.prepare_data_for_single_task(
                    FLAGS.data_dir, FLAGS.babi_task_id, FLAGS.validation_frac, vocab_dict=vocab_dict)

            # instantiate the model
            model = MemoryNetwork(vocab_size=len(vocab_dict),
                              embedding_dim=FLAGS.embedding_dim,
                              number_of_hops=FLAGS.number_of_hops,
                              batch_size=FLAGS.batch_size,
                              number_of_memories=FLAGS.number_of_memories,
                              max_sentence_len=babi.max_sentence_len,
                              gradient_clip=FLAGS.gradient_clip,
                              weight_tying_scheme=FLAGS.weight_tying_scheme,
                              position_encoding=FLAGS.position_encoding,
                              linear_start=FLAGS.linear_start)

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            if FLAGS.load:
                print("attempting to restore from {}".format(FLAGS.checkpoint_dir))
                model.load(sess, FLAGS.checkpoint_dir)

            nr_training_examples = len(train)
            nr_validation_examples = len(val)
            nr_test_examples = len(test)

            if FLAGS.mode == 'train':

                for epoch in range(0, FLAGS.epochs):
                    for i in range(0, nr_training_examples, FLAGS.batch_size):

                        if (i + FLAGS.batch_size) > nr_training_examples:
                            break

                        start_idx = i
                        end_idx = i + FLAGS.batch_size

                        sqa_batch = train[start_idx:end_idx]

                        sqa_batch_standardized = list(map(
                            lambda sqa: bb.bAbI.standardize_features(
                                sqa,
                                babi.max_sentence_len,
                                FLAGS.number_of_memories,
                                babi.vocab_dict[babi.pad_token],
                                add_empty_memories=FLAGS.random_noise
                            ),
                            sqa_batch
                        ))

                        sentences_ints, question_ints, answer_ints = zip(*sqa_batch_standardized)

                        feed_dict = {
                            model.sentences_ints_batch: sentences_ints,
                            model.question_ints_batch: question_ints,
                            model.answer_ints_batch: answer_ints,
                            model.learning_rate: learning_rate
                        }

                        _, loss, acc = sess.run(
                            [model.train_op, model.summed_cross_entropy_batch, model.acc_batch],
                            feed_dict=feed_dict
                        )

                        mean_cross_entropy = loss / float(FLAGS.batch_size)

                        print("epoch {}, iter {}, batch mean_cross_entropy {}, batch accuracy {}".format(
                            epoch, i, mean_cross_entropy, acc
                        ))

                    if (epoch > 0) and (epoch % FLAGS.anneal_epochs) == 0:
                        learning_rate *= FLAGS.anneal_const
                    if (epoch > 0) and (epoch % FLAGS.save_freq_epochs) == 0:
                        model.save(sess, FLAGS.checkpoint_dir)

                model.save(sess, FLAGS.checkpoint_dir)

                print("finished training!")

                sum_cross_entropy = 0
                nr_correct = 0

                if nr_validation_examples == 0:
                    print("no validation examples. exiting now.")
                    sys.exit(0)

                for i in range(0, nr_validation_examples, FLAGS.batch_size):

                    if (i + FLAGS.batch_size) > nr_validation_examples:
                        break

                    start_idx = i
                    end_idx = i + FLAGS.batch_size

                    sqa_batch = val[start_idx:end_idx]

                    sqa_batch_standardized = list(map(
                        lambda sqa: bb.bAbI.standardize_features(
                            sqa,
                            babi.max_sentence_len,
                            FLAGS.number_of_memories,
                            babi.vocab_dict[babi.pad_token],
                            add_empty_memories=FLAGS.random_noise
                        ),
                        sqa_batch
                    ))

                    sentences_ints, question_ints, answer_ints = zip(*sqa_batch_standardized)

                    feed_dict = {
                        model.learning_rate: 0.0,
                        model.sentences_ints_batch: sentences_ints,
                        model.question_ints_batch: question_ints,
                        model.answer_ints_batch: answer_ints
                    }

                    _, loss, acc = sess.run(
                        [model.train_op, model.summed_cross_entropy_batch, model.acc_batch],
                        feed_dict=feed_dict
                    )

                    mean_cross_entropy = loss / float(FLAGS.batch_size)

                    sum_cross_entropy += loss
                    nr_correct += int(acc * FLAGS.batch_size)

                    print("validation set, iter {}, batch mean_cross_entropy {}, batch accuracy {}".format(
                        i, mean_cross_entropy, acc
                    ))

                mean_cross_entropy = sum_cross_entropy / float(nr_validation_examples)
                accuracy = nr_correct / float(nr_validation_examples)
                error_rate = 1.0 - accuracy

                print("mean cross_entropy on validation set: {}, \naccuracy: {}, \nerror_rate{}".format(
                    mean_cross_entropy, accuracy, error_rate
                ))

            if FLAGS.mode == 'test':

                sum_cross_entropy = 0
                nr_correct = 0

                for epoch in range(0, 1):
                    for i in range(0, nr_test_examples, FLAGS.batch_size):

                        if (i + FLAGS.batch_size) > nr_test_examples:
                            break

                        start_idx = i
                        end_idx = i + FLAGS.batch_size

                        sqa_batch = test[start_idx:end_idx]

                        sqa_batch_standardized = list(map(
                            lambda sqa: bb.bAbI.standardize_features(
                                sqa,
                                babi.max_sentence_len,
                                FLAGS.number_of_memories,
                                babi.vocab_dict[babi.pad_token],
                                add_empty_memories=FLAGS.random_noise
                            ),
                            sqa_batch
                        ))

                        sentences_ints, question_ints, answer_ints = zip(*sqa_batch_standardized)

                        feed_dict = {
                            model.learning_rate: 0.0,
                            model.sentences_ints_batch: sentences_ints,
                            model.question_ints_batch: question_ints,
                            model.answer_ints_batch: answer_ints
                        }

                        _, loss, acc = sess.run(
                            [model.train_op, model.summed_cross_entropy_batch, model.acc_batch],
                            feed_dict=feed_dict
                        )

                        mean_cross_entropy = loss / float(FLAGS.batch_size)

                        sum_cross_entropy += loss
                        nr_correct += int(acc * FLAGS.batch_size)

                        print("test set, iter {}, batch mean_cross_entropy {}, batch accuracy {}".format(
                            i, mean_cross_entropy, acc
                        ))

                mean_cross_entropy = sum_cross_entropy / float(nr_test_examples)
                accuracy = nr_correct / float(nr_test_examples)
                error_rate = 1.0 - accuracy

                print("mean cross_entropy on test set: {}, \naccuracy: {}, \nerror_rate{}".format(
                    mean_cross_entropy, accuracy, error_rate
                ))


main()
