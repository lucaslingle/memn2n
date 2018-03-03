import tensorflow as tf
from MemoryNetwork import MemoryNetwork
import babi_dataset_utils as bb
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import errno

flags = tf.app.flags

# dataset configs
flags.DEFINE_string("dataset_selector", "babi", "dataset selector: 'babi' or 'penn' [babi]")
flags.DEFINE_string("data_dir", 'datasets/bAbI/tasks_1-20_v1-2/en/', "Data directory [datasets/bAbI/tasks_1-20_v1-2/en/]")
flags.DEFINE_boolean("babi_joint", False, "run jointly on all bAbI tasks, if applicable [False]")
flags.DEFINE_integer("babi_task_id", 1, "bAbI task to train on, if applicable [1]")
flags.DEFINE_float("validation_frac", 0.1, "train-validation split [0.1]")
flags.DEFINE_string("vocab_dir", 'vocab/', "directory to persist vocab-int dictionary [vocab/]")
flags.DEFINE_string("vocab_filename", "", "optional flag to allow us to load a persisted vocab dictionary from a pkl file")
flags.DEFINE_string("max_sentence_len_filename", "", "optional flag to allow us to load a persisted max_sentence_len value from a pkl file")


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
flags.DEFINE_float("gradient_noise_scale", 0.001, "stddev for adding gaussian noise to gradient [0.001]")
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
flags.DEFINE_string("word_emb_initializer", 'random_normal_initializer', "weight initializer class name for word embedding weights. [random_normal_initializer]")
flags.DEFINE_float("word_emb_init_scale", 0.1, "value for stddev or gain argument of the word_emb_initializer [0.1]")
flags.DEFINE_string("temporal_emb_initializer", 'random_normal_initializer', "weight initializer class name for temporal embedding weights. [random_normal_initializer]")
flags.DEFINE_float("temporal_emb_init_scale", 0.1, "value for stddev or gain argument of the temporal_emb_initializer [0.1]")

FLAGS = flags.FLAGS


def get_vocab_filename_from_settings(FLAGS):
    if len(FLAGS.vocab_filename) > 0:
        candidate_vocab_filename = FLAGS.vocab_filename
        return candidate_vocab_filename

    candidate_vocab_filename = 'vocab_{}_{}_{}.pkl'.format(
            FLAGS.dataset_selector,
            FLAGS.data_dir.strip("/").split("/")[-1],
            'joint' if FLAGS.babi_joint else 'task_{}'.format(FLAGS.babi_task_id)
    )

    return candidate_vocab_filename

def get_max_sentence_len_filename_from_settings(FLAGS):
    if len(FLAGS.max_sentence_len_filename) > 0:
        candidate_max_sentence_len_filename = FLAGS.max_sentence_len_filename
        return candidate_max_sentence_len_filename

    candidate_max_sentence_len_filename = 'max_sentence_len_{}_{}_{}.pkl'.format(
            FLAGS.dataset_selector,
            FLAGS.data_dir.strip("/").split("/")[-1],
            'joint' if FLAGS.babi_joint else 'task_{}'.format(FLAGS.babi_task_id)
    )

    return candidate_max_sentence_len_filename

def compute_and_save_babi_vocab_meta(FLAGS, vocab_save_fp, max_sentence_len_save_fp):
    # compute and save a vocab dictionary as a pickle file

    babi = bb.bAbI()

    if FLAGS.babi_joint:
        _, _, _ = babi.prepare_data_for_joint_tasks(
            FLAGS.data_dir, FLAGS.validation_frac, vocab_dict=None, max_sentence_len=None)
    else:
        _, _, _ = babi.prepare_data_for_single_task(
            FLAGS.data_dir, FLAGS.babi_task_id, FLAGS.validation_frac, vocab_dict=None, max_sentence_len=None)

    babi.save_vocab_dict_to_file(data=babi.vocab_dict, fp=vocab_save_fp)
    babi.save_max_sentence_len_to_file(data=babi.max_sentence_len, fp=max_sentence_len_save_fp)
    return

def main():

    if FLAGS.dataset_selector == 'babi':

        babi = bb.bAbI()
        learning_rate = FLAGS.initial_learning_rate

        candidate_vocab_filename = get_vocab_filename_from_settings(FLAGS)
        candidate_max_sentence_len_filename = get_max_sentence_len_filename_from_settings(FLAGS)

        candidate_vocab_fp = os.path.join(FLAGS.vocab_dir, candidate_vocab_filename)
        vocab_fp_exists = os.path.exists(candidate_vocab_fp)

        candidate_max_sentence_len_fp = os.path.join(FLAGS.vocab_dir, candidate_max_sentence_len_filename)
        max_sentence_len_fp_exists = os.path.exists(candidate_max_sentence_len_fp)

        # must compute and persist vocab metadata if we aren't loading anything
        if not FLAGS.load:
            compute_and_save_babi_vocab_meta(FLAGS, candidate_vocab_fp, candidate_max_sentence_len_fp)

        if FLAGS.load and not vocab_fp_exists:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), candidate_vocab_fp)

        if FLAGS.load and not max_sentence_len_fp_exists:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), candidate_max_sentence_len_fp)

        with tf.Graph().as_default() as graph:

            # load our vocab dictionary
            vocab_dict = babi.load_vocab_dict_from_file(candidate_vocab_fp)
            max_sentence_len = babi.load_max_sentence_len_from_file(candidate_max_sentence_len_fp)

            # prepare the data
            if FLAGS.babi_joint:
                train, val, test = babi.prepare_data_for_joint_tasks(
                    FLAGS.data_dir, FLAGS.validation_frac,
                    vocab_dict=vocab_dict, max_sentence_len=max_sentence_len
                )
            else:
                train, val, test = babi.prepare_data_for_single_task(
                    FLAGS.data_dir, FLAGS.babi_task_id, FLAGS.validation_frac,
                    vocab_dict=vocab_dict, max_sentence_len=max_sentence_len
                )

            print("len(vocab_dict) is {}, and max_sentence_len is {}".format(
                len(vocab_dict), max_sentence_len
            ))

            # instantiate the model
            model = MemoryNetwork(vocab_size=len(vocab_dict),
                                  embedding_dim=FLAGS.embedding_dim,
                                  number_of_hops=FLAGS.number_of_hops,
                                  batch_size=FLAGS.batch_size,
                                  number_of_memories=FLAGS.number_of_memories,
                                  max_sentence_len=babi.max_sentence_len,
                                  gradient_clip=FLAGS.gradient_clip,
                                  gradient_noise_scale=FLAGS.gradient_noise_scale,
                                  weight_tying_scheme=FLAGS.weight_tying_scheme,
                                  position_encoding=FLAGS.position_encoding,
                                  word_emb_initializer=FLAGS.word_emb_initializer,
                                  word_emb_init_scale=FLAGS.word_emb_init_scale,
                                  temporal_emb_initializer=FLAGS.temporal_emb_initializer,
                                  temporal_emb_init_scale=FLAGS.temporal_emb_init_scale)

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            if FLAGS.load:
                print("attempting to restore from {}".format(FLAGS.checkpoint_dir))
                model.load(sess, FLAGS.checkpoint_dir)
                word_emb_varnames_in_restored_list_order = list(map(lambda v: v.name, model.embedding_matrices['word']))
                print(word_emb_varnames_in_restored_list_order)

            nr_training_examples = len(train)
            nr_validation_examples = len(val)
            nr_test_examples = len(test)

            if FLAGS.mode == 'train':
                print("mode: train")
                print("nr_training_examples: {}, nr_validation_examples: {}, nr_test_examples: {}".format(
                    nr_training_examples, nr_validation_examples, nr_test_examples
                ))

                for epoch in range(1, FLAGS.epochs + 1):

                    # reshuffle training data before commencing an epoch,
                    # get new batches each time rather than cycling thru previously seen batches
                    # (should improve quality of gradient estimates a bit)

                    np.random.shuffle(train)

                    for i in range(0, nr_training_examples, FLAGS.batch_size):
                        sqa_batch = None

                        start_idx = i
                        end_idx = i + FLAGS.batch_size

                        if (i + FLAGS.batch_size) > nr_training_examples:
                            _tmp = train[0:i][:]
                            sqa_tuples_to_pad_batch = list(filter(lambda sqa: sqa.story_task_id == 15, _tmp))
                            np.random.shuffle(sqa_tuples_to_pad_batch)
                            sqa_batch = train[start_idx:]
                            sqa_batch.extend(sqa_tuples_to_pad_batch[0:(FLAGS.batch_size-(nr_training_examples-start_idx))])
                        else:
                            sqa_batch = train[start_idx:end_idx]

                        sqa_batch_standardized = list(map(
                            lambda sqa: bb.bAbI.standardize_features(
                                sqa,
                                babi.max_sentence_len,
                                FLAGS.number_of_memories,
                                babi.vocab_dict[babi.pad_token],
                                intersperse_empty_memories=FLAGS.random_noise
                            ),
                            sqa_batch
                        ))

                        sentences_ints, sentences_timeword_ints, question_ints, answer_ints = zip(*sqa_batch_standardized)

                        feed_dict = {
                            model.linear_start_indicator: FLAGS.linear_start,
                            model.learning_rate: learning_rate,
                            model.sentences_ints_batch: sentences_ints,
                            model.sentences_timewords_ints_batch: sentences_timeword_ints,
                            model.question_ints_batch: question_ints,
                            model.answer_ints_batch: answer_ints,
                        }

                        _, loss, acc = sess.run(
                            [model.train_op, model.summed_cross_entropy_batch, model.acc_batch],
                            feed_dict=feed_dict
                        )

                        mean_cross_entropy = loss / float(FLAGS.batch_size)

                        print("epoch {}, iter {}, batch mean_cross_entropy {}, batch accuracy {}".format(
                            epoch, i, mean_cross_entropy, acc
                        ))

                    if epoch > 1 and (epoch % FLAGS.anneal_epochs) == 0:
                        learning_rate *= FLAGS.anneal_const
                    if epoch > 1 and (epoch % FLAGS.save_freq_epochs) == 0:
                        model.save(
                            session=sess,
                            checkpoint_dir=FLAGS.checkpoint_dir,
                            checkpoint_name='{}_epoch{}'.format(FLAGS.model_name, epoch)
                        )

                model.save(
                    session=sess,
                    checkpoint_dir=FLAGS.checkpoint_dir,
                    checkpoint_name='{}_epoch{}'.format(FLAGS.model_name, FLAGS.epochs)
                )

                print("finished training!")

                sum_cross_entropy = 0.0
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
                            intersperse_empty_memories=FLAGS.random_noise
                        ),
                        sqa_batch
                    ))

                    sentences_ints, sentences_timeword_ints, question_ints, answer_ints = zip(*sqa_batch_standardized)

                    feed_dict = {
                        model.linear_start_indicator: FLAGS.linear_start,
                        model.learning_rate: 0.0,
                        model.sentences_ints_batch: sentences_ints,
                        model.sentences_timewords_ints_batch: sentences_timeword_ints,
                        model.question_ints_batch: question_ints,
                        model.answer_ints_batch: answer_ints,
                    }

                    loss, acc = sess.run(
                        [model.summed_cross_entropy_batch, model.acc_batch],
                        feed_dict=feed_dict
                    )

                    mean_cross_entropy = loss / float(FLAGS.batch_size)

                    sum_cross_entropy += loss
                    nr_correct += int(acc * FLAGS.batch_size)

                    print("validation set, iter {}, batch mean_cross_entropy {}, batch accuracy {}".format(
                        i, mean_cross_entropy, acc
                    ))

                mean_cross_entropy = sum_cross_entropy / float(nr_validation_examples - (nr_validation_examples % FLAGS.batch_size))
                accuracy = nr_correct / float(nr_validation_examples - (nr_validation_examples % FLAGS.batch_size))
                error_rate = 1.0 - accuracy

                print("mean cross_entropy on validation set: {}, \naccuracy: {}, \nerror_rate: {}".format(
                    mean_cross_entropy, accuracy, error_rate
                ))

            if FLAGS.mode == 'test':

                sum_cross_entropy = 0.0
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
                                intersperse_empty_memories=FLAGS.random_noise
                            ),
                            sqa_batch
                        ))

                        sentences_ints, sentences_timeword_ints, question_ints, answer_ints = zip(*sqa_batch_standardized)

                        feed_dict = {
                            model.linear_start_indicator: FLAGS.linear_start,
                            model.learning_rate: 0.0,
                            model.sentences_ints_batch: sentences_ints,
                            model.sentences_timewords_ints_batch: sentences_timeword_ints,
                            model.question_ints_batch: question_ints,
                            model.answer_ints_batch: answer_ints,
                        }

                        loss, acc = sess.run(
                            [model.summed_cross_entropy_batch, model.acc_batch],
                            feed_dict=feed_dict
                        )

                        mean_cross_entropy = loss / float(FLAGS.batch_size)

                        sum_cross_entropy += loss
                        nr_correct += int(acc * FLAGS.batch_size)

                        print("test set, iter {}, batch mean_cross_entropy {}, batch accuracy {}".format(
                            i, mean_cross_entropy, acc
                        ))

                mean_cross_entropy = sum_cross_entropy / float(nr_test_examples - (nr_test_examples % FLAGS.batch_size))
                accuracy = nr_correct / float(nr_test_examples - (nr_test_examples % FLAGS.batch_size))
                error_rate = 1.0 - accuracy

                print("mean cross_entropy on test set: {}, \naccuracy: {}, \nerror_rate: {}".format(
                    mean_cross_entropy, accuracy, error_rate
                ))

            if FLAGS.mode == 'viz':

                for epoch in range(0, 1):
                    for i in range(0, FLAGS.batch_size, FLAGS.batch_size):

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
                                intersperse_empty_memories=FLAGS.random_noise
                            ),
                            sqa_batch
                        ))

                        sentences_ints, sentences_timeword_ints, question_ints, answer_ints = zip(*sqa_batch_standardized)

                        feed_dict = {
                            model.linear_start_indicator: FLAGS.linear_start,
                            model.learning_rate: 0.0,
                            model.sentences_ints_batch: sentences_ints,
                            model.sentences_timewords_ints_batch: sentences_timeword_ints,
                            model.question_ints_batch: question_ints,
                            model.answer_ints_batch: answer_ints,
                        }

                        show_temporal = True
                        temporal_matrices = sess.run(
                                [
                                    model.embedding_matrices['temporal'][
                                        model.routing_formulas['temporal'][FLAGS.weight_tying_scheme]['T_A'](0)
                                    ],
                                    model.embedding_matrices['temporal'][
                                        model.routing_formulas['temporal'][FLAGS.weight_tying_scheme]['T_A'](1)
                                    ],
                                    model.embedding_matrices['temporal'][
                                        model.routing_formulas['temporal'][FLAGS.weight_tying_scheme]['T_A'](2)
                                    ],
                                    model.embedding_matrices['temporal'][
                                        model.routing_formulas['temporal'][FLAGS.weight_tying_scheme]['T_C'](2)
                                    ]
                                ],
                                feed_dict=feed_dict
                        )

                        for temporal_matrix in temporal_matrices:

                            column_labels = [str(i) for i in range(model.d)]
                            row_labels = [str(i) for i in range(model.M)]

                            fig, ax = plt.subplots()
                            heatmap = ax.pcolor(temporal_matrix, cmap=plt.cm.get_cmap('Blues'))

                            # put the major ticks at the middle of each cell
                            ax.set_xticks(np.arange(temporal_matrix.shape[0]) + 0.5, minor=False)
                            ax.set_yticks(np.arange(temporal_matrix.shape[1]) + 0.5, minor=False)

                            # want a more natural, table-like display
                            ax.invert_yaxis()
                            ax.xaxis.tick_top()

                            ax.set_xticklabels(row_labels, minor=False)
                            ax.set_yticklabels(column_labels, minor=False)
                            plt.show()
                        return


main()
