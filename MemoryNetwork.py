import tensorflow as tf
import os
from tensorflow.python.ops import control_flow_ops

class WeightInitializerHelper:
    def __init__(self):

        self.initializers = {
            'word': {
                'random_normal_initializer': lambda scale: tf.random_normal_initializer(
                    mean=0.0, stddev=scale
                ),
                'truncated_normal_initializer': lambda scale: tf.truncated_normal_initializer(
                    mean=0.0, stddev=scale
                ),
                'orthogonal_initializer': lambda scale: tf.orthogonal_initializer(
                    gain=scale
                ),
                'xavier_normal_initializer': lambda scale: tf.contrib.layers.xavier_initializer(
                    uniform=False
                ),
                'xavier_uniform_initializer': lambda scale: tf.contrib.layers.xavier_initializer(
                    uniform=True
                )
            },
            'temporal': {
                'random_normal_initializer': lambda scale: tf.random_normal_initializer(
                    mean=0.0, stddev=scale
                ),
                'truncated_normal_initializer': lambda scale: tf.truncated_normal_initializer(
                    mean=0.0, stddev=scale
                ),
                'orthogonal_initializer': lambda scale: tf.orthogonal_initializer(
                    gain=scale
                ),
                'xavier_normal_initializer': lambda scale: tf.contrib.layers.xavier_initializer(
                    uniform=False
                ),
                'xavier_uniform_initializer': lambda scale: tf.contrib.layers.xavier_initializer(
                    uniform=True
                )
            }
        }

class MemoryNetwork:
    def __init__(self, vocab_size, embedding_dim, number_of_hops,
                 batch_size, number_of_memories, max_sentence_len,
                 gradient_clip=40,
                 gradient_noise_scale=0.001,
                 weight_tying_scheme='adj',
                 position_encoding=True,
                 word_emb_initializer='random_normal_initializer',
                 word_emb_init_scale=0.1,
                 temporal_emb_initializer='random_normal_initializer',
                 temporal_emb_init_scale=0.1
                 ):

        self.V = int(vocab_size)
        self.d = int(embedding_dim)
        self.number_of_hops = int(number_of_hops)

        self.batch_size = int(batch_size)
        self.M = int(number_of_memories)
        self.J = int(max_sentence_len)

        self.gradient_clip = float(gradient_clip)
        self.gradient_noise_scale = float(gradient_noise_scale)

        self.weight_tying_scheme = weight_tying_scheme
        self.position_encoding = position_encoding

        self.weight_init = WeightInitializerHelper()
        self.word_initializer = self.weight_init.initializers['word'][word_emb_initializer](word_emb_init_scale)
        self.nonword_initializer = self.weight_init.initializers['temporal'][temporal_emb_initializer](temporal_emb_init_scale)

        self.nr_embedding_matrices_formulas = {
            'word': {
                'adj': 1 + (self.number_of_hops - 1) + 1,
                'rnnlike': 1 + 2 + 1,
                'allsame': 1,
                'alldiff': 1 + (2 * self.number_of_hops) + 1
            },
            'temporal': {
                'adj': 1 + (self.number_of_hops - 1) + 1,
                'rnnlike': 2,
                'allsame': 1,
                'alldiff': (2 * self.number_of_hops)
            }
        }

        # formulas for retrieving the appropriate word embedding matrix in a list of distinct word embedding matrices
        # the argument i assumes memory layers are indexed from 0 to K-1 (whereas the paper used 1-based indices)
        self.routing_formulas = {
            'word': {
                'adj': {
                    'B': 0,
                    'A': lambda i: i,
                    'C': lambda i: i+1,
                    'W': self.nr_embedding_matrices_formulas['word']['adj'] - 1
                },
                'rnnlike': {
                    'B': 0,
                    'A': lambda i: 1,
                    'C': lambda i: 2,
                    'W': self.nr_embedding_matrices_formulas['word']['rnnlike'] - 1
                },
                'allsame': {
                    'B': 0,
                    'A': lambda i: 0,
                    'C': lambda i: 0,
                    'W': self.nr_embedding_matrices_formulas['word']['allsame'] - 1
                },
                'alldiff': {
                    'B': 0,
                    'A': lambda i: 2*i + 1,
                    'C': lambda i: 2*i + 2,
                    'W': self.nr_embedding_matrices_formulas['word']['alldiff'] - 1
                }
            },
            'temporal': {
                'adj': {
                    'T_A': lambda i: i,
                    'T_C': lambda i: i+1
                },
                'rnnlike': {
                    'T_A': lambda i: 0,
                    'T_C': lambda i: 1
                },
                'allsame': {
                    'T_A': lambda i: 0,
                    'T_C': lambda i: 0
                },
                'alldiff': {
                    'T_A': lambda i: 2*i,
                    'T_C': lambda i: 2*i + 1
                }
            }
        }

        self.nr_embedding_matrices = {
            'word':     self.nr_embedding_matrices_formulas['word'][weight_tying_scheme],
            'temporal': self.nr_embedding_matrices_formulas['temporal'][weight_tying_scheme]
        }

        self.embedding_matrices = {
            'word':     {
                str(idx): self.build_word_embedding_matrix(idx)
                for idx in range(self.nr_embedding_matrices['word'])
            },
            'temporal': {
                str(idx): self.build_temporal_embedding_matrix(idx)
                for idx in range(self.nr_embedding_matrices['temporal'])
            }
        }

        self.sentence_position_encoders = {
            'position_encoding': self.build_position_encoding(),
            'bag_of_words': self.build_bag_of_words_encoding()
        }

        self.attention_mechanisms = {
            'softmax': lambda memory_scores: self.build_softmax_attention(memory_scores),
            'linear': lambda memory_scores: self.build_linear_attention(memory_scores)
        }

        self.layer_transition_operators = {
            'adj': tf.eye(self.d),
            'rnnlike': self.build_H_mapping(scope_name='rnnlike'),
            'allsame': tf.eye(self.d),
            'alldiff': self.build_H_mapping(scope_name='alldiff')
        }

        self._vars_with_nils = set([emb for emb in self.embedding_matrices['word']])

        # build placeholders
        self.sentences_ints_batch, self.question_ints_batch, self.answer_ints_batch = self.build_data_inputs()
        self.linear_start_indicator, self.learning_rate = self.build_control_inputs()

        self.encoding_type = 'position_encoding' if position_encoding else 'bag_of_words'
        self.l = self.sentence_position_encoders[self.encoding_type]

        # encode questions
        self.u_batch = self.get_encoded_questions(weight_tying_scheme, self.question_ints_batch)

        # run query against memory network
        self.layer_results = self.build_and_stack_memory_layers(
            weight_tying_scheme,
            self.sentences_ints_batch,
            self.u_batch)

        self.memory_output_batch = self.layer_results[-1]

        # decode to get answer logits
        self.answer_logits_batch, self.answer_probs_batch = self.get_answer_logits(weight_tying_scheme, self.memory_output_batch)

        # build loss and other metrics
        self.answer_onehot_labels_batch = tf.nn.embedding_lookup(tf.eye(self.V), self.answer_ints_batch)

        self.summed_cross_entropy_batch, self.acc_batch, self.err_batch = self.build_loss_func(
            self.answer_logits_batch, self.answer_onehot_labels_batch
        )

        # build training operation
        self.train_op = self.build_training_op(self.summed_cross_entropy_batch,
                                               learning_rate=self.learning_rate,
                                               gradient_clip=self.gradient_clip,
                                               gradient_noise_scale=self.gradient_noise_scale)

        # build a saver for all this
        self.saver = tf.train.Saver()

    def build_control_inputs(self):
        linear_start_indicator = tf.placeholder(tf.bool)
        learning_rate = tf.placeholder(tf.float32)

        return linear_start_indicator, learning_rate

    def build_data_inputs(self):
        sentences = tf.placeholder(tf.int32, [self.batch_size, self.M, self.J])
        question = tf.placeholder(tf.int32, [self.batch_size, self.J])
        answer = tf.placeholder(tf.int32, [self.batch_size])
        return sentences, question, answer

    def get_encoded_questions(self, weight_tying_scheme, q_batch):

        B_retrieval_idx = self.routing_formulas['word'][weight_tying_scheme]['B']
        B = self.embedding_matrices['word'][str(B_retrieval_idx)]

        B_word_embeddings = tf.nn.embedding_lookup(B, q_batch)  # [batch_size, J, d]

        l_3dim = tf.expand_dims(self.l, 0)                      # [1, J, d]
        B_word_embeddings = B_word_embeddings * l_3dim          # [batch_size, J, d] x [1, J, d] = [batch_size, J, d]

        u_batch = tf.reduce_sum(B_word_embeddings, 1)           # [batch_size, d]

        return u_batch

    def build_and_stack_memory_layers(self, weight_tying_scheme, input_sentences_ints_batch, u_batch):
        layer_results = [u_batch]
        u_next = u_batch

        for i in range(0, self.number_of_hops):

            # get embedding matrices for memory layer i

            A_retrieval_idx = self.routing_formulas['word'][weight_tying_scheme]['A'](i)
            A = self.embedding_matrices['word'][str(A_retrieval_idx)]

            C_retrieval_idx = self.routing_formulas['word'][weight_tying_scheme]['C'](i)
            C = self.embedding_matrices['word'][str(C_retrieval_idx)]

            T_A_retrieval_idx = self.routing_formulas['temporal'][weight_tying_scheme]['T_A'](i)
            T_A = self.embedding_matrices['temporal'][str(T_A_retrieval_idx)]

            T_C_retrieval_idx = self.routing_formulas['temporal'][weight_tying_scheme]['T_C'](i)
            T_C = self.embedding_matrices['temporal'][str(T_C_retrieval_idx)]

            H = self.layer_transition_operators[weight_tying_scheme]

            # now we call a function that will:
            #  - compute memory layer i's contents, using the input sentences and the embedding matrices.
            #  - run query against memory layer i to obtain that layer's response

            u_next = self.build_memory_layer(
                input_sentences_ints_batch=input_sentences_ints_batch,
                u_batch=u_next,
                A=A,
                C=C,
                T_A=T_A,
                T_C=T_C,
                H=H
            )

            layer_results.append(u_next)

        return layer_results

    def build_memory_layer(self, input_sentences_ints_batch, u_batch,
                           A, C, T_A, T_C, H):

        A_word_embeddings = tf.nn.embedding_lookup(A, input_sentences_ints_batch)  # [batch_size, M, J, d]
        C_word_embeddings = tf.nn.embedding_lookup(C, input_sentences_ints_batch)  # [batch_size, M, J, d]

        l_4dim = tf.expand_dims(tf.expand_dims(self.l, 0), 0)                      # [1, 1, J, d]

        A_word_embeddings = A_word_embeddings * l_4dim    # [batch_size, M, J, d] x [1, 1, J, d] = [batch_size, M, J, d]
        C_word_embeddings = C_word_embeddings * l_4dim    # [batch_size, M, J, d] x [1, 1, J, d] = [batch_size, M, J, d]

        m_A_without_temporal = tf.reduce_sum(A_word_embeddings, 2) # [batch_size, M, d]
        m_C_without_temporal = tf.reduce_sum(C_word_embeddings, 2) # [batch_size, M, d]

        m_A = m_A_without_temporal + tf.expand_dims(T_A, 0)
        m_C = m_C_without_temporal + tf.expand_dims(T_C, 0)

        u_aug = tf.expand_dims(u_batch, -1)                # [batch_size, d, 1]
        memory_scores = tf.matmul(m_A, u_aug)              # [batch_size, M, d] x [batch_size, d, 1] = [batch_size, M, 1]
        memory_scores = tf.squeeze(memory_scores, [2])     # [batch_size, M]

        p = self.attention_mechanism(memory_scores)        # [batch_size, M]

        p = tf.expand_dims(p, 1)                           # [batch_size, 1, M]

        o = tf.matmul(p, m_C)                              # [batch_size, 1, M] x [batch_size, M, d] = [batch_size, 1, d]
        o = tf.squeeze(o, [1])                             # [batch_size, d]

        u_next = tf.matmul(o, H) + u_batch                 # [batch_size, d]

        return u_next

    def get_answer_logits(self, weight_tying_scheme, memory_output_batch):
        W_retrieval_idx = self.routing_formulas['word'][weight_tying_scheme]['W']
        W = tf.transpose(self.embedding_matrices['word'][str(W_retrieval_idx)])

        answer_logits_batch = tf.matmul(memory_output_batch, W)          # [batch_size, d] x [d, V] = [batch_size, V]
        answer_probabilities_batch = tf.nn.softmax(answer_logits_batch)  # [batch_size, V]

        return answer_logits_batch, answer_probabilities_batch

    def build_word_embedding_matrix(self, matrix_id):
        # According to 'End-to-End Memory Networks' section 4.2:
        #  "The embedding of the null symbol was constrained to be zero."
        #
        # In this implementation, we assume that the vocab dictionary maps the null word to int value 0.
        #
        pad_embedding = tf.zeros([1, self.d])
        nonpad_embeddings = tf.get_variable('word_embedding_matrix_' + str(matrix_id),
                                            dtype='float',
                                            shape=[self.V - 1, self.d],
                                            initializer=self.word_initializer
                                            )

        word_embedding_matrix = tf.concat([pad_embedding, nonpad_embeddings], axis = 0)
        return word_embedding_matrix

    def build_temporal_embedding_matrix(self, idx):

        temporal_embedding_matrix = tf.get_variable('temporal_embedding_matrix_' + str(idx),
                                                    dtype='float',
                                                    shape=[self.M, self.d],
                                                    initializer=self.nonword_initializer
                                                    )

        return temporal_embedding_matrix

    def build_H_mapping(self, scope_name, reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            H_matrix = tf.get_variable('H_matrix',
                                       dtype='float',
                                       shape=[self.d, self.d],
                                       initializer=self.nonword_initializer
            )
            return H_matrix

    def build_bag_of_words_encoding(self):
        return tf.ones([self.J, self.d])

    def build_position_encoding(self):
        """ EDIT: Facebook's matlab implementation uses a formula written differently than the formula in the paper.
                  See https://github.com/facebook/MemNN/blob/master/MemN2N-babi-matlab/build_model.m

            | % construct model
            |if use_bow == false
            |    config.weight = ones(config.input_dim, config.max_words, 'single');
            |    for i = 1:config.input_dim
            |        for j = 1:config.max_words
            |            config.weight(i,j) = (i-(config.input_dim+1)/2)*(j-(config.max_words+1)/2);
            |        end
            |    end
            |config.weight = 1 + 4 * config.weight / config.input_dim / config.max_words;

            Using wolfram alpha, one may observe that the resulting formula differs from the one in the paper.
            Most notably, there is a "2" missing from in front of the (k/d) portion of the formula in the paper.

            The result is that the paper's formula corresponds to weights that adjust each coordinate k of the embedding.
            Considered as a continuous function, the partial derivative w.r.t. the embedding coordinate k, for k = 1, ..., d
            is
                (j - (J/2)) / (J * d / 2)
            whereas the partial derivative for the position encoding formula used in the official implementation
            is
                (j - (J/2)) / (J * d / 4)

            Thus, the partial derivative from the paper is twice that of the partial derivative
            from the formula printed in the paper. As printed, the position encoding described in the paper is flatter,
            and may not distinguish as well between words in different positions within a sentence.

            Empirical observation shows that using the formula printed in the paper has a deleterious impact on the learning process,
            particularly for task 15 and 16, "basic deduction" and "basic induction".
        """

        def l_kj(k, j):
            # formula from paper
            original_formula = (1.0 - (float(j) / float(self.J))) - (float(k) / float(self.d)) * (1.0 - (2.0 * float(j) / float(self.J)))

            # corrected formula. embedding seems to need the extra variation added by the factor of 2.
            corrected_formula = 2.0 * original_formula

            return corrected_formula


        def l_j(j):
            return [l_kj(k, j) for k in range(1, self.d + 1)]

        l_list = [l_j(j) for j in range(1, self.J + 1)]

        l = tf.constant(l_list, shape=[self.J, self.d], name='l')

        return l

    def build_softmax_attention(self, memory_scores):
        return tf.nn.softmax(memory_scores, dim=1)

    def build_linear_attention(self, memory_scores):
        return tf.identity(memory_scores)

    def attention_mechanism(self, memory_scores):
        # This function exists so that we can change from attention being linear to softmax
        # During so-called 'Linear Start' training

        attn = tf.cond(
            tf.equal(self.linear_start_indicator, tf.constant(True)),
            lambda: self.attention_mechanisms['linear'](memory_scores),
            lambda: self.attention_mechanisms['softmax'](memory_scores)
        )

        return attn

    def build_loss_func(self, answer_logits, answers_one_hot):
        cross_entropy_batch = tf.nn.softmax_cross_entropy_with_logits(logits=answer_logits, labels=answers_one_hot)

        # per section 4.2, "
        # "All training uses a batch size of 32 (but cost is not averaged over a batch),
        #  and gradients with an `2 norm larger than 40 are divided by a scalar to have norm 40. "
        #
        # in order to replicate paper's gradient clipping, we will also *NOT* average the loss over a batch, but sum it.
        #
        summed_cross_entropy_batch = tf.reduce_sum(cross_entropy_batch, 0)

        predictions_batch = tf.argmax(answer_logits, 1)
        correct_predictions_batch = tf.equal(predictions_batch, tf.argmax(answers_one_hot, 1))
        accuracy_batch = tf.reduce_mean(tf.cast(correct_predictions_batch, dtype=tf.float32), 0)
        error_rate_batch = tf.constant(1.0) - accuracy_batch

        return summed_cross_entropy_batch, accuracy_batch, error_rate_batch

    def add_gradient_noise(self, t, stddev=1e-3, name=None):
        """
        Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
        The input Tensor `t` should be a gradient.
        The output will be `t` + gaussian noise.
        0.001 was said to be a good fixed value for memory networks [2].
        """
        with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
            t = tf.convert_to_tensor(t, name="t")
            gn = tf.random_normal(tf.shape(t), stddev=stddev)
            return tf.add(t, gn, name=name)

    def build_training_op(self, loss, learning_rate, gradient_clip, gradient_noise_scale):

        tvars = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(learning_rate)

        print(tvars)

        print([v.name for v in tvars])

        gradients, _ = zip(*opt.compute_gradients(loss, tvars))

        print(gradients)

        gradients = [
            None if gradient is None else tf.clip_by_norm(gradient, gradient_clip)
            for gradient in gradients]

        print(gradients)

        gradients = [
            None if gradient is None else self.add_gradient_noise(gradient, stddev=gradient_noise_scale)
            for gradient in gradients]

        print(gradients)

        grad_updates = opt.apply_gradients(list(zip(gradients, tvars)))
        train_tensor = control_flow_ops.with_dependencies([grad_updates], loss)
        return train_tensor


    def save(self, session, checkpoint_dir, checkpoint_name='memn2n_model'):
        checkpoint_fp = os.path.join(checkpoint_dir, checkpoint_name)
        self.saver.save(session, checkpoint_fp)
        print("[*] Successfully saved model to checkpoint {}".format(checkpoint_fp))

    def load(self, session, checkpoint_dir, checkpoint_name=None):

        # load latest
        if checkpoint_name is None:
            try:
                checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
                self.saver.restore(session, checkpoint)
                print("[*] Successfully loaded model from checkpoint {}".format(checkpoint_dir))
                return
            except Exception():
                raise Exception(" [!] Failed to load model. No checkpoints in {}".format(checkpoint_dir))

        # load a specified model checkpoint
        checkpoint_fp = os.path.join(checkpoint_dir, checkpoint_name)
        self.saver.restore(session, checkpoint_fp)
        print("[*] Successfully loaded model from checkpoint {}".format(checkpoint_fp))

