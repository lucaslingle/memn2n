import tensorflow as tf


class MemoryNetwork:
    def __init__(self, vocab_size, embedding_dim, number_of_hops,
                 batch_size, number_of_memories, max_sentence_len,
                 weight_tying_scheme='adj',
                 position_encoding=True):

        self.V = vocab_size
        self.d = embedding_dim
        self.number_of_hops = int(number_of_hops)

        self.batch_size = batch_size
        self.M = number_of_memories
        self.J = max_sentence_len

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
                    'W': -1
                },
                'rnnlike': {
                    'B': 0,
                    'A': lambda i: 1,
                    'C': lambda i: 2,
                    'W': -1
                },
                'allsame': {
                    'B': 0,
                    'A': lambda i: 0,
                    'C': lambda i: 0,
                    'W': 0
                },
                'alldiff': {
                    'B': 0,
                    'A': lambda i: 2*i + 1,
                    'C': lambda i: 2*i + 2,
                    'W': -1
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
            'word':     [self.build_word_embedding(idx) for idx in range(self.nr_embedding_matrices['word'])],
            'temporal': [self.build_temporal_embedding(idx) for idx in range(self.nr_embedding_matrices['temporal'])]
        }

        self.layer_transition_operator = {
            'adj': tf.constant(tf.eye(self.d)),
            'rnnlike': self.build_H_mapping(),
            'allsame': tf.constant(tf.eye(self.d)),
            'alldiff': self.build_H_mapping()
        }

        self.H = self.layer_transition_operator[weight_tying_scheme]
        self.l = self.build_position_encoding() if position_encoding else self.build_bag_of_words_encoding()

        B_retrieval_idx = self.routing_formulas['word'][weight_tying_scheme]['B']
        self.B = self.embedding_matrices['word'][B_retrieval_idx]

        W_retrieval_idx = self.routing_formulas['word'][weight_tying_scheme]['W']
        self.W = self.embedding_matrices['word'][W_retrieval_idx].T

        sentences_ints_batch, question_ints_batch, answer_ints_batch = self.build_inputs()
        u_batch = self.get_encoded_questions(question_ints_batch)

        memory_output_batch = self.build_and_stack_memory_layers(
            weight_tying_scheme,
            sentences_ints_batch,
            u_batch)

        answer_logits_batch = tf.matmul(memory_output_batch, self.W)     # [batch_size, d] x [d, V] = [batch_size, V]
        answer_probabilities_batch = tf.nn.softmax(answer_logits_batch)  # [batch_size, V]


    def build_inputs(self):
        sentences = tf.placeholder(tf.int32, [self.batch_size, self.M, self.J])
        question = tf.placeholder(tf.int32, [self.batch_size, self.J])
        answer = tf.placeholder(tf.int32, [self.batch_size])
        return sentences, question, answer

    def get_encoded_questions(self, q_batch):
        B_word_embeddings = tf.nn.embedding_lookup(self.B, q_batch)  # [batch_size, J, d]

        l_3dim = tf.expand_dims(self.l, 0)                           # [1, J, d]
        B_word_embeddings = B_word_embeddings * l_3dim               # [batch_size, J, d] x [1, J, d] = [batch_size, J, d]

        u_batch = tf.reduce_sum(B_word_embeddings, 1)                # [batch_size, d]

        return u_batch

    def build_and_stack_memory_layers(self, weight_tying_scheme, input_sentences_ints_batch, u_batch):
        layer_results = [u_batch]

        for i in range(0, self.number_of_hops):

            # get embedding matrices for memory layer i

            A_retrieval_idx = self.routing_formulas['word'][weight_tying_scheme]['A'](i)
            A = self.embedding_matrices['word'][A_retrieval_idx]

            C_retrieval_idx = self.routing_formulas['word'][weight_tying_scheme]['C'](i)
            C = self.embedding_matrices['word'][C_retrieval_idx]

            T_A_retrieval_idx = self.routing_formulas['temporal'][weight_tying_scheme]['T_A'](i)
            T_A = self.embedding_matrices['temporal'][T_A_retrieval_idx]

            T_C_retrieval_idx = self.routing_formulas['temporal'][weight_tying_scheme]['T_C'](i)
            T_C = self.embedding_matrices['temporal'][T_C_retrieval_idx]

            # now we call a function that will:
            #  - compute memory layer i's contents, using the input sentences and the embedding matrices.
            #  - run query against memory layer i to obtain that layer's response

            u_next = self.build_memory_layer(
                input_sentences_ints_batch=input_sentences_ints_batch,
                u_batch=layer_results[-1],
                A=A,
                C=C,
                T_A=T_A,
                T_C=T_C
            )

            layer_results.append(u_next)

        return layer_results[-1]

    def build_memory_layer(self, input_sentences_ints_batch, u_batch,
                           A, C, T_A, T_C):

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
        p = tf.nn.softmax(memory_scores)                   # [batch_size, M]

        p = tf.reshape(p, [self.batch_size, 1, self.M])    # [batch_size, 1, M]

        o = tf.matmul(p, m_C)                              # [batch_size, 1, M] x [batch_size, M, d] = [batch_size, 1, d]
        o = tf.squeeze(o, [1])                             # [batch_size, d]

        u_next = tf.matmul(o, self.H) + u_batch            # [batch_size, d]

        return u_next

    def build_word_embedding(self, idx):
        embedding_matrix = tf.get_variable('word_embedding_matrix_' + str(idx), dtype='float', shape=[self.V, self.d])
        return embedding_matrix

    def build_temporal_embedding(self, idx):
        embedding_matrix = tf.get_variable('temporal_embedding_matrix_' + str(idx), dtype='float', shape=[self.M, self.d])
        return embedding_matrix

    def build_H_mapping(self):
        H_matrix = tf.get_variable('H_matrix', dtype='float', shape=[self.d, self.d])
        return H_matrix

    def build_bag_of_words_encoding(self):
        return tf.constant(tf.ones([self.J, self.d]))

    def build_position_encoding(self):
        def l_kj(k, j):
            return (1.0 - (float(j) / float(self.J))) - (float(k) / float(self.d)) * (1.0 - (2.0 * float(j) / float(self.J)))

        def l_j(j):
            return [l_kj(k, j) for k in range(1, self.d + 1)]

        l_list = [l_j(j) for j in range(1, self.J + 1)]

        l = tf.constant(l_list, shape=[self.J, self.d], name='l')

        return l

