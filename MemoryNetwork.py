import tensorflow as tf



class MemoryNetwork:
    def __init__(self, vocab_size, embedding_dim, number_of_hops,
                 batch_size, number_of_memories, max_sentence_len):

        self.V = vocab_size
        self.d = embedding_dim
        self.K = number_of_hops

        self.batch_size = batch_size
        self.M = number_of_memories
        self.J = max_sentence_len

    def build_from_weight_scheme(self, weight_tying_scheme, input_sentences_ints_batch, u_batch):
        answer_logits = None

        if weight_tying_scheme == 'adj':
            answer_logits = self.build_from_adj_weight_scheme(input_sentences_ints_batch, u_batch)

        if weight_tying_scheme == 'rnnlike':
            answer_logits = self.build_from_rnnlike_weight_scheme(input_sentences_ints_batch, u_batch)

        return answer_logits


    def build_from_adj_weight_scheme(self, input_sentences_ints_batch, u_batch):
        H = tf.eye(self.d)

        A, C, l, T_A, T_C, _, u_next = self.build_memory_layer(input_sentences_ints_batch, u_batch, H=H)

        for k in range(1, self.K):
            A, C, _, T_A, T_C, _, u_next = self.build_memory_layer(input_sentences_ints_batch, u_next,
                                                   A=C, C=None,
                                                   l=l,
                                                   T_A=T_C, T_C=None,
                                                   H=H
                                                   )

        W = self.build_W_mapping("W")

        answer_logits = tf.matmul(u_next, W) # [batch_size, d] x [d, V] = [batch_size, V]
        return answer_logits

    def build_from_rnnlike_weight_scheme(self, input_sentences_ints_batch, u_batch):

        A, C, l, T_A, T_C, H, u_next = self.build_memory_layer(input_sentences_ints_batch, u_batch)

        for k in range(1, self.K):
            _, _, _, _, _, _, u_next = self.build_memory_layer(input_sentences_ints_batch, u_next,
                                                   A=A, C=C,
                                                   l=l,
                                                   T_A=T_A, T_C=T_C,
                                                   H=H
                                                   )

        W = self.build_W_mapping("W")

        answer_logits = tf.matmul(u_next, W) # [batch_size, d] x [d, V] = [batch_size, V]
        return answer_logits

    def build_memory_layer(self, input_sentences_ints_batch, u_batch,
                           A=None, C=None,
                           l=None,
                           T_A=None, T_C=None,
                           H=None):
        A = tf.identity(A) if (A is not None) else self.build_memory_embedding("A_" + str(0))
        C = tf.identity(C) if (C is not None) else self.build_memory_embedding("C_" + str(0))

        A_word_embeddings = tf.nn.embedding_lookup(A, input_sentences_ints_batch)  # [batch_size, M, J, d]
        C_word_embeddings = tf.nn.embedding_lookup(C, input_sentences_ints_batch)   # [batch_size, M, J, d]

        l = tf.identity(l) if (l is not None) else self.build_position_encoding("l_encoding")  # [1, 1, J, d]

        A_word_embeddings *= l
        C_word_embeddings *= l

        T_A = tf.identity(T_A) if (T_A is not None) else self.build_temporal_embedding("T_A_" + str(0))
        T_C = tf.identity(T_C) if (T_C is not None) else self.build_temporal_embedding("T_C_" + str(0))

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

        H = tf.identity(H) if (H is not None) else self.build_H_mapping("H_" + str(0))

        u_next = tf.matmul(H, o) + u_batch

        return A, C, l, T_A, T_C, H, u_next

    def build_word_embedding(self, variable_scope_name, reuse=False):
        with tf.variable_scope(variable_scope_name, reuse=reuse):
            embedding_matrix = tf.get_variable('embedding_matrix', dtype='float', shape=[self.V, self.d])
            return embedding_matrix

    def build_temporal_embedding(self, variable_scope_name, reuse=False):
        with tf.variable_scope(variable_scope_name, reuse=reuse):
            embedding_matrix = tf.get_variable('embedding_matrix', dtype='float', shape=[self.M, self.d])
            return embedding_matrix

    def build_H_mapping(self, variable_scope_name, reuse=False):
        with tf.variable_scope(variable_scope_name, reuse=reuse):
            embedding_matrix = tf.get_variable('embedding_matrix', dtype='float', shape=[self.d, self.d])
            return embedding_matrix

    def build_W_mapping(self, variable_scope_name, reuse=False):
        with tf.variable_scope(variable_scope_name, reuse=reuse):
            embedding_matrix = tf.get_variable('embedding_matrix', dtype='float', shape=[self.d, self.V])
            return embedding_matrix

    def build_position_encoding(self):
        def l_kj(k, j):
            return (1.0 - (float(j) / float(self.J))) - (float(k) / float(self.d)) * (1.0 - (2.0 * float(j) / float(self.J)))

        def l_j(j):
            return [l_kj(k, j) for k in range(1, self.d + 1)]

        l = [l_j(j) for j in range(1, self.J + 1)]

        l_tensor = tf.constant(l, shape=[self.J, self.d], name='l')

        l_withMdim = tf.expand_dims(l_tensor, 0, name='l_withMdim')
        l_withMdim_withbatchdim = tf.expand_dims(l_withMdim, 0, name='l_withMdim_withbatchdim')

        return l_withMdim_withbatchdim










