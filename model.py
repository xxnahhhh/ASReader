import tensorflow as tf
from config import *

class ASReader(object):
    def __init__(self,embedding_dim, hidden_dim, dLen, qLen, learning_rate, **kwargs):
        allowed_args = {'vocab_size', 'activations'}
        for k in kwargs:
            assert k in allowed_args, 'Invalid keyword arguments: ' + k

        activations = kwargs.get('activations')
        vocab_size = kwargs.get('vocab_size')
        if not vocab_size or activations:
            vocab_size = VOCAB_SIZE
            activations = tf.nn.relu

        self.lr = learning_rate
        self.activations = activations
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dLen = dLen
        self.qLen = qLen

    def BuildP(self):
        d = tf.placeholder(dtype=tf.int32, shape=[None, self.dLen], name='document')
        q = tf.placeholder(dtype=tf.int32, shape=[None, self.qLen], name='query')
        a = tf.placeholder(dtype=tf.int32, shape=[None], name='answer')

        placeholders = {'d': d, 'q': q, 'a': a}
        return placeholders

    def postsi(self):
        with tf.variable_scope('ASReader'):
            with tf.variable_scope('e'):
                embeddings = tf.Variable(tf.random_uniform(shape=[self.vocab_size, self.embedding_dim],minval=-0.1,maxval=0.1,dtype=tf.float32))

            with tf.variable_scope('f'):
                # 1st document encoder bidirectionalGRU
                d = self.placeholders['d']
                embedD = tf.nn.embedding_lookup(params=embeddings, ids=d)   # embed: [batch_size, dLen, embedding_dim]
                seqdLen = tf.reduce_sum(tf.cast(tf.not_equal(d, 0), tf.int32), 1)  # [batch_size]
                cell = tf.nn.rnn_cell.GRUCell(num_units=self.hidden_dim)
                hi, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,cell_bw=cell,inputs=embedD,sequence_length=seqdLen, dtype=tf.float32)

                fid = tf.concat(hi, 2)
            with tf.variable_scope('g'):
                # 2nd query encoder bidirectionalGRU
                q = self.placeholders['q']
                embedQ = tf.nn.embedding_lookup(embeddings, ids=q)
                seqqLen = tf.reduce_sum(tf.cast(tf.not_equal(q, 0), tf.int32), 1)
                cell = tf.nn.rnn_cell.GRUCell(num_units=self.hidden_dim)
                _, q = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell,inputs=embedQ,sequence_length=seqqLen, dtype=tf.float32)

                gq = tf.concat(q, 1)

            with tf.variable_scope('AttenSum'):
                fqDotPro = tf.matmul(fid, tf.expand_dims(gq, -1))
                si = tf.reshape(tf.nn.softmax(fqDotPro, axis=1), shape=[-1, self.dLen])

        return si

    def _loss(self):
        corA = self.placeholders['a']
        d = self.placeholders['d']
        loss = tf.reduce_mean(-tf.log(tf.reduce_sum(tf.cast(tf.equal(tf.expand_dims(corA, -1), d), tf.float32) * self.si, 1)))
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        return loss

    def _accuracy(self):
        accu = 0
        # self.accuracy = accu
        raise NotImplementedError
        # return

    def _predict(self):
        idx = 0
        raise NotImplementedError
        # return idx

    def build(self):
        self.placeholders = self.BuildP()
        self.si = self.postsi()
        self.loss = self._loss()
        # self.predictions = self.predict()