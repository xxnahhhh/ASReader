from model import ASReader
from config import *
import tensorflow as tf
import pickle
import numpy as np
import time
from collections import defaultdict

def GetBatch(data, batch_size, data_size):
    idx = np.arange(0, data_size)
    permutation = np.random.permutation(idx)
    pickIdx = permutation[:batch_size]
    d, q, a = [], [], []
    for idx in pickIdx:
        d.append(data[idx][0])
        q.append(data[idx][1])
        a.append(data[idx][2])
    return np.array(d), np.array(q), np.array(a)

def Accuracy(si, d, a):
    coRRect = 0
    for sampleId in range(d.shape[0]):
        prob = defaultdict(float)  # defaultdict类的初始化函数接受一个类型作为参数，当所访问的键不存在的时候，可以实例化一个值作为默认值, 默认值:0.0
        for idx, wordId in enumerate(d[sampleId]):
            prob[wordId] += si[sampleId][idx]
        predict = max(prob, key=prob.get)
        if predict == a[sampleId]:
            coRRect += 1
    accuracy = coRRect / d.shape[0]
    return accuracy


def main():
    vocab_size, vocab, word2idx, idx2word, dl, ql, cl = pickle.load(open(vocab_path, 'rb'))
    train_data = pickle.load(open(train_idx_path, 'rb'))
    valid_data = pickle.load(open(valid_idx_path, 'rb'))
    model = ASReader(embedding_dim=384, hidden_dim=384, dLen=dl, qLen=ql, learning_rate=learning_rate)
    model.build()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.latest_checkpoint('./checkpoint')
        if ckpt != None:
            saver.restore(sess, ckpt)
        else:
            print('No pre-trained Model!')

        for epoch in range(MAX_EPOCH):
            t = time.time()
            for step in range(TRAIN_SIZE // BATCH_SIZE):
                Bdocument, Bquery, Banswer = GetBatch(train_data, BATCH_SIZE, TRAIN_SIZE)
                tloss, _, tsi = sess.run([model.loss, model.opt, model.si], feed_dict={
                     model.placeholders['d']: Bdocument, model.placeholders['q']: Bquery,
                     model.placeholders['a']: Banswer})
                accuT = Accuracy(tsi, Bdocument, Banswer)

                if step % 10 == 0:
                    Bdv, Bqv, Bav = GetBatch(valid_data, batch_size=128, data_size=VALID_SIZE)
                    vsi, valid_loss = sess.run([model.si, model.loss], feed_dict={model.placeholders['d']: Bdv, model.placeholders['q']: Bqv,
                         model.placeholders['a']: Bav})

                    accuV = Accuracy(vsi, Bdv, Bav)
                    print("EPOCH{}- step{}: Training loss: {}, Training accuracy: {}, Valid loss: {}, Valid accuracy: {} cost {}".format(epoch,step, tloss, accuT, valid_loss, accuV, time.time()-t))
                    accu = [0]
                    if accuV > accu[-1]:
                        saver.save(sess, './checkpoint/asreader.ckpt', global_step= epoch+1)
                        accu.pop()
                        accu.append(accuV)

def readPara():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./ckpt/asreader.ckpt-1.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./ckpt'))

        graph = tf.get_default_graph()
        e = np.array(sess.run(graph.get_tensor_by_name('ASReader/e/Variable:0')))
        pickle.dump(e, open('./pretrained_embeddings.pkl', 'wb'))
        print(e, e.shape)

"""[[ 0.10221114 -0.09302849  0.05798885 ...  0.02761985 -0.01969989
   0.03968412]
 [ 0.06374814  0.02662068  0.07478125 ... -0.02934434 -0.02946329
   0.07817588]
 [ 0.01854263 -0.10979425 -0.00092206 ... -0.05068436  0.02310773
   0.06515026]
 ...
 [-0.02676753 -0.09359378 -0.08950463 ... -0.08989225 -0.00453988
   0.12297522]
 [ 0.08937415 -0.13032386 -0.0491985  ... -0.01787065 -0.09334439
  -0.08978292]
 [ 0.062757   -0.12380081 -0.01832102 ... -0.08106015 -0.06441952
   0.08802645]]"""
if __name__ == '__main__':
    main()
    # readPara()
