# What data.py file will do is:
# (1)read data file into list as [document, query, answer, candidates]
# (2)build training data dictionary as vocabulary, mapping word2idx & idx2word
# (3)use vocabulary dict to transform all elements in data list to idx ver.

from config import *
import itertools
import pickle

def read(data_path):
    with open(data_path) as data:
        cases = []
        document = []
        for line in data:
            line = line.strip()
            if not line:
                continue
            idx, line = line.split(' ', 1)
            if 'XXXXX' in line:
                q, a, _, candidates = line.split('\t')
                q = q.split()
                candidates = candidates.split('|')
                # print(line,'_', q,a,candidates)
                cases.append((document, q, a, candidates))
                document = []
            else:
                words = line.split()
                document.extend(words)
    print('Number of samples in {} is {}'.format(data_path, len(cases)))
    return cases

def BuildVoc(train, valid):
    # Big Notice:itertools.chain treat ['the',['the']] as different data, 1st 'the' will be divided into 't','h','e',
    # ['the'] will be not, as it treat words within lists as inidividual word, treat string as character division
    # So, it would be added [[]] to a cuz it'a string-->Wrong, it has not to
    # Because d+q+[a]+c is [sample1,....]->sample:[w1,..,wn(d),...,w(a),...]
    alldata = train + valid
    voc = set(itertools.chain(*(d+q+[a]+c for d,q,a,c in alldata)))
    vocab_size = len(voc) + 1
    word2idx = {w: idx+1 for idx, w in enumerate(voc)}
    word2idx.update({'pad': 0})
    idx2word = {v: k for k,v in word2idx.items()}

    # documentLen, queryLen, candidateLen
    documentLen = max([len(d) for d,_,_,_ in alldata])
    queryLen = max([len(q) for _,q,_,_ in alldata])
    candidateLen = max([len(c) for _,_,_,c in alldata])
    return vocab_size, voc, word2idx, idx2word, documentLen, queryLen, candidateLen

def PAD(seq, length):
    if len(seq) < length:
        seq.extend([0]*(length - len(seq)))
    return seq

def TransForm(word2idx, data, dl, ql):
    idx_data = []
    for d, q, a, c in  data:
        d = [word2idx[w] for w in d]
        q = [word2idx[w] for w in q]
        a = word2idx[a]
        c = [word2idx[w] for w in c]
        d = PAD(d, dl)
        q = PAD(q, ql)
        idx_data.append((d, q, a, c))
    return idx_data

def Write():
    # Store data
    pickle.dump(train_data, open('./train_data_idx.pkl', 'wb'))
    pickle.dump(valid_data, open('./valid_data_idx.pkl', 'wb'))
    pickle.dump((vocab_size, vocab, word2idx, idx2word, dl, ql, cl), open('./vocab.pkl', 'wb'))

    # Wirte all configures into config.py
    with open('./config.py', 'a') as cf:
        cf.write('VOCAB_SIZE = ' + str(vocab_size) + '\n')
        cf.write('TRAIN_SIZE = ' + str(len(train_data)) + '\n')
        cf.write('VALID_SIZE = ' + str(len(valid_data)) + '\n')
    return

if __name__ == '__main__':
    # (1) [document, query, answer, candidates]
    train = read(train_data_path)
    valid = read(valid_data_path)

    # (2) build training data dictionary
    vocab_size, vocab, word2idx, idx2word, dl, ql, cl = BuildVoc(train, valid)
    print(vocab_size) # 61200

    # (3) idx ver. [document, query, answer, candidates]
    train_data = TransForm(word2idx, train, dl, ql)
    valid_data = TransForm(word2idx, valid, dl, ql)

    # Write data to store
    Write()

