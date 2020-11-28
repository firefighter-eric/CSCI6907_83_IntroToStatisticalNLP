from tool_function import *
from time import time
import numpy as np


class Encode:
    def __init__(self, data):
        tag_set = set()
        word_set = set()
        for line in data:
            for word, tag in line:
                word_set.add(word)
                tag_set.add(tag)

        i2word = ['\\OOV'] + list(word_set)
        word2i = {word: i for i, word in enumerate(i2word)}
        i2tag = list(tag_set)
        tag2i = {tag: i for i, tag in enumerate(i2tag)}
        self.i2word, self.word2i, self.i2tag, self.tag2i = i2word, word2i, i2tag, tag2i

    def w2i(self, word):
        return self.word2i.get(word, 0)


class HMMTagger:
    def __init__(self, train_data, decode_function):
        self.add_one = 1e-2
        self.E = Encode(train_data)
        self.A, self.B, self.PI = self.fit(train_data)
        self.decode_function = self.viterbi if decode_function == 'viterbi' else self.beam_search

    def fit(self, train_data):
        A, PI = self.get_transition_prob(train_data, self.E.tag2i)
        B = self.get_emission_prob(train_data, self.E.word2i, self.E.tag2i)
        self.normalization(A, self.add_one)
        self.normalization(PI, self.add_one)
        self.normalization(B, self.add_one)
        return A, B, PI

    @staticmethod
    def normalization(matrix, k):
        matrix += k
        matrix /= matrix.sum(axis=1).reshape(-1, 1)
        np.log2(matrix, out=matrix)

    @staticmethod
    def get_transition_prob(data, tag2i):
        n_tag = len(tag2i)
        PI = np.zeros((1, n_tag))
        A = np.zeros((n_tag, n_tag))
        for line in data:
            PI[0, tag2i[line[0][1]]] += 1
            for i in range(1, len(line)):
                i_tag_1, i_tag_2 = tag2i[line[i - 1][1]], tag2i[line[i][1]]
                A[i_tag_1, i_tag_2] += 1
        return A, PI

    @staticmethod
    def get_emission_prob(data, word2i, tag2i):
        n_tag = len(tag2i)
        n_word = len(word2i)
        B = np.zeros((n_tag, n_word))
        for line in data:
            for word, tag in line:
                i_word, i_tag = word2i[word], tag2i[tag]
                B[i_tag, i_word] += 1
        return B

    def predict(self, data):
        p = []
        for line in data:
            tmp = [self.E.w2i(word) for word, _ in line]
            v = self.decode_function(tmp)
            for n in v:
                p.append(self.E.i2tag[n])
        return p

    def viterbi(self, order):
        A, B, PI = self.A, self.B, self.PI
        n_tag, L = A.shape[0], len(order)
        delta = np.empty((L, n_tag))
        phi = np.zeros((L, n_tag), dtype=np.int)
        delta[0, :] = PI + B[:, order[0]]

        for i in range(1, L):
            tmp = delta[i - 1, :].reshape(-1, 1) + A
            phi[i, :] = np.argmax(tmp, axis=0)
            delta[i, :] = np.max(tmp, axis=0) + B[:, order[i]]

        path = np.empty((L,), dtype=np.int)
        path[-1] = np.argmax(delta[-1, :])
        for i in range(L - 2, -1, -1):
            path[i] = phi[i + 1, path[i + 1]]
        return path

    def beam_search(self, order):
        n_beam = 10
        A, B, PI = self.A, self.B, self.PI
        L = len(order)
        delta = np.empty((L, n_beam))
        phi_in = np.zeros((L, n_beam), dtype=np.int)
        phi_out = np.zeros((L, n_beam), dtype=np.int)

        tmp = PI + B[:, order[0]]
        index = np.unravel_index(np.argsort(tmp, axis=None)[-n_beam:], tmp.shape)
        phi_out[0] = index[1]
        delta[0, :] = tmp[index]

        for i in range(1, L):
            tmp = delta[i - 1, :].reshape(-1, 1) + A[index[1]] + B[:, order[i]].reshape((1, -1))
            index = np.unravel_index(np.argsort(tmp, axis=None)[-n_beam:], tmp.shape)
            phi_in[i] = index[0]
            phi_out[i] = index[1]
            delta[i, :] = tmp[index]

        path = np.empty((L,), dtype=np.int)
        path[-1] = phi_out[-1, -1]
        i_path = phi_in[-1, -1]
        for i in range(L - 2, -1, -1):
            path[i] = phi_out[i, phi_in[i + 1, i_path]]
            i_path = phi_in[i + 1, i_path]
        return path


if __name__ == '__main__':
    TrainData = read_data('brown.train.tagged.txt')
    TestData = read_data('brown.test.tagged.txt')

    m1 = HMMTagger(TrainData, 'viterbi')
    t1 = time()
    p_t = m1.predict(TestData)
    t1 = time() - t1
    # p_v = m.predict(TrainData)
    # acc_v_1, e_v_2 = get_accuracy(p_v, TrainData)
    acc_t_1, e_t_1 = get_accuracy(p_t, TestData)
    print('------HMM Tagger with Viterbi------')
    print('test accuracy:', acc_t_1)
    print('Time:', t1)

    m2 = HMMTagger(TrainData, 'beam search')
    t2 = time()
    p_t = m2.predict(TestData)
    t2 = time() - t2
    acc_t_2, e_t_2 = get_accuracy(p_t, TestData)
    print('------HMM Tagger with Beam Search------')
    print('test accuracy:', acc_t_2)
    print('Time:', t2)

