from collections import defaultdict
from math import log2


class Prob:
    def __init__(self):
        self.data = defaultdict(dict)

    def set(self, k1, k2, v):
        self.data[k1][k2] = v

    def add_1(self, k1, k2):
        self.data[k1][k2] = self.data[k1].get(k2, 0) + 1

    def get(self, a, b):
        """
        :param a: word[i-1]
        :param b: word[i]
        """
        if a not in self.data or b not in self.data[a]:
            return None
        else:
            return self.data[a][b]

    def get_v(self):
        """
        :return: number of vocabulary
        """
        return len(self.data)

    def items(self):
        return self.data.items()


class UniGramModel:
    def __init__(self, data):
        self.data = data
        self.P1, self.P_UNK = self.train(data)

    def train(self, data):
        counter = self.count(data)
        p, unk = self.normalization(counter)
        p, unk = self.log_prob(p), log2(unk)
        return p, unk

    @staticmethod
    def count(data):
        counter = defaultdict(int)
        for x in data:
            for n in x:
                counter[n] += 1
        return counter

    @staticmethod
    def normalization(counter):
        s = sum(counter.values())
        for k in counter.keys():
            counter[k] /= s
        return counter, 1 / s

    @staticmethod
    def log_prob(p):
        for k in p.keys():
            p[k] = log2(p[k])
        return p

    def get_p(self, x0):
        if x0 in self.P1:
            return self.P1[x0]
        else:
            return self.P_UNK

    def predict(self, data):
        p = 0
        for x in data:
            p += self.get_p(x)
        return p


class BiGramModel(UniGramModel):
    def __init__(self, data):
        self.sub_model = UniGramModel(data)
        self.P1, self.P_UNK = self.sub_model.P1, self.sub_model.P_UNK
        self.P2 = self.train(data)

    def train(self, data: list):
        counter = self.count(data)
        P2 = self.mle_normalization(counter)
        P2 = self.log_prob(P2)
        return P2

    def count(self, data: list):
        # count gram in datasets
        counter = Prob()
        for x in data:
            counter.add_1(' ', x[0])
            counter.add_1(x[-1], ' ')
            for i in range(1, len(x)):
                counter.add_1(x[i - 1], x[i])
        return counter

    @staticmethod
    def mle_normalization(counter):
        for k1, v1 in counter.items():
            S = sum(v1.values())
            for k2 in v1.keys():
                v1[k2] /= S
        return counter

    @staticmethod
    def log_prob(p):
        # log all prob
        for v in p.data.values():
            for k in v:
                v[k] = log2(v[k])
        return p

    def get_p(self, *x):
        a, b = x
        p = self.P2.get(a, b)
        if p is None:
            p = self.P_UNK
        return p

    def predict(self, x):
        p = self.get_p(' ', x[0]) + self.get_p(x[-1], ' ')
        for i in range(1, len(x)):
            p += self.get_p(x[i - 1], x[i])
        return p

    def predict_union(self, data: list):
        return -sum(map(self.predict, data)) / (len(data) + 2)


class BiGramModelWithAddOne(BiGramModel):
    def train(self, data):
        counter = self.count(data)
        P2 = self.add_one_smooth(counter)
        P2 = self.log_prob(P2)
        return P2

    @staticmethod
    def add_one_smooth(counter):
        """
        '#': P_UNK
        S: count of word[i-1]
        V: number of word set
        W: sum of S
        """
        V = counter.get_v()
        W = 0
        for k1, v1 in counter.data.items():
            S = sum(v1.values())
            W += S
            for k2 in v1.keys():
                v1[k2] = (v1[k2] + 1) / (V + S)
            counter.set(k1, '#', 1 / (V + S))
        return counter

    def get_p(self, *x):
        a, b = x
        p = self.P2.get(a, b)
        if p is None:
            p = self.P2.get(a, '#')
        if p is None:
            p = self.P_UNK
        return p


class BiGramModelWithGoodTuring(BiGramModel):
    def train(self, data):
        counter = self.count(data)
        P2 = self.good_turing_smooth(counter)
        P2 = self.log_prob(P2)
        return P2

    @staticmethod
    def good_turing_smooth(counter):
        """
        S: count of word[i-1]
        c: count of count(word[i|i-1]) == 1
        """
        V = counter.get_v()
        W = 0
        for k1, v1 in counter.items():
            S = sum(v1.values())
            W += S
            c = 0
            for k2 in v1.keys():
                v1[k2] = v1[k2] / S
                if v1[k2] == 1:
                    c += 1
            if c == 0:
                c = 1
            counter.set(k1, '#', c / S)
        return counter

    def get_p(self, *x):
        a, b = x
        p = self.P2.get(a, b)
        if p is None:
            p = self.P2.get(a, '#')
        if p is None:
            p = self.P_UNK
        return p


class BiGramModelWithKNSmoothing(BiGramModel):
    def kn_smoothing(self, counter):
        for k1, v1 in counter.items():
            S = sum(v1.values())
            c = 0
            for k2 in v1.keys():
                if v1[k2] > 0:
                    c += 1
            lamb = 0.75 * c / S
            for k2 in v1.keys():
                v1[k2] = (v1[k2] - 0.75) / S + lamb * c * self.P_UNK
        return counter

    def get_p(self, *x):
        a, b = x
        p = self.P2.get(a, b)
        if p is None:
            p = self.sub_model.get_p(b)
        return p


class TriGramModelWithKNSmoothing(BiGramModelWithKNSmoothing):
    def __init__(self, data):
        self.data = data
        self.sub_model = BiGramModelWithKNSmoothing(data)
        self.P2, self.P1, self.P_UNK = self.sub_model.P2, self.sub_model.P1, self.sub_model.P_UNK
        self.P3 = self.train(data)

    def count(self, data: list):
        # count gram in datasets
        counter = Prob()
        for x in data:
            if len(x) == 1:
                counter.add_1(' ' + x[0], ' ')
                continue
            counter.add_1(' ' + x[0], x[1])
            counter.add_1(x[-2] + x[-1], ' ')
            for i in range(2, len(x)):
                counter.add_1(x[i - 2] + x[i - 1], x[i])
        return counter

    def get_p(self, *x):
        a, b, c = x
        p = self.P3.get(a + b, c)
        if p is None:
            p = self.sub_model.get_p(b, c)
        return p

    def predict(self, x):
        if len(x) == 1:
            return self.get_p(' ', x[0], ' ')
        p = self.get_p(' ', x[0], x[1]) + self.get_p(x[-2], x[-1], ' ')
        for i in range(2, len(x)):
            p += self.get_p(x[i - 2], x[i - 1], x[i])
        return p
