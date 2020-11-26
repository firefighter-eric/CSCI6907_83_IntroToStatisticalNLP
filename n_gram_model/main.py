from preprocess import *
from model import *


class NGramProcess:
    def __init__(self, train_seg, test_seg, model_type):
        self.LANG = ['EN', 'FR', 'GR']
        self.train_data = self.train_data_preprocess(train_seg)
        self.test_x, self.test_y = self.test_data_preprocess(test_seg)

        self.models = list(map(model_type, self.train_data))
        self.prob, self.perplexity = self.predict(self.models, self.test_x)
        self.show()

    @staticmethod
    def train_data_preprocess(train_seg):
        LANG = ['EN', 'FR', 'GR']
        filename = [_ + '.txt' for _ in LANG]
        raw_data = map(read_file, filename)
        word_sets = map(train_seg, raw_data)
        return word_sets

    @staticmethod
    def test_data_preprocess(test_seg):
        test_x_raw, test_y = test_data()
        test_x = []
        for line in test_x_raw:
            test_x.append(test_seg(line))
        return test_x, test_y

    def predict(self, models, test_x):
        pp = []
        for x in test_x:
            pp.append([])
            for m in models:
                pp[-1].append(m.predict_union(x))

        prob = self.log2prob(pp)
        return pp, prob

    def show(self):
        def get_accuracy(x, y):
            count = 0
            for i in range(len(x)):
                if x[i] == y[i]:
                    count += 1
                # else:
                    # print(i)
            return count / len(x)

        result = []
        for p in self.perplexity:
            result.append(self.LANG[p.index(min(p))])

        # print(perplexity)
        # print(prob)
        # print(result)
        print('Accuracy:', get_accuracy(result, self.test_y))

    @staticmethod
    def log2prob(log_prob: list):
        N = len(log_prob[0])
        prob = []
        for line in log_prob:
            prob.append(line[:])
            mean = sum(prob[-1]) / N
            for i in range(N):
                prob[-1][i] -= mean
            for i in range(N):
                prob[-1][i] = 2 ** prob[-1][i]
            s = sum(prob[-1])
            for i in range(N):
                prob[-1][i] /= s
        return prob


if __name__ == '__main__':
    print('letter bigram')
    letter_bigram = NGramProcess(gen_word_set, word_segmentation, BiGramModel)
    print('word bigram with add one smoothing')
    word_bigram_with_add_one_smoothing = NGramProcess(raw2sentence_map, sentence_segmentation, BiGramModelWithAddOne)
    print('word bigram with good turing smoothing')
    word_bigram_with_good_turing_smoothing = NGramProcess(raw2sentence_map, sentence_segmentation,
                                                          BiGramModelWithGoodTuring)
    print('word trigram with good turing smoothing')
    word_trigram_with_kn_smoothing = NGramProcess(raw2sentence_map, sentence_segmentation, TriGramModelWithKNSmoothing)
