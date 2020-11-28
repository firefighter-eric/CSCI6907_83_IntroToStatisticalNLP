from tool_function import *
from collections import defaultdict


class MajorityClassModel:
    def __init__(self, train_data):
        self.D, self.model = self.fit(train_data)

    @staticmethod
    def fit(train_data):
        # count the tag of word
        count = defaultdict(dict)
        for line in train_data:
            for word, tag in line:
                count[word][tag] = count[word].get(tag, 0) + 1

        # get the most frequent tag
        word_tag_max = {}
        for word in count.keys():
            tag_max = max(count[word].items(), key=lambda x: x[1])[0]
            word_tag_max[word] = tag_max
        return count, word_tag_max

    def predict_word(self, word):
        return self.model.get(word, 'nn')

    def predict(self, data):
        p = []
        for line in data:
            for word, _ in line:
                p.append(self.predict_word(word))
        return p


class MajorityClassModelWithRules(MajorityClassModel):
    def __init__(self, train_data):
        super().__init__(train_data)
        self.rules = self.get_rules(15, train_data)

    def get_rules(self, n, train_data):
        # get n words with large error
        p = self.predict(train_data)
        _, error = get_accuracy(p, train_data)
        error = sorted(error.items(), key=lambda x: x[1], reverse=True)[:n]

        # make n rules
        rules = {}
        # words = ['to', 'that', 'it', 'as', 'more', 'her', 'so', 'you', 'there', 'about', 'before', 'one', 'on']
        words = [word for word, _ in error]
        for word in words:
            rules[word] = self.get_rule(word, train_data)
        return rules

    @staticmethod
    def get_rule(target_word, train_data):
        count = defaultdict(dict)
        for line in train_data:
            for i, (word, tag) in enumerate(line):
                if i < len(line) - 1 and word == target_word:
                    next_word = line[i + 1][1]
                    count[next_word][tag] = count[next_word].get(tag, 0) + 1

        rule = {}
        for next_word_tag, tag in count.items():
            rule[next_word_tag] = max(count[next_word_tag].items(), key=lambda x: x[1])[0]
        return rule

    def predict_word_with_rule(self, line, i):
        word = line[i][0]
        if i < len(line) - 1 and word in self.rules:
            next_word = line[i + 1][0]
            next_word_tag = self.predict_word(next_word)
            if next_word_tag in self.rules[word]:
                return self.rules[word][next_word_tag]
        return self.model.get(word, 'nn')

    def predict_with_rule(self, data):
        p = []
        for line in data:
            for i in range(len(line)):
                p.append(self.predict_word_with_rule(line, i))
        return p


if __name__ == '__main__':
    TrainData = read_data('brown.train.tagged.txt')
    TestData = read_data('brown.test.tagged.txt')

    m1 = MajorityClassModel(TrainData)
    p_v_1 = m1.predict(TrainData)
    p_t_1 = m1.predict(TestData)
    acc_v_1, e_v_1 = get_accuracy(p_v_1, TrainData)
    acc_t_1, e_t_1 = get_accuracy(p_t_1, TestData)
    print('------Majority Class Model------')
    print('validation accuracy:', acc_v_1)
    print('test accuracy', acc_t_1)

    m2 = MajorityClassModelWithRules(TrainData)
    p_v_2 = m2.predict_with_rule(TrainData)
    p_t_2 = m2.predict_with_rule(TestData)
    acc_v_2, e_v_2 = get_accuracy(p_v_2, TrainData)
    acc_t_2, e_t_2 = get_accuracy(p_t_2, TestData)
    print('------Majority Class Model With Rules------')
    print('validation accuracy:', acc_v_2)
    print('test accuracy', acc_t_2)
