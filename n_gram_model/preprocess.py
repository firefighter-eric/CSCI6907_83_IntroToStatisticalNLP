import re


def read_file(filename: str):
    f = open(filename, encoding='utf8')
    data = []
    for line in f:
        line = re.sub(r"\'", r"", line)
        data.append(line.strip())
    f.close()
    return data


def sentence_segmentation(text: str):
    sentence = []
    line = re.split(r"\.|,|:|;|\"|\?|<|>|\(|\)|\d", text)
    for clause in line:
        clause = clause.strip()
        if clause:
            sentence.append(clause.split())
    return sentence


def raw2sentence_map(data_sets: list):
    sentences = []
    for line in data_sets:
        sentence = sentence_segmentation(line)
        for s in sentence:
            sentences.append(s)
    return sentences


def word_segmentation(text: str):
    words = []
    tmp = re.split(r"\.|,|:|;|\"|!|\?|<|>|\(|\)| |\d", text)
    for word in tmp:
        if word:
            words.append(word.lower())
    return words


def gen_word_set(raw: list):
    word_set = set()
    for line in raw:
        words = word_segmentation(line)
        for word in words:
            word_set.add(word)
    return list(word_set)


def test_data():
    def set_index(data):
        for i in range(len(data)):
            for j in range(len(data[i])):
                if data[i][j] == '.':
                    data[i] = data[i][j + 1:].strip()
                    break
        return data

    x = read_file('LangID.test.txt')
    y = read_file('LangID.gold.txt')[1:]
    set_index(x)
    set_index(y)
    return x, y


if __name__ == '__main__':
    D = read_file('EN.txt')
    S = raw2sentence_map(D)
    W = gen_word_set(D)
    # print(W)
    X, Y = test_data()
    # print(X)
