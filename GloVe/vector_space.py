from preprocess import *
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from matplotlib.pyplot import bar, show
from gensim.models import Word2Vec


def common_word(data):
    model = Word2Vec(sentences=data, size=1000)
    like = model.wv.most_similar(positive=['like'])
    hate = model.wv.most_similar(positive=['hate'])
    return like, hate


def most_10_common_words(data, label):
    h = []
    nh = []
    for i in range(len(data)):
        if label[i] == 0:
            nh += data[i]
        else:
            h += data[i]

    non_hatred_dict = nltk.FreqDist(nh)
    hatred_dict = nltk.FreqDist(h)

    nh_mc = non_hatred_dict.most_common(10)
    h_mc = hatred_dict.most_common(10)

    nh_x, nh_y = [x for x, _ in nh_mc], [y for _, y in nh_mc]
    h_x, h_y = [x for x, _ in h_mc], [y for _, y in h_mc]

    bar(nh_x, nh_y)
    show()
    bar(h_x, h_y)
    show()

    return non_hatred_dict.most_common(10), hatred_dict.most_common(10)


def get_analogy(data):
    model = Word2Vec(sentences=data)
    out = model.wv.most_similar(positive=['love', 'life'], negative=['hate'])
    return out


if __name__ == '__main__':
    train_data, train_label = read_data('train.csv')
    Like, Hate = common_word(train_data)
    print("Like:", Like, "\nHate:", Hate)
    print()
    # non_hatred, hatred = most_10_common_words(train_data, train_label)
    # print('most 10 common words', '\nnon_hatred:', non_hatred, '\nhatred:', hatred)
    # print()
    # analogy_word = get_analogy(train_data)
    # print("analogy word:\n", analogy_word)
