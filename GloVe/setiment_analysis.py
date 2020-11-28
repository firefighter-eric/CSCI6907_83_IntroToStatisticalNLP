from preprocess import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report


def word2vec(data, ngram_range):
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=10000, binary=True)
    data = [' '.join(line) for line in data]
    data_vec = vectorizer.fit_transform(data)
    return data_vec.toarray()


def naive_bayes(data, label):
    clf = MultinomialNB()
    pred = cross_val_predict(clf, data, label, cv=3)
    cla_repo = classification_report(label, pred)
    return cla_repo


def logistic_regression(data, label):
    clf = LogisticRegression(penalty='l2', max_iter=500)
    pred = cross_val_predict(clf, data, label, cv=3)
    cla_repo = classification_report(label, pred)
    return cla_repo


if __name__ == '__main__':
    train_data, train_label = read_data('train.csv')
    train_data_unigram_and_bigram = word2vec(train_data, ngram_range=(1, 2))
    train_data_binary = word2vec(train_data, ngram_range=(1, 1))

    cla_repo_nb = naive_bayes(train_data_unigram_and_bigram, train_label)
    print('Naive Bayes Classifier with unigrams and bigrams:\n' + cla_repo_nb)
    cla_repo_lr_default = logistic_regression(train_data_binary, train_label)
    print('Linear Regression Classifier using binary bag-of-words features:\n' + cla_repo_lr_default)
    cla_repo_lr_unigram_and_bigram = logistic_regression(train_data_unigram_and_bigram, train_label)
    print('Naive Bayes Classifier with unigrams and bigrams:\n' + cla_repo_lr_unigram_and_bigram)
