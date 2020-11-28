import pandas as pd


def read_data(filename):
    df = pd.read_csv(filename)
    text = []
    label = []
    for line in df.tweet:
        text.append(preprocess(line))
    if 'label' in df:
        label = df.label.tolist()
    return text, label


def preprocess(line):
    line = line.lower().split()
    out = []
    for w in line:
        if '#' in w or '@' in w or 'http' in w:
            continue

        w_out = []
        for c in w:
            if c.isalpha() and c.isascii():
                w_out.append(c)
            w = "".join(w_out)
        if w:
            out.append(w)

    return out


if __name__ == '__main__':
    T, L = read_data('train.csv')
