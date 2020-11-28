from collections import defaultdict


def read_data(filename):
    f = open(filename)
    data = []
    for line in f:
        line = line.strip().lower().split()
        data.append([])
        for raw in line:
            for i in range(len(raw) - 1, -1, -1):
                if raw[i] == '/':
                    data[-1].append((raw[:i], raw[i + 1:]))
                    break
    return data


def get_accuracy(p, data):
    error = defaultdict(int)
    count = 0

    i = 0
    for line in data:
        for word, tag in line:
            if p[i] == tag:
                count += 1
            else:
                error[word] += 1
            i += 1
    accuracy = count / len(p)
    return accuracy, error
