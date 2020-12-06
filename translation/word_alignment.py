from nltk.translate.ibm1 import IBMModel1
from nltk.translate.api import AlignedSent

# english = 'let ud all strive to live'.split()
# french = 'employons-nous tous à vivre et à laisser vivre'.split()


Foreign = ['das Haus', 'das Buch', 'ein Buch']
English = ['the house', 'the book', 'a book']

text = []
for i in range(3):
    text.append(AlignedSent(Foreign[i].split(), English[i].split()))

ibm1 = IBMModel1(text, 1)

for w1, d2 in ibm1.translation_table.items():
    for w2, p in d2.items():
        print(w1, '\t', w2, '\t', p)
