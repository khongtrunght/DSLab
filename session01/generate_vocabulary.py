import os
from typing import DefaultDict
import numpy as np



def generate_vocabulary(data_path):
    def compute_idf(df, corpus_size):
        assert df > 0
        return np.log10(corpus_size * 1. / df)
    
    with open(data_path) as f:
        lines = f.read().splitlines()
    doc_count = DefaultDict(int)
    corpus_size = len(lines)

    for line in lines:
        features = line.split('<fff>')
        text = features[-1]
        words = list(set(text.split()))
        for word in words:
            doc_count[word] += 1
    words_idf = [(word, compute_idf(document_freq, corpus_size))
            for word, document_freq in zip(doc_count.keys(), doc_count.values())
            if document_freq > 10 and not word.isdigit()
    ]
    words_idf.sort(key = lambda  word: -word[1])
    print('Vocabulary size: {}'.format(len(words_idf)))
    global dirname
    dirname = os.path.dirname(__file__)
    words_idfs_path = os.path.join(dirname, '../datasets/20news-bydate/words_idfs.txt')
    with open(words_idfs_path, 'w') as f:
        f.write('\n'.join([word + '<fff>' + str(idf) for word, idf in words_idf]))









if __name__ == '__main__':
    # gather_20newsgroups_data()
    dirname = os.path.dirname(__file__)
    generate_vocabulary(os.path.join(dirname,'../datasets/20news-bydate/20news-train-processed.txt',))
