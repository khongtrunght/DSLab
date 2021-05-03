import os
import numpy as np

def get_tf_idf(data_path, path_out):
    # get pre_compute data from file
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, '../datasets/20news-bydate/words_idfs.txt'), 'r') as f:
        words_idfs = [(line.split('<fff>')[0], float(line.split('<fff>')[1]))
                      for line in f.read().splitlines()]
        word_IDs = dict([(word, index) for
                         index, (word, idf) in enumerate(words_idfs)])
        idfs = dict(words_idfs)

    with open(data_path, 'r') as f:
        documents = [
            (int(line.split('<fff>')[0]),
             int(line.split('<fff>')[1]),
             line.split('<fff>')[2])
            for line in f.read().splitlines()
        ]
        data_tf_idf = []
        for document in documents:
            label, doc_id, text = document
            words = [word for word in text.split() if word in idfs]
            word_set = list(set(words))
            max_term_freq = max([words.count(word) for word in word_set])
            words_tfidfs = []
            sum_squares = 0.0
            for word in word_set:
                term_freq = words.count(word)
                tf_idf_value = term_freq * 1. / max_term_freq * idfs[word]
                words_tfidfs.append((word_IDs[word], tf_idf_value))
                sum_squares += tf_idf_value ** 2

            words_tfidfs_normalized = [str(index) + ':'
                                           + str(tf_idf_value / np.sqrt(sum_squares))
                                           for index, tf_idf_value in words_tfidfs]
            sparse_rep = ' '.join(words_tfidfs_normalized)
            data_tf_idf.append((label, doc_id, sparse_rep))
        
        dirname = os.path.dirname(__file__)
        
        with open(path_out, 'w') as f:
            f.write('\n'.join([str(label) + '<fff>' + str(doc_id) + '<fff>' + sparse_rep for label, doc_id, sparse_rep in data_tf_idf]))


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../datasets/20news-bydate/20news-full-processed.txt')
    get_tf_idf(path, path_out = os.path.join(dirname, '../datasets/20news-bydate/20news-full-processed_tf_idf.txt'))
    get_tf_idf('../datasets/20news-bydate/20news-train-processed.txt', 
                '../datasets/20news-bydate/20news-train-processed_tf_idf.txt')
    get_tf_idf('../datasets/20news-bydate/20news-test-processed.txt',
                '../datasets/20news-bydate/20news-test-processed_tf_idf.txt')

