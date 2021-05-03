import os
from gather_data import gather_20newsgroups_data
from generate_vocabulary import generate_vocabulary
from compute_tf_idf import get_tf_idf

dirname = os.path.dirname(__file__)
gather_20newsgroups_data()
generate_vocabulary(os.path.join(dirname,'../datasets/20news-bydate/20news-full-processed.txt',))
get_tf_idf(os.path.join(dirname, '../datasets/20news-bydate/20news-full-processed.txt'))