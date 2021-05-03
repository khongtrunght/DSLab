from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
import os
from os import listdir
from os.path import isfile
import re
import platform

def gather_20newsgroups_data():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../datasets/20news-bydate/')
    # path = '../datasets/20news-bydate/'
    dirs = [path + sub_dir_name + '/' 
            for sub_dir_name in listdir(path)
            if not isfile(path + sub_dir_name) ]
    train_dir, test_dir = (dirs[0], dirs[1]) if 'train' in dirs[0] \
        else (dirs[1], dirs[0])
    # get list of the newsgorups in train_dir
    list_newsgroups = [newsgroup
                        for newsgroup in  listdir(train_dir)]
    list_newsgroups.sort()

    stemmer = PorterStemmer()

    def collect_data_from(parent_dir, newsgroup_list):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            label = group_id
            dir_path = parent_dir + '/' + newsgroup + '/'
            files = [(filename, dir_path + filename)
                    for filename in listdir(dir_path)
                    if isfile(dir_path + filename)]
            files.sort()
            if platform.system() == 'Windows':
                for filename, filepath in files:
                    with open(filepath,'r') as f:
                        text = f.read().lower()
                        # remove stop words
                        # \W+ stands for non-word characters 
                        words = [stemmer.stem(word) for word in re.split('\W+',text)
                                    if word not in stop_words]
                        # combine remaining words in to string
                        content = ' '.join(words)
                        assert len(content.splitlines()) == 1
                        data.append(str(label) +'<fff>' +
                                filename + '<fff>' + content)
            else:
                for filename, filepath in files:
                    with open(filepath,'r', errors = 'ignore') as f:
                        text = f.read().lower()
                        # remove stop words
                        # \W+ stands for non-word characters 
                        words = [stemmer.stem(word) for word in re.split('\W+',text)
                                    if word not in stop_words]
                        # combine remaining words in to string
                        content = ' '.join(words)
                        assert len(content.splitlines()) == 1
                        data.append(str(label) +'<fff>' +
                                filename + '<fff>' + content)

        return data

    stop_words = get_stop_words('en')
   
    train_data = collect_data_from(
        parent_dir= train_dir, 
        newsgroup_list= list_newsgroups
    )
    test_data = collect_data_from(
        parent_dir= test_dir,
        newsgroup_list= list_newsgroups
    )

    full_data = train_data + test_data
    with open(os.path.join(dirname,'../datasets/20news-bydate/20news-train-processed.txt'), 'w') as f:
        f.write('\n'.join(train_data))
    
    with open(os.path.join(dirname,'../datasets/20news-bydate/20news-test-processed.txt'), 'w') as f:
        f.write('\n'.join(test_data))
        
    with open(os.path.join(dirname,'../datasets/20news-bydate/20news-full-processed.txt'), 'w') as f:
        f.write('\n'.join(full_data))

if __name__ == '__main__':
    gather_20newsgroups_data() 