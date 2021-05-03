import os
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from collections import defaultdict
import numpy as np



def load_data(data_path):
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0 for _ in range(vocab_size)]
        sparse_r_d_list = sparse_r_d.split()
        for index_tf_idf in sparse_r_d_list:
            index, tf_idf = int(index_tf_idf.split(':')[0]), float(
                index_tf_idf.split(':')[1])
            r_d[index] = tf_idf
        return np.array(r_d)
    with open(data_path) as f:
        d_lines = f.read().splitlines()
    with open(os.path.join(this_path, '../datasets/20news-bydate/words_idfs.txt')) as f:
        vocab_size = len(f.read().splitlines())
    _data = []
    labels = []
    _label_count = defaultdict(int)
    for data_id, data in enumerate(d_lines):
        features = data.split('<fff>')
        label, doc_id = int(features[0]), int(features[1])
        _label_count[label] += 1
        r_d = sparse_to_dense(
            sparse_r_d=features[2], vocab_size=vocab_size)
        _data.append(r_d)
        labels.append(label)
        
    return np.array(_data) , np.array(labels)

def clustering_with_KMeans():
    global this_path
    this_path = os.path.dirname(__file__)
    data_path = os.path.join(this_path, '../datasets/20news-bydate/data_tf_idf.txt')
    data, labels = load_data(data_path = data_path)
    # print(labels)
    X = csr_matrix(data)
    kmeans = KMeans(
        n_clusters = 20,
        init='k-means++',
        n_init=5,
        tol=1e-4,
        random_state=2018
    ).fit(X)
    labels2 = kmeans.labels_
    print(labels2 != labels)
    print(np.sum(labels2 == labels)/ len(labels))
    print(kmeans.cluster_centers_ )
    
def compute_accuracy(predicted_y, expected_y):
    matches = np.equal(predicted_y, expected_y)
    accuracy = np.sum(matches.astype(float))/expected_y.size
    return accuracy

def classifying_with_linear_SVMs():
    train_X, train_Y = load_data('../datasets/20news-bydate/20news-train-processed_tf_idf.txt')
    classifier = LinearSVC(
        C = 10.0,
        tol = 0.001,
        verbose = True
    )
    classifier.fit(train_X, train_Y)
    test_X, test_Y = load_data('../datasets/20news-bydate/20news-test-processed_tf_idf.txt')
    predicted_Y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y = predicted_Y, expected_y = test_Y)
    print("Accuracy:", accuracy)

def classifying_with_kernel_SVMs():
    train_X, train_Y = load_data('../datasets/20news-bydate/20news-train-processed_tf_idf.txt')
    classifier = SVC(
        C=50.0,
        kernel = 'rbf',
        gamma = 0.1,
        tol=0.001,
        verbose=True
    )
    classifier.fit(train_X, train_Y)
    
    test_X, test_Y = load_data('../datasets/20news-bydate/20news-test-processed_tf_idf.txt')
    predicted_Y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y=predicted_Y, expected_y=test_Y)
    print("Accuracy:", accuracy)

if __name__ == '__main__':
    this_path = os.path.dirname(__file__)
    # clustering_with_KMeans()
    # classifying_with_kernel_SVMs()
    classifying_with_linear_SVMs()