import tensorflow as tf
import numpy as np


class MLP:
    def __init__(self, vocab_size, hidden_size):
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size

    def build_graph(self):
        self._X = tf.compat.v1.placeholder(tf.float32, shape=[None, self._vocab_size])
        self._real_Y = tf.compat.v1.placeholder(tf.int32, shape=[None,])
        
        weights_1 = tf.get_variable(
            name='weights_input_hidden',
            shape=(self._vocab_size, self._hidden_size),
            initializer=tf.random_normal_initializer(seed=2021)
        )
        biases_1 = tf.get_variable(
            name='biases_input_hidden',
            shape=(self._hidden_size),
            initializer=tf.random_normal_initializer(seed=2021)
        )
        
        weights_2 = tf.compat.v1.get_variable(
            name='weights_hidden_output',
            shape=(self._hidden_size, NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2021)
        )
        biases_2 = tf.compat.v1.get_variable(
            name='biases_hidden_output',
            shape=(NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2021)
        )
        
        hidden = tf.matmul(self._X, weights_1) + biases_1
        hidden = tf.sigmoid(hidden)
        logits = tf.matmul(hidden, weights_2) + biases_2

        labels_one_hot = tf.one_hot(indices = self._real_Y, depth = NUM_CLASSES, dtype = tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels_one_hot, logits = logits)
        loss = tf.reduce_mean(loss)

        probs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probs, axis=1)
        predicted_labels = tf.squeeze(predicted_labels)

        return predicted_labels, loss

    def trainer(self, loss, learning_rate):
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op

    
class DataReader:
    def __init__(self, data_path, batch_size, vocab_size):
        self._batch_size = batch_size
        with open(data_path) as f:
            d_lines = f.rad().splitlines()

        self._data = []
        self._labels = []
        for data_id, line in enumerate(d_lines):
            vector = [0.0 for _ in range(vocab_size)]
            features = line.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            tokens = features[2].split()
            for token in tokens:
                index, value = int(token.split(':')[0]), float(token.split(':')[1])
                vector[index] = value
            self._data.append(vector)
            self._labels.append(label)

        self._data = np.array(self._data)
        self._labels = np.array(self._labels)
        self._num_epoch = 0 
        self._batch_id = 0
    
    def next_batch(self):
        start = self._batch_id * self._batch_size
        end = start + self._batch_size
        self._batch_id += 1

        if end + self._batch_size > len(self._data):
            end = len(self._data)
            self._num_epoch += 1
            self._batch_id = 0
            indices = list(range(len(self._data)))
            random.seed(2021)
            random.shuffle(indices)
            self._data, self._labels = self._data[indices], self._labels[indices]
        return self.data[start:end], self._labels[start:end]

def load_dataset():
    train_data_reader = DataReader(
        data_path='../datasets/20news-bydate/20news-train-processed_tf_idf.txt',
        batch_size=50,
        vocab_size= vocab_size
    )
    test_data_reader = DataReader(
        data_path='../datasets/20news-bydate/20news-test-processed_tf_idf.txt',
        batch_size=50,
        vocab_size= vocab_size
    )
    return train_data_reader, test_data_reader
    


