import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile


batch_size = 50
lstmUnits = 64
numClasses = 2
maxSeqLength = 10 #Maximum length of sentence
numDimensions = 300 #Dimensions for each word vector


def extract_data(filename):
    """Extract data from tarball and store as list of strings"""
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'reviews/')):
        with tarfile.open(filename, "r") as tarball:
            dir = os.path.dirname(__file__)
            tarball.extractall(os.path.join(dir, 'reviews/'))
    return

def read_data():
    print("READING DATA")
    data = []
    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir, 'reviews/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir, 'reviews/neg/*')))
    print("Parsing %s files" % len(file_list))
    return file_list

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    extract_data('reviews.tar.gz')
    data = read_data()
    return data

def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """

    data = open("glove.6B.50d.txt",'r', encoding="utf-8")
    lines = data.readlines()
    #if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    i = 0
    lineCount = len(lines)
    print(lineCount)
    embeddings = np.zeros(shape=(lineCount,50))
    word_index_dict = {}

    for line in lines:
        lineSplit = line.split(' ', 1)
        word = lineSplit[0]
        line = lineSplit[1]

        npArray = np.fromstring(line, dtype=float, sep=' ')

        embeddings[i] = npArray

        word_index_dict[word] = i
        i = i + 1

    #test print
    print(embeddings[1])
    print(word_index_dict[","])
    return embeddings, word_index_dict


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, dropout_keep_prob, optimizer, accuracy and loss
    tensors"""
    lstm_units = 64

    labels = tf.placeholder(name="labels", dtype=tf.float32, shape=[batch_size,2])
    input_data = tf.placeholder(name="input_data", dtype=tf.int32, shape=[batch_size,40])
    dropout_keep_prob = tf.placeholder_with_default(0.75, name="dropout_keep_prob", shape=())

    data = tf.nn.embedding_lookup(glove_embeddings_arr, input_data)

    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_units)
    lstm = tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob=dropout_keep_prob)
    value, _ = tf.nn.dynamic_rnn(lstm, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstm_units, 2]))
    bias = tf.Variable(tf.constant(0.1, shape=[2]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, dtype=tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    print(loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    accuracy = tf.identity(accuracy, name="accuracy")
    loss = tf.identity(loss, name="loss")

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
