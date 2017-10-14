import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile


batch_size = 50


def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""

    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'reviews/')):
        with tarfile.open(filename, "r") as tarball:
            dir = os.path.dirname(__file__)
            tarball.extractall(os.path.join(dir, 'reviews/'))

    data = []
    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir, 'reviews/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir, 'reviews/neg/*')))

    for f in file_list:
        with open(f, "r") as openf:
            s = openf.read()
            no_punct = ''.join(c for c in s if c not in string.punctuation)
            data.extend(no_punct.split())

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
    data = open("glove.6B.50d.txt",'r',encoding="utf-8")
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
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
