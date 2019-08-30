import numpy as np
import pandas as pd
import nltk
from nltk.stem.lancaster import LancasterStemmer
import tflearn
import tensorflow as tf
import random
import string
import unicodedata
import sys
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')

df=pd.read_csv('bbc-text.csv')

# a table structure to hold the different punctuation used
tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))
# method to remove punctuations from sentences.
def remove_punctuation(text):
    return text.translate(tbl)

# initialize the stemmer
stemmer = LancasterStemmer()

#Splitting 70:30
df_train=df[0:-667]
df_test=df[-667:]

#convert data into dictionary
data=df_train.groupby('category')['text'].apply(list).to_dict()

# get a list of all categories to train for
categories = list(data.keys())
words = []
# a list of tuples with words in the sentence and category name
docs = []

for each_category in data.keys():
    print(each_category)
    for each_sentence in data[each_category]:
        # remove any punctuation from the sentence
        each_sentence = remove_punctuation(each_sentence)
        # extract words from each sentence and append to the word list
        w = nltk.word_tokenize(each_sentence)
        words.extend(w)
        docs.append((w, each_category))
        
# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(categories)


for doc in docs:
    # initialize our bag of words(bow) for each document in the list
    bow = []
    # list of tokenized words for the pattern
    token_words = doc[0]
    # stem each word
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    # create our bag of words array
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)

    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1

    # our training set will contain a the bag of words model and the output row that tells
    # which catefory that bow belongs to.
    training.append([bow, output_row])

# shuffle our features and turn into np.array as tensorflow takes in numpy array
random.shuffle(training)
training = np.array(training)
# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

print("Training started...")
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=50, batch_size=8, validation_set=0.1, show_metric=True)
model.save('model.tflearn')

print("Testing started...")
def get_tf_record(sentence):
    global words
    sentence = remove_punctuation(sentence)
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1
    return(np.array(bow))

df_test['predict']=np.nan
for i in range(df_test.shape[0]):
    df_test.iloc[i,2]=categories[np.argmax(model.predict([get_tf_record(df_test.iloc[i,1])]))]
    print(i)
    
num_correct=df_test[df_test.category==df_test.predict].shape[0]
num_incorrect=df_test[df_test.category!=df_test.predict].shape[0]
total=num_correct+num_incorrect
print(total)
print("Testing Accuracy=",num_correct/total)
