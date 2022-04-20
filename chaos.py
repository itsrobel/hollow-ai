import json
import string
import random
import nltk
import eventlet
import socketio

import numpy as np
import tensorflow as tf

from nltk.stem import WordNetLemmatizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('omw-1.4')
data_file = open('chaosN.json').read()
data = json.loads(data_file)
sio = socketio.Server(cors_allowed_origins='*')

app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})

lemmatizer = WordNetLemmatizer()  # Each list to create
words = []
classes = []
doc_X = []
doc_y = []  # Loop through all the intents
# tokenize each pattern and append tokens to words, the patterns and
# the associated tag to their associated list
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["tag"])

    # add the tag to the classes if it's not there already
    if intent["tag"] not in classes:
        # lemmatize all the words in the vocab and convert them to lowercase
        classes.append(intent["tag"])
# if the words don't appear in punctuation
# sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
words = [lemmatizer.lemmatize(word.lower())
         for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))

training = []
out_empty = [0] * len(classes)  # creating the bag of words model
for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        # mark the index of class that the current pattern is associated
        bow.append(1) if word in text else bow.append(0)
    # to
    output_row = list(out_empty)
    # add the one hot encoded BoW and associated classes to training
    output_row[classes.index(doc_y[idx])] = 1
    # shuffle the data and convert it to an array
    training.append([bow, output_row])
random.shuffle(training)
# split the features and target labels
training = np.array(training, dtype=object)
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 400  # the deep learning model
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation="softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=["accuracy"])
print(model.summary())
model.fit(x=train_X, y=train_y, epochs=epochs, verbose=1)


def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
        return np.array(bow)


def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    # print(f'vocab {vocab}')
    # print(f'labels {labels}')
    # print(f'bow {bow}')
    result = model.predict(np.array([bow]))[0]
    # print(f'result {result}')
    # print(result.sort(reverse=True))
    # print(labels[np.where(result == result.max())]) 
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


@sio.event
def connect(sid, environ):
    print('connect', sid)

@sio.event
def message(sid, res):
    print(f'($): {res["msg"]}')
    intents = pred_class(res["msg"], words, classes)
    print(intents)
    response = get_response(intents, data)
    sio.emit('message', {'user': 'Leo' , 'msg': response}, room=sid)


@sio.event
def disconnect(sid):
    # print('disconnected from server')
    print(f'bye user {sid}')


eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
