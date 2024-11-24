import random
import pickle
import json
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import os

from tensorflow.keras.models import load_model


# Define o diretório base (onde está seu código)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define os caminhos dos arquivos
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'chatbot_model.keras')
CLASSES_PATH = os.path.join(BASE_DIR, 'model', 'classes.pkl')
WORDS_PATH = os.path.join(BASE_DIR, 'model', 'words.pkl')
INTENTS_PATH = os.path.join(BASE_DIR, 'model', 'intents.json')

def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    ignore_symbols = ['?', '!', '.', ',']

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words if word not in ignore_symbols]

    return sentence_words


def bag_of_words(sentence):
    words = pickle.load(open(WORDS_PATH, 'rb'))

    sentence_words = clean_up_sentence(sentence)

    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag) 


def predict_class(sentence):
    classes = pickle.load(open('model/classes.pkl', 'rb'))
    model = load_model(MODEL_PATH)

    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list


def get_response(intents_list):
    intents_json = json.loads(open(INTENTS_PATH).read())

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    
    return result