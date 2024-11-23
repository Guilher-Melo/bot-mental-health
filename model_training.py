import random
import pickle
import json
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

intents = json.load(open('intents.json'))

words = []
classes = []
documents = []
ignore_symbols = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_symbols]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('model/words.pkl', 'wb'))
pickle.dump(classes, open('model/classes.pkl', 'wb'))

# Inicializar listas vazias para os dados de treinamento
X = []  # para armazenar os bags of words
y = []  # para armazenar as classes correspondentes

# Criar o array vazio para output
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Criar o bag of words
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # Copiar o output_empty para não modificar a referência original
    output_row = output_empty.copy()
    output_row[classes.index(document[1])] = 1
    
    # Adicionar às listas de treinamento
    X.append(bag)
    y.append(output_row)

# Converter para arrays numpy antes de embaralhar
X = np.array(X)
y = np.array(y)

# Criar índices e embaralhar
indices = np.arange(len(X))
np.random.shuffle(indices)

# Reorganizar X e y usando os índices embaralhados
X = X[indices]
y = y[indices]

# Criar e treinar o modelo
model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))

# Atualizar para a nova API do Keras
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Treinar o modelo
hist = model.fit(X, y, epochs=200, batch_size=5, verbose=1)

# Salvar o modelo
model.save('model/chatbot_model.keras')