import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import requests

import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify

# Настройка логгирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Создание обработчика для записи логов в файл
file_handler = RotatingFileHandler('app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(file_handler)

# Пример датасета с вопросами и ответами
f = open("questions.txt", "r", encoding='utf-8')
f1 = open("answers.txt", "r", encoding='utf-8')

questions = []
answers = []

for line in f:
    questions.append(line.strip())

for line in f1:
    answers.append(line.strip())


answer_labels = {answer: i for i, answer in enumerate(answers)}
answer_labels_list = [answer_labels[answer] for answer in answers]

# Преобразование списка в массив Numpy
answer_labels_array = np.array(answer_labels_list)


# Токенизация текста
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
word_index = tokenizer.word_index

# Создание обучающих данных
sequences = tokenizer.texts_to_sequences(questions)
padded_sequences = pad_sequences(sequences)

# Создание модели нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index) + 1, 64),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(answers), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
model.fit(padded_sequences, answer_labels_array, epochs=100)

# Функция для предсказания ответа на вопрос
def generate_answer(question):
    sequence = tokenizer.texts_to_sequences([question])
    padded_sequence = pad_sequences(sequence, maxlen=len(padded_sequences[0]))  # Выравнивание до длины обучающих данных
    prediction = model.predict(padded_sequence)
    predicted_answer_index = np.argmax(prediction, axis=1)[0]
    return answers[predicted_answer_index]

app = Flask(__name__)


@app.route('/process_message', methods=['POST'])
def generate_answer_from_laravel():
    data = request.get_json()
    message = data['message']

    # Логгирование получения сообщения
    logger.info(f'Received message: {message}')

    message = generate_answer(message)
    # Логгирование отправки ответа
    logger.info(f'Sent answer: {message}')
    return jsonify({'message': message})


app.run()

#Пример использования
# question = str(input())
# answer = generate_answer(question)
# print("Question:", question)
# print("Answer:", answer)
