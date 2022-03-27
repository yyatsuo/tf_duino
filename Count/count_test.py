import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

def main():
    epoch = 1000

    with open('test_data.txt', 'r') as f:
        lines = f.readlines()

    data = []
    for l in lines:
        d = l.strip().split()
        data.append(list(map(int, d)))
    test_data = np.array(data, dtype=np.float32)

    model = keras.models.load_model(os.path.join('result','outmodel'))

    predictions = model.predict(test_data)
    print(predictions)

    for i, prediction in enumerate(predictions):
        result = np.argmax(prediction)
        print(f'input: {test_data[i] }, result: {result}')

if __name__ == '__main__':
    main()