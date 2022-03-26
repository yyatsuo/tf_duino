import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

def main():
    print("START")
    epoch = 1000

    input_data = np.array(
        (
            [0,0,0],
            [0,0,1],
            [0,1,0],
            [0,1,1],
            [1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1],
        ),
        dtype=np.float32,
    )

    label_data= np.array([0,1,1,2,1,2,2,3], dtype=np.int32)
    train_data, train_label = input_data, label_data

    validation_data, validation_label = input_data, label_data

    model = keras.Sequential(
        [
            keras.layers.Dense(6, activation='relu'),
            keras.layers.Dense(6, activation='relu'),
            keras.layers.Dense(4, activation='softmax'),
        ]
    )

    #model = keras.models.load_model(os.path.join('result','outmodel'))
    model.compile(
        optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']
    )

    tb_cb = keras.callbacks.TensorBoard(log_dir='log', histogram_freq=1)

    model.fit(
        train_data,
        train_label,
        epochs=epoch,
        batch_size=8,
        callbacks=[tb_cb],
        validation_data=(validation_data, validation_label),
    )
    model.save(os.path.join('result','outmodel'))

    print("DONE")

if __name__ == '__main__':
    main()