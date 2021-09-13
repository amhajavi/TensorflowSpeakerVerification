import numpy as np

from tensorflow.keras import layers as tkl
from tensorflow.keras.optimizers import Adam
from base_models.sv_model import SpeakerVerificationModel

def make_toy_model():
    input = tkl.Input(shape=(4,))
    FCN = tkl.Dense(10)(input)
    FCN = tkl.Dense(1, activation='sigmoid')(FCN)
    model = SpeakerVerificationModel(input, FCN)
    model.compile(optimizer=Adam(learning_rate=1),
                  loss="binary_crossentropy",
                  metrics=["accuracy"],)
    return model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = make_toy_model()
    ones = np.ones(shape=(100, 4))
    zeros = np.zeros(shape=(100, 4))
    input = np.concatenate((ones, zeros))
    output_ones = np.ones(shape=(100, 1))
    output_zeros = np.zeros(shape=(100, 1))
    output = np.concatenate((output_ones, output_zeros))
    history = model.fit(
        input,
        output,
        epochs=100,
        validation_split=0.2,
    )
    print(model.predict(input))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
