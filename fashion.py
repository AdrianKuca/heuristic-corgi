import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

pathToFile = "./1booknoN.txt"
text = open(pathToFile, "rb").read().decode(encoding="utf-8")
print(len(text))
vocab = sorted(set(text))
print("{} unikalne znaki".format(len(vocab)))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
textAsNumbers = np.array([char2idx[c] for c in text])
seqLen = 100
examplesPerEpoch = 500
charDataset = tf.data.Dataset.from_tensor_slices(textAsNumbers)
sequences = charDataset.batch(seqLen + 1, drop_remainder=True)

dataset = sequences.map(lambda chunk: (chunk[:-1], chunk[1:]))

batchSize = 5
stepsPerEpoch = examplesPerEpoch
bufferSize = 10000

vocabSize = len(vocab)
embeddingDim = 256
rnnUnits = 1024
if tf.test.is_gpu_available():
    rnn = tf.keras.layers.CuDNNGRU
    print("GPU I JEDZIEMY C:")
else:
    import functools

    rnn = functools.partial(tf.keras.layers.GRU, recurrent_activation="sigmoid")
    print("CPU I STOIMY :CCC")


def buildModel(rnnUnits, vocabSize, embeddingDim, batchSize):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Embedding(
            vocabSize, embeddingDim, batch_input_shape=[batchSize, None]
        )
    )
    model.add(
        rnn(
            rnnUnits,
            return_sequences=True,
            recurrent_initializer="glorot_uniform",
            stateful=True,
        )
    )
    model.add(
        rnn(
            rnnUnits // 2,
            return_sequences=True,
            recurrent_initializer="glorot_uniform",
            stateful=True,
        )
    )
    model.add(tf.keras.layers.Dense(vocabSize))
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )


EPOCHS = 1
model = buildModel(rnnUnits, vocabSize, embeddingDim, batchSize)
predictionModel = buildModel(rnnUnits, vocabSize, embeddingDim, 1)
model.compile(optimizer=tf.optimizers.Adam(), loss=loss)


checkpointDir = "./czek"
checkpointPrefix = "./czek/fanfik{loss}"
checkpointCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpointPrefix, save_weights_only=True
)
if tf.train.latest_checkpoint(checkpointDir) != None:
    if input(tf.train.latest_checkpoint(checkpointDir) + " load? y/n") == "y":
        model.load_weights(tf.train.latest_checkpoint(checkpointDir))


def generateText(model, startString):
    numGenerate = 2000
    inputEval = [char2idx[s] for s in startString]
    inputEval = tf.expand_dims(inputEval, 0)
    textGenerated = []
    temperature = 1.0
    model.reset_states()
    for i in range(numGenerate):
        predictions = model(inputEval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predictedId = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        inputEval = tf.expand_dims([predictedId], 0)
        textGenerated.append(idx2char[predictedId])
    return startString + "".join(textGenerated)


counter = 1
while True:
    datasetBatch = dataset.shuffle(bufferSize).batch(batchSize, drop_remainder=True)
    history = model.fit(
        datasetBatch.repeat(),
        epochs=EPOCHS,
        steps_per_epoch=stepsPerEpoch,
        callbacks=[checkpointCallback],
    )
    predictionModel.set_weights(model.get_weights())
    with open(
        "./generatedTexts/FANFIK" + str(counter) + ".txt", "w", encoding="utf-8"
    ) as f:
        f.write(generateText(predictionModel, "Otwarł drzwi."))
    counter += 1
