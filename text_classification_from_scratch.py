import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np
from keras import layers

#load the data 

batch_size = 32
raw_train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=1337,
)

raw_val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=1337,
)

raw_test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)

print(f"Number of batches in raw_training_ds:{raw_train_ds.cardinality()}")
print(f"Number of batches in raw_val_ds:{raw_val_ds.cardinality()}")
print(f"Number of batches in raw_test_ds:{raw_test_ds.cardinality()}")

for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(5):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])


#prepare the dataa
import string
import re

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]",""
    )

max_features = 20000
embedding_dim = 128
sequence_length = 500

vectorize_layer = keras.layers.TextVectorization(
    standardize = custom_standardization,
    max_tokens = max_features,
    output_mode = "int",
    output_sequence_length = sequence_length,
)


text_ds = raw_train_ds.map(lambda x, y:x)

vectorize_layer.adapt(text_ds)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

train_ds = train_ds.cache().prefetch(buffer_size = 10)
val_ds = val_ds.cache().prefetch(buffer_size = 10)
test_ds = test_ds.cache().prefetch(buffer_size=10)

inputs = keras.Input(shape=(None,), dtype="int64")

x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = keras.Model(inputs, predictions)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


#train the model
epochs = 3

model.fit(train_ds, validation_data=val_ds,epochs=epochs)


model.summary()

def save_model(model, filename="text_classification.keras"):
    model.save(filename)
    print(f"Model saved to {filename}")

save_model(model,"text_classification.keras")


#evaluate the model on the test set

model.evaluate(test_ds)




"""
#make end to end model


inputs = keras.Input(shape=(1,), dtype = "shape")

indicies = vectorize_layer(inputs)

outputs = model(indicies)

end_to_end_model = keras.Model(inputs, outputs)
end_to_end_model.compile(
    loss = "binary_crossentropy", optimizer
)
"""