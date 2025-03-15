import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np
from keras import layers
from tensorflow.keras.models import load_model

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
#15 epocjhs is too much eval dropped from 86 to 85 (3 epochs to 15)
epochs = 10




def save_model(model, filename="text_classification_experiment1.keras"):
    model.save(filename)
    print(f"Model saved to {filename}")


def load_trained_model(model,filename="text_classification_experiment1.keras"):
    model = load_model(filename)
    print(f"Model {filename} loaded")





def predict_text(model, vectorize_layer, text):
    text = tf.expand_dims(text, -1)
    #text = tf.convert_to_tensor([text])  # Convert input text to tensor
    vectorized_text = vectorize_layer(text)  # Apply the same vectorization
    vectorized_text = tf.cast(vectorized_text, tf.int64)
    prediction = model.predict(vectorized_text)  # Get model prediction
    sentiment = "Positive" if prediction[0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]  # Return sentiment and confidence score





train_mode = False
if train_mode is True:
    model.fit(train_ds, validation_data=val_ds,epochs=epochs)
    print("------------------")
    print("model summary")
    model.summary()
    print("------------------")
    save_model(model,"text_classification_experiment1.keras")
    print("------------------")
    model.evaluate(test_ds)
else:
    load_trained_model("text_classification_experiment1.keras")
    print("------------------")
    print("predicting the sample")


    sample_text2_neg = "I'm not going to say that this movie is horrible, because I have seen worse, but it's not even halfway decent. <br /><br />The plot is very confusing. I couldn't really figure out what was happening and where things were going. When the movie was over, I was left scratching my head. I watched through to the end of the credits to see if they had something after them that may clear things up, but once the credits were over, that was it. I felt like I was jarred from one weak plot point to another throughout the whole movie, with little or no transition between the two. <br /><br />Character development is very shallow. I couldn't figure out when somebody was angry or had a grudge against someone. I couldn't tell if half of the characters were just supposed to be drunk, stoned, mentally challenged or they just had a bad actor to portray them. This film seems to be based around stereotypes (to it's credit, they are hard to avoid using when you are making a film about a singer in a rock band), which SHOULD make character development easier, since so many other films have already illustrated the suffering of an abused child, or the trials of a heroin addict trying to come clean. Stereotypes are easy to depict, which would explain why so many bad films tend to overuse stereotypical characters. This film, on the other hand, uses stereotypical characters left and right, but then tries to keep them as incomprehensible as possible.<br /><br />Another problem with the characters is that they seemed to be dismissed with no explanation. I guess that's OK because so little time was spent developing the characters that I really didn't get a chance to know any of them, so I never really missed any of them.<br /><br />And last but not least was Sadie's singing. It's awful. The music backing her up is not prize winner, but it is usually drowned out by the screeches that are released from Sadie's vocal cords. I swear that there's one point in the movie where she sings a song for at least 10 minutes. I seriously thought I was going to have to turn it off during this howl-a-thon.<br /><br />As a whole, this movie is confusing. Characters are ill-developed, Georgia's acting is wooden and stiff, Sadie's character is yanked from one bad situation to another, with no back story or explanation. The music was unbearable, and I can think of no good reasons to see this film unless you have a thirst for cinematic pain."
    sample_text3_neg = "And look how a true story,' ... with a little help of it's friends... ': a very bad and terrible script, a bad directing and a surprising boring acting from a bunch of 'no-name' actors, especially from the 4-yr-old Jodelle Ferland, becomes a could have been a must seen movie. unfortuntaly 3/10."
    sample_text1_pos = "If Jean Renoir's first film 'Whirlpool of Fate' first takes us into the world of the countryside, the rivers, the lives of the peasantry that he will continue to explore, it seems only fitting that his second film deals for the most part with the wealthy and the privileged, the upper classes and those who are trying to claw their way upwards. Put the characters from the first two films together and you have the seeds of his great 'Grand Illusion' and 'Rules of the Game.' This is beautifully filmed, with the restless camera making full use of the amazingly huge apartments and backstage areas that dominate the film's interiors, and the acting though frequently overwrought offers some great moments as well, particularly from Werner Krauss' Muffat. But the glamorous and sultry Ms. Hessling, who at first appears as if she might give Louise Brooks a run for her money in vampishness, never goes beyond a one note, selfish harlot portrayal. Perhaps this is in part a problem with the script, which does seem to mostly go for high points and outraged emotions; not having read the novel I'm not really clear on whether the choices were well-made or not.<br /><br />Still, the differences between Nana's suitors are well-drawn, and I particularly liked the relationship between Muffat and Jean Angelo's Vandeuvres -- the tragic understandings that each seems to have of his ultimate fate and their sympathy with each other, particularly in the scene at the bottom of the enormous staircase where Vandeuvres warns Muffat, and we wonder if violence will erupt -- this and other gleanings of the ridiculousness of the idle rich help give the film the depth it has.<br /><br />Far from his greatest achievement, and for me probably just shy overall of 'Whirlpool of Fate', this is still well worth seeing for Renoir fans or those interested in silent cinema generally."
    sample_text2_pos = "Rented and watched this short (< 90 minutes) work. It's by far the best treatment Modesty has received on film -- and her creator, Peter O'Donnell, agrees, participating as a ;Creative Consultant.' The character, and we who love her, are handled with respect. Spiegel's direction is the best he's done to date, and the casting was very well done. Alexandra Staden is almost physically perfect as a match to the original Jim Holdaway illustrations of Modesty. A terrific find by whoever cast her! Raymond Cruz as a young Rafael Garcia was also excellent. I hope that Tarantino & co. will go on to make more in the series -- I'm especially interested to see whom they'd choose to be the incomparable Willie Garvin!"
    sample_text3_pos = "Normally, I don't watch action movies because of the fact that they are usually all pretty similar. This movie did have many stereotypical action movie scenes, but the characters and the originality of the film's premise made it much easier to watch. David Duchovny bended his normal acting approach, which was great to see. Angelina Jolie, of course, was beautiful and did great acting. Great cast all together. A must see for people bored with the same old action movie."
    sample = "the sky is blue and i am studying "
    #test_samples = [sample_text1_pos, sample_text2_pos, sample_text3_pos, sample_text2_neg, sample_text3_neg]
    test_samples = [sample]
    for item in test_samples:
        sentiment, confidence = predict_text(model, vectorize_layer, item)
        print(f" Predicted Sentiment: {sentiment} (Confidence: {confidence:.4f})")



#the model gest 4/5 correct but the confidence needs to be more precise. right now confidence is around 50%. 
#re run samples on other keras tokenizer sample - same results?


 



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