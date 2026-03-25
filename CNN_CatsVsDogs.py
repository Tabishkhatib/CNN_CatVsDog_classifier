
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import OneHotEncoder  
from sklearn.model_selection import train_test_split
from keras.utils import load_img, img_to_array
from keras.callbacks import EarlyStopping, ReduceLROnPlateau , ModelCheckpoint
from keras import regularizers
import tensorflow as tf
from keras.layers import LeakyReLU , SpatialDropout2D
import os
from datasets import load_dataset
ds = load_dataset("microsoft/cats_vs_dogs")

# Data generator: automatically rescales pixels to 0-1
# Create train / validation split
ds = ds["train"].train_test_split(test_size=0.2)

train_ds = ds["train"]
val_ds = ds["test"]
print("All images loaded:", train_ds)


def preprocess(example):
    image = example["image"]
    image = image.convert("RGB")
    image = image.resize(( 128, 128 ))  # force size
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image,0.1)
    image = np.array(image) / 255.0   # normalize

    return {
        "pixel_values": image,
        "labels": example["labels"]
    }
    
    
def preprocess_no_aug(example):
    image = example["image"]
    image = image.convert("RGB")
    image = image.resize((128, 128))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0

    return {
        "pixel_values": image,
        "labels": example["labels"]
    }  
train_tf = train_ds.map(preprocess)
val_tf = val_ds.map(preprocess_no_aug)

train_tf = train_tf.to_tf_dataset(
    columns=["pixel_values"],
    label_cols=["labels"],
    shuffle=True,
    batch_size=32,
)
dataset = train_tf.prefetch(buffer_size=tf.data.AUTOTUNE)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


val_tf = val_tf.to_tf_dataset(
    columns=["pixel_values"],
    label_cols=["labels"],
    shuffle=False,
    batch_size=32
)       


lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # halves the LR
    patience=2,
    min_lr=1e-6,
    verbose=1
)


model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=(128, 128, 3)))  # The first convolutional layer. Input shape is 128x128x32 and activation function is Rectified Linear Unit (ReLU) with l1 regularization
model.add(keras.layers.BatchNormalization())  
model.add(LeakyReLU(alpha=0.1))                                                      
  
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))  # The first maxpooling layer. The pool size and stride are both (2,2), thus the output shape is 64x64x32, where 64 = 128/2.

model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(SpatialDropout2D(0.1))

model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))  

model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),padding='same',kernel_regularizer=regularizers.L2(0.001)))
model.add(keras.layers.BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(SpatialDropout2D(0.2))

model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))  

model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),padding='same',kernel_regularizer=regularizers.L2(0.001)))
model.add(keras.layers.BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(SpatialDropout2D(0.2))

model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))  

model.add(tf.keras.layers.GlobalAveragePooling2D())                         # The previous output is flattened to be a vector with a size of 1x128.
model.add(keras.layers.Dense(128,kernel_regularizer=regularizers.L2(0.001)))
model.add(LeakyReLU(alpha=0.1))# The first fully connected layer.
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(84))
model.add(LeakyReLU(alpha=0.1))# The second fully connected layer.
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(64))
model.add(LeakyReLU(alpha=0.1))# The THIRD fully connected layer.
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(1, activation='sigmoid'))  
model.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), 'binary_crossentropy', metrics=['acc']) # Model construction with a Adam optimizer, a binary crossentropy loss function, and an accuracy metric.

model.summary() # Summary the constructed model.
checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True)

model.fit(train_tf, epochs = 60,validation_data=val_tf,callbacks=[early_stop,lr_reduce,checkpoint] ) 