#!/usr/bin/env python3

"""Alex Net Architecture explanation in layman's term
60 million parameters
1. Preprocessing of input images
    1a. input image size: 227x227x3 https://discuss.pytorch.org/t/alexnet-input-size-224-or-227/41272
2. NN model
    2a. convolve with 96 kernel size: 11x11x3 with stride = 4, output = 55x55x96
    2b. max pooling with 3x3 kernel stride = 2, output = 27x27x96
    2c. convolve with 256 kernel size: 5x5 with same padding, output = 27x27x256
    2d. max pooling  with 3x3 kernel stride = 2, output = 13x13x256
    2e. convolve with 384 kernel size: 3x3 with same padding, output = 13x13x384
    2f. convolve with 384 kernel size: 3x3 with same padding, output = 13x13x384
    2g. convolve with 256 kernel size: 3x3 with same padding, output = 13x13x384
    2h. max pooling  with 3x3 kernel stride = 2, output = 6x6x256 (9216)
    2i. flatten the layer into 9216 neurons
    2j. Full connected layer: 4096
    2k. Full connected layer: 4096
    2l. softmax layer for output
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)
from keras.preprocessing.image import (
    ImageDataGenerator, 
)

if os.getenv("OS") == "MAC":
    DATASET_PATH_DIR = "/Users/vagrant/ml-scripts/data/cats_dogs_light/"
elif os.getenv("OS") == "LINUX_DOCKER":
    DATASET_PATH_DIR = "/home/data/cats_dogs_light/"


class AlexNetArchitecture(tf.keras.Sequential):

    def __init__(self, input_shape, num_classes):
        super().__init__()
        assert input_shape == (227, 227, 3), "bad shape"
    
    def compose(self):
        self.add(Conv2D(input_shape=input_shape, filters=96, kernel_size=(11, 11), strides=(4, 4)))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        self.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding="same"))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        self.add(Conv2D(filters=384, kernel_size=(3, 3), padding="same"))
        self.add(Conv2D(filters=384, kernel_size=(3, 3), padding="same"))
        self.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same"))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        self.add(Flatten())
        self.add(Dense(units=4096, activation= 'relu'))
        self.add(Dense(units=4096, activation= 'relu'))
        self.add(Dense(units=num_classes, activation= 'softmax'))

    def compile_model(self):
        self.compile(
            optimizer= tf.keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def fitting_model(self, train_size, train_generator, val_generator, epochs):
        batches = train_size / 100
        self.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            steps_per_epoch=batches,
        )

    def initialise_dataset(self):
        pass
    

    def predict_category(self):
        # self.predict()
        pass



def pull_images_dataset():
    assert os.path.isdir(DATASET_PATH_DIR) == True
    train_dir = os.path.join(DATASET_PATH_DIR, "train")
    test_dir = os.path.join(DATASET_PATH_DIR, "test")
    
    # cat = f"{train_dir}/cat.10185.jpg"
    # cat = plt.imread(cat)
    assert len(os.listdir(train_dir)) == 1000
    assert len(os.listdir(test_dir)) == 400
    return train_dir, test_dir


def convert_into_df(traindir):
    # 0. convert data into df format.
    # 1. split into train and validation set
    ret = []
    write = ret.append
    for image in os.listdir(traindir):
        assert image.split(".")[2] == "jpg"
        write("dog") if image.split(".")[0] == "dog" else write("cat")

    images_fname = os.listdir(traindir)
    df_set = pd.DataFrame({"images_filename": images_fname, "target": ret})

    train_df, validation_df = train_test_split(df_set, test_size=0.2, random_state=42)
    # reset indexs
    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    assert len(train_df) == 800
    assert len(validation_df) == 200
    return train_df, validation_df

def preprocessing_df_xcol(train_df, validation_df):
    # data augmentation
    train_augmentation = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    xtrain_generator = train_augmentation.flow_from_dataframe(
        train_df,
        os.path.join(DATASET_PATH_DIR, "train"),
        x_col="images_filename",
        y_col="target",
        target_size=(227,227),
        class_mode='categorical',
        batch_size=15
    )
    val_augmentation = ImageDataGenerator(rescale=1./255)
    val_generator = val_augmentation.flow_from_dataframe(
        validation_df,
        os.path.join(DATASET_PATH_DIR, "train"),
        x_col="images_filename",
        y_col="target",
        target_size=(227,227),
        class_mode='categorical',
        batch_size=15
    )
    return xtrain_generator, val_generator
    


if __name__ == "__main__":

    train, test = pull_images_dataset()
    traindf, valdf = convert_into_df(train)
    train_generator, val_generator = preprocessing_df_xcol(traindf, valdf)

    input_shape = (227, 227, 3)
    num_classes = 2
    alexnet = AlexNetArchitecture(input_shape, num_classes)
    alexnet.compose()
    alexnet.compile_model()
    alexnet.summary()
    alexnet.fitting_model(
        len(traindf),
        train_generator,
        val_generator,
        epochs=30,
    )