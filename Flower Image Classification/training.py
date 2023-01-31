import numpy as np
import cv2
import os
import logging
import json
import tensorflow as tf
from datetime import datetime
logging.basicConfig(filename = 'logs.txt', level = logging.INFO, format = '%(message)s')

class ImageClassificationTraining:

    def __init__(self):
        self.BATCH_SIZE = 32
        self.IMG_HEIGHT = 200
        self.IMG_WIDTH = 200
        self.TRAINING_DIRECTORY = "flowerData/"
    
    def dataSetCreation(self):
        """
        Creating empty list to store images and the label (flower name)
        """
        images, labels = [], []
        labelsDictionary = {}
        counter = -1

        """
        Using os library to iterate over main folder and inside that each subfolder contains images of flowers. 
        Additionally, name of each subfolder is the label also.
        Hence, below code iterates over main folder and inside each subfolder to load every image. It converts them to a size of 200 X 200
        a standard size so that CNN model can be trained and the label for that image is the name of the 
        subfolder inside which image file is present.
        """
        for folder in os.listdir(self.TRAINING_DIRECTORY):
            if not folder.endswith('.DS_Store'):
                for file in os.listdir(self.TRAINING_DIRECTORY + folder):
                    _image = cv2.imread(self.TRAINING_DIRECTORY + folder + '/' + file)
                    _image = cv2.resize(_image, (self.IMG_HEIGHT, self.IMG_WIDTH))
                    
                    images.append(_image)
                    if folder not in labelsDictionary:
                        counter += 1
                        labelsDictionary[folder] = counter
                    labels.append(counter)
        
        """
        Converting training images and labels list to numpy arrays. 
        """
        self.images = np.array(images)
        self.labels = np.array(labels)

        """
        Finding number of unique classes for passing it to CNN in order for Multi Class Classification
        """
        self.NUM_CLASSES = len(labelsDictionary.keys())

        with open("classesMapping.json", 'w') as file:
            json.dump(labelsDictionary, file, indent = 6)
    
    def modelCreation(self):
        self.model = tf.keras.models.Sequential(
            [
            tf.keras.layers.experimental.preprocessing.Rescaling(1/255, input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding = 'same', activation = 'relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation = 'relu'),
            tf.keras.layers.Dense(self.NUM_CLASSES)
            ],
                                                )

    def modelTraining(self):
        self.dataSetCreation()
        logging.info(f"Data Successfully Loaded at {datetime.now().strftime('%d-%m-%Y:%H-%M-%S')}")

        self.modelCreation()
        logging.info(f"CNN Model Created at {datetime.now().strftime('%d-%m-%Y:%H-%M-%S')}")
        
        self.model.compile(optimizer = "adam",
                          loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics = ['accuracy']
                          )
        logging.info(f"Model Compiled at {datetime.now().strftime('%d-%m-%Y:%H-%M-%S')}")

        logging.info(f"Model Training started at {datetime.now().strftime('%d-%m-%Y:%H-%M-%S')}")
        history = self.model.fit(self.images, self.labels,
                                epochs = 10,
                                batch_size = self.BATCH_SIZE)
        logging.info(f"Model Training completed at {datetime.now().strftime('%d-%m-%Y:%H-%M-%S')}")

        self.model.save("flowerTFModel")
        logging.info(f"Model successfully saved at {datetime.now().strftime('%d-%m-%Y:%H-%M-%S')}")


if __name__ == "__main__":
    imgClassification = ImageClassificationTraining()
    imgClassification.modelTraining()