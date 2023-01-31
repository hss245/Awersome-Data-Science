import numpy as np
import cv2
import json

class ImageClassificationInference:

    def __init__(self):
        self.IMG_HEIGHT = 200
        self.IMG_WIDTH = 200
        with open("classesMapping.json", 'r') as file:   
            self.MAPPING_DICT = json.load(file)
        self.MAPPING_DICT = {v : k for k,v in self.MAPPING_DICT.items()}
    
    def imageProcessing(self, imageName):
        _image = cv2.imread(imageName)
        _image = cv2.resize(_image, (self.IMG_HEIGHT, self.IMG_WIDTH))
        _image = np.expand_dims(_image, axis = 0)
        return _image
    
    def modelInferencing(self, imageName, model):
        _image = self.imageProcessing(imageName)
        _prediction = model.predict(_image)
        _prediction = np.argmax(_prediction)
        _prediction = self.MAPPING_DICT.get(_prediction)
        return _prediction