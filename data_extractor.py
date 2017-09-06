from collections import Counter
from itertools import chain
from string import digits
import os
import pickle
import time

import json
import h5py
import numpy as np
# import pandas as pd


class DataExtractor(object):
    """
    Data extraction and preprocessing
    """
    
    def __init__(self, data_file='./dataset_flickr8k.json', max_sentence_length=20,
                 extract_features=False, img_path='./dataset/flickr8k/', 
                 feat_path='./dataset/flickr8k_features/', cnn='vgg16'):
        
        self.data_file = data_file
        self.max_caption_length = max_caption_length
        self.extract_features = extract_features
        self.img_path = img_path
        self.feat_path = feat_path
        self.cnn = cnn

        # for vgg16 and vgg19, the size of the features is 4096
        # can be different if we use another cnn architecture
        self.feat_size = 4096

        # set the value of special tokens (start, end and unknown)
        self.start_token = 'START'
        self.end_token = 'END'
        self.unk_token = 'UNK'

    def preprocess_data(self):
        print 'Preprocessing the data:'
        start_time = time.monotonic()
        
        # load images and captions
        self.load(self.data_file)
        # get info about the data
        self.get_corpus_info()
        # extract image features and dump results (only the first time) 
        if self.extract_features:
            self.get_features(self.img_path)
            self.dump_features()


    def load(data_file):
        """
        JSON structure

        dataset_flickr8k:
            [images]
                filename: "9i78971290.jpg"
                imgid: n
                [sentences]
                    tokens: ["la", "mesa", ...]
                    raw: "La mesa..."
                    imgid: int
                    sentid: int

                split: 'train', 'val', 'test'
                sentids: [int]

            dataset: "flickr8k"
        """

        print 'Loading data...'

        f = json.load(open(data_file))
        data = []

        for img_elem in data['images']:
            img_filename = img_elem['filename']
            for sent in img_elem['sentences']:
                data.append((img_filename, sent))
        
        data = np.asarray(data)
        # shuffle dataset for training
        np.random.shuffle(data)
        self.images = data[:, 0]
        self.sentences = data[:, 1]
        # 40000 for the flickr8k dataset
        self.data_size = self.images.shape[0]
        print data_size
        

    def get_corpus_info(self):
        self.word_frequencies = Counter(chain(*self.captions)).most_common()      


    def get_image_features(self):
        """
        use a CNN to extract image features
        """
        
        # change this when using other architecture
        from keras.preprocessing import image
        from keras.models import Model
        from keras.applications.vgg16 import preprocess_input
        from keras.applications import VGG16

        # load VGG16 architecture, initialized with the imagenet weights 
        # (i.e. pretrained on imagenet)
        model_cnn = VGG16(weights='imagenet')
        # the features are the outputs of the second fc layer of the cnn
        feat_extractor = Model(input=model_cnn.input,
                               output=model_cnn.get_layer('fc2').output)
        self.img_features = []
        # create a list of images (without repetition)
        self.image_features_files = list(set(self.image_files))
        # 8000 for the flickr8k dataset
        self.nb_images = len(self.image_features_files)
        for img_file in self.image_feature_files:
            # load image and reshape it
            img = image.load_img(self.img_path + img_file, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            CNN_features = feat_extractor.predict(img)
            self.img_features.append(np.squeeze(CNN_features)
        
        self.img_features = np.asarray(self.extracted_features)


    def dump_features(self):
        """
        SAVE FEATURES
        """
        


if __name__ == '__main__':
    data_ext = DataExtractor()
    data_ext.preprocess_data()
