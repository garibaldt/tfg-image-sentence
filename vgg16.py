# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX
import numpy as np
import matplotlib.pyplot as plt
import urllib
import io
import skimage.transform
import json
import os
from six.moves import cPickle


def build_vgg16():
    """The structure of our model is adapted so we can use pretrained models.
    See 'https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg16.py'
    for details.

    """

    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2, flip_filters=False)
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
    net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
    net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
    net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
    output_layer = net['fc8']
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net


def get_features(
        path_pkl="./pretrained/vgg_cnn_s.pkl",
        path_json="./data/dataset_flickr8k.json",
        path_images="./data/flickr8k/",
        path_features="./data/flickr8k_features/"
    ):
    """Extract feature vectors for the image urls in "path_json"
    (where the urls and captions of the dataset are)
    using the network. We use the output of the fc7 layer,
    before the softmax clasifier (fc8 and prob8). For each image,
    its feature vector is stored. This feature vector will be used
    by the RNN.

    JSON structure:

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
    model = cPickle.load(open(path_pkl))
    vgg = build_vgg16()
    MEAN_IMAGE = model['mean image']

    lasagne.layers.set_all_param_values(vgg['fc8'], model['values'])

    data = json.load(open(path_json))

    for img_elem in data['images']:
        print 'Processing image {}'.format(img_elem['imgid'])
        img_id = img_elem['filename'].split('.')[0]
        print img_id
        img_file = os.path.join(path_images, img_elem['filename'])
        _, img = preprocess_img(img_file, MEAN_IMAGE)

        # the feature is a 1d array of shape (4096)
        feature = np.array(lasagne.layers.get_output(vgg['drop7'], img, deterministic=True).eval())

        f = open(os.path.join(path_features, img_id + '.save'), 'wb')
        cPickle.dump(feature, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()



def preprocess_img(path_img, mean=0):
    """Transforms the input image into a np.array of shape (224, 224, 3) to be
    fed into the network.

    Returns the raw image np.array and the preprocessed image np.array

    """

    im = plt.imread(path_img)

    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]

    rawim = np.copy(im).astype('uint8')

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - mean

    return rawim, floatX(im[np.newaxis])



if __name__ == '__main__':
    # let's experiment...
    #
    model = cPickle.load(open('./pretrained/vgg_cnn_s.pkl'))
    vgg = build_vgg16()
    CLASSES = model['synset words']
    MEAN_IMAGE = model['mean image']

    lasagne.layers.set_all_param_values(vgg['fc8'], model['values'])

    # testing the model with some images
    image_urls = [
        "http://farm1.static.flickr.com/8/12567442_838940c1f1.jpg",
        "http://static.flickr.com/1372/582719105_4e4016397e.jpg",
        "http://static.flickr.com/229/499621053_7aa6875b19.jpg",
        "http://static.flickr.com/1144/1164305600_969cb0e5ac.jpg",
        "http://www.telegraph.co.uk/content/dam/Travel/Destinations/Europe/France/Paris/paris-attractions-xlarge.jpg",
        "http://www.image-net.org/nodes/11/02084071/4a/4a00095ae0575cad61012dd04b6bd8a02d595c7d.thumb",
        "http://promocionmusical.es/wp-content/uploads/2015/06/ep.jpg",
    ]

    def prep_image(url):
        ext = url.split('.')[-1]
        im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)

        # Resize so smallest dim = 256, preserving aspect ratio
        h, w, _ = im.shape
        if h < w:
            im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
        else:
            im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

        # Central crop to 224x224
        h, w, _ = im.shape
        im = im[h//2-112:h//2+112, w//2-112:w//2+112]

        rawim = np.copy(im).astype('uint8')

        # Shuffle axes to c01
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

        # Convert to BGR
        im = im[::-1, :, :]

        im = im - MEAN_IMAGE
        return rawim, floatX(im[np.newaxis])


    lista = []

    for url in image_urls:
        try:
            rawim, im = prep_image(url)

            prob = np.array(lasagne.layers.get_output(vgg['fc8'], im, deterministic=True).eval())
            top5 = np.argsort(prob[0])[-1:-6:-1]

            # plot image and 5 categories
            plt.figure()
            plt.imshow(rawim.astype('uint8'))
            plt.axis('off')
            for n, label in enumerate(top5):
                plt.text(250, 70 + n * 20, '{}. {}'.format(n+1, CLASSES[label]), fontsize=14)
        except IOError:
            print('bad url: ' + url)

        lista.append(im)

    print(type(np.concatenate(lista, axis=0)))
    print(np.concatenate(lista, axis=0).shape)
    prob = np.array(lasagne.layers.get_output(vgg['fc8'], np.concatenate(lista, axis=0), deterministic=True).eval())


