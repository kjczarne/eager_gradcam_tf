import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import cv2


class tuple_dict(dict):
    """
    Custom dict object accepting tuples as 'keys' and unpacking their elements
    to get dict items corresponding to keys contained within the tuple.
    """
    def __init__(self, *args):
        dict.__init__(self, args)

    def __getitem__(self, i):
        if type(i) == tuple:
            lst = []
            for n in i:
                lst.append(dict.__getitem__(self, n))
            return lst
        elif type(i) == str:
            return dict.__getitem__(self, i)


def get_outputs_at_each_layer(model, input_image, layer_type):
    """
    Returns outputs and gradients of the score function with respect to
    those layer-wise outputs. Restricts layer type for gradient calculation
    to optimize derivation with respect to feature maps of convolutional layers.
    This function works when eager mode in TensorFlow is enabled.
    :param model: tf.keras.models Model or Sequential object
    :param input_image: input image as a tf.Tensor
    :param layer_type: layer class e.g. tf.keras.layers.Conv2D
    :return: tuple of ouptuts of respective layers and gradients associated with them
    """
    with tf.GradientTape() as tape:
        outputs = tuple_dict()  # custom dict object
        outputs['input_image'] = input_image  # initialize first input
        # this is because the first layer doesn't take input from any other layer below
        restricted_outputs = []

        nodes_by_depth_keys = list(model._nodes_by_depth.keys())
        nodes_by_depth_keys.sort(reverse=True)  # this is the proper order of executing ops
        for k in nodes_by_depth_keys:
            nodes = model._nodes_by_depth[k]
            for n in nodes:
                config = n.get_config()  # returns a dict
                # inbound_node in config gets the input node
                # outbound_node in config points to the operation
                if type(config['outbound_layer']) == list:  # convert lists to tuples
                    obl = tuple(config['outbound_layer'])
                else:
                    obl = config['outbound_layer']
                if type(config['inbound_layers']) == list:
                    if len(config['inbound_layers']) == 0:
                        # we expect only the first layer inbound nodes to be non-existent
                        ibl = 'input_image'
                    else:
                        ibl = tuple(config['inbound_layers'])
                else:
                    ibl = config['inbound_layers']
                out = model.get_layer(obl)(outputs[ibl])  # magic happens here
                # we call each outbound node with its inbound nodes...
                outputs[obl] = out  # ...and we append it back to the dict containing outputs
                # keys in the dict are names of layers/nodes, which allows this loop
                # to get them anytime multiple inputs are needed
                if isinstance(model.get_layer(obl), layer_type):
                    # we return only those layers that we want to see (conv)
                    restricted_outputs.append(outputs[obl])
    gradients = tape.gradient(out, restricted_outputs)  # record gradients
    return restricted_outputs, gradients


def grad_cam(image, model, image_dims, return_switch=None, watch_layer_instances=tf.keras.layers.Conv2D):
    """
    Grad-CAM visualization function.
    :param image: path to image as a string
    :param model: tf.keras.models Model or Sequential object
    :param image_dims: tuple specifying size of the output photo
    :param return_switch: 'gradients', 'maps', 'both', 'upsampled' or 'summed'
                          switches output of the function to return gradients,
                          feature maps, both gradients and feature maps,
                          upsampled feature maps or summed feature maps respectively
    :param watch_layer_instances: single class or tuple of classes of layers to watch
                                  this is useful for tracking special offshoots of conv layers
    :return: values as specified by return_switch
    This function produces Grad-CAM plots as a side effect
    """
    im = Image.open(image)
    im_tf = np.expand_dims((np.array(im.resize(image_dims))/255).astype(np.float32), axis=0)
    A_k, dy_dA_k = get_outputs_at_each_layer(model, tf.cast(im_tf, tf.float32), watch_layer_instances)
    L_c = [tf.keras.layers.ReLU()(tf.math.reduce_sum(tf.math.multiply(dy_dA_k[i], A_k[i]), axis=(3))) for i, _ in enumerate(dy_dA_k)]
    up_all = [np.array(Image.fromarray(i.numpy()[0, :, :]).resize(image_dims, resample=Image.BILINEAR)) for i in L_c]
    summed_maps = tf.keras.layers.ReLU()(np.sum(up_all, axis=0))
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(131)
    ax1.title.set_text('Gradient with 0.9 alpha')
    plt.axis('off')
    plt_im1 = plt.imshow(im_tf[0,:,:], cmap=plt.cm.gray, interpolation='bilinear')
    plt_im2 = plt.imshow(summed_maps.numpy(), cmap='magma', alpha=.95, interpolation='nearest')

    ax2 = fig.add_subplot(132)
    ax2.title.set_text('Gradient with 0.6 alpha')
    plt.axis('off')
    plt_im1 = plt.imshow(im_tf[0,:,:], cmap=plt.cm.gray, interpolation='bilinear')
    plt_im2 = plt.imshow(summed_maps.numpy(), cmap='magma', alpha=.6, interpolation='nearest')

    ax2 = fig.add_subplot(133)
    ax2.title.set_text('Original image')
    plt.axis('off')
    plt_im1 = plt.imshow(im_tf[0,:,:], cmap=plt.cm.gray, interpolation='bilinear')
    plt.subplots_adjust(
        hspace = 4,
        wspace = 0.3,
        right = 0.8,
        left = - 0.8
    )
    plt.show()

    if isinstance(return_switch, str):
        if (return_switch == 'gradients'):
            return dy_dA_k
        elif (return_switch == 'both'):
            return A_k, dy_dA_k
        elif (return_switch == 'maps'):
            return A_k
        elif (return_switch == 'upsampled'):
            return up_all
        elif (return_switch == 'summed'):
            return summed_maps
        else:
            return None
    elif (return_switch is None):
        return None
    else:
        raise RuntimeError('Invalid return value switch!')
