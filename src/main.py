## Notes taken from Deep Learning with Python by Francois Chollet Chapter 8 - Generative Learning
import os

import numpy as np

from keras.applications import inception_v3
from keras import backend as K

from utilities import resize_img, save_img, preprocess_image, deprocess_image

K.set_learning_phase(0)

""" Load pretrained inception v3 model """
model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

""" Set up the DeepDream configuration """
layer_contributions = {
    'mixed2' : 0.2,
    'mixed3' : 3.,
    'mixed4' : 2.,
    'mixed5' : 1.5
}

""" Define loss to be maximized """
# Create a dictionary that maps layer names to layer instancines
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# Define Loss by adding layer contributions to this variable
loss = 0.
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    # retrieve the layer's output
    activation = layer_dict[layer_name].output

    # Add the L2 norm of the features of a layer to the loss. Border artifacts are avoided by only involving nonborder pixels inthe loss.
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
#    print(loss)
    loss += coeff * K.sum(K.square(activation[:, 2:-2, 2:-2, :])) / scaling


""" Gradient Assent """
dream = model.input
grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

def gradient_ascent(x, iterations, step, max_loss = None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value >  max_loss:
            break
        print('...Loss value at', i, ":", loss_value)
        x += step * grad_values
    return x

step = 0.01
num_octave = 3
octave_scale = 1.4
iterations = 20
max_loss = 10.
test_image_name = "sunrise_virginia_beach"
base_image_path = os.getcwd() + os.path.sep + ".." + os.path.sep + "test_images" + os.path.sep + "sunrise_virginia_beach.jpg"

img = preprocess_image(base_image_path)
original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim/(octave_scale**i)) for dim in original_shape])
    successive_shapes.append(shape)

successive_shapes = successive_shapes[::-1]

original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img, iterations=iterations, step=step, max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='..' + os.path.sep + 'output' + os.path.sep + test_image_name + '_dream_at_scale_' + str(shape) + '.png')
save_img(img, fname='..' + os.path.sep + 'output' + os.path.sep + test_image_name+ '_final_dream.png')
