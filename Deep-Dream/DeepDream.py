
# coding: utf-8

# In[2]:


from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b
import time

from keras.applications import vgg16
from keras import backend as K
from keras.layers import Input
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


img_width, img_height = 800, 600

image_path = './2007_000323.jpg'
output_name = 'results/im'


# In[4]:


img_size = (img_height, img_width, 3)

inp_img = Input(batch_shape=(1,) + img_size)


# In[5]:


model = vgg16.VGG16(input_tensor=inp_img,
                    weights='imagenet', include_top=False)


# In[6]:


model.summary()


# In[7]:


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    # img = np.asarray(255*np.random.random((img_height, img_width, 3)), dtype='uint8')
    img = img_to_array(img)
    # print img.shape
    img = np.expand_dims(img, axis=0)
    # print img.shape
    img = vgg16.preprocess_input(img)
    # print img.shape
    return img


# In[8]:


def deprocess_image(x):
    x = x.reshape((img_height, img_width, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# In[9]:


layer_dict = dict([(layer.name, layer) for layer in model.layers])

for key in layer_dict:
    print key, layer_dict[key]


# In[1]:


layer_out = layer_dict['block4_conv2'].output
layer_out_01 = layer_dict['block5_conv2'].output


loss = -0.2*K.mean(layer_out[:, :, :, 5])
loss -= 0.3*K.mean(layer_out_01[:, :, :, 10])
loss += 0.05*K.mean(inp_img)

# loss = -K.mean(model.output[5])


# In[30]:


grads = K.gradients(loss, inp_img)
# print grads

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([inp_img], outputs)


# In[31]:


def eval_loss_and_grads(x):
    x = x.reshape((1,) + img_size)
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grad_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()


# In[32]:


settings = {
    'jitter': 0,
}

# fmin_l_bfgs_b?
all_loss = []


# In[33]:


x = preprocess_image('./2007_001834.jpg')

for i in range(15):
    print 'Start of iteration', i
    start_time = time.time()

    random_jitter = (settings['jitter'] * 2) * (np.random.random(img_size) - 0.5)
    x += random_jitter

    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=7)
    print 'Current loss value:', min_val
    all_loss.append(min_val)
    
    # Decode the dream and save it
    x = x.reshape(img_size)
    x -= random_jitter
    img = deprocess_image(np.copy(x))
    
    plt.figure(i)
    plt.imshow(img)
    
    fname = 'results/new_img' + '_at_iteration_%d.png' % i
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
plt.show()


# In[36]:


plt.plot(all_loss)

