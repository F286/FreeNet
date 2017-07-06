import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

img = tf.placeholder(tf.float32, shape=(None, 784))

from keras.layers import Dense, Activation

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.nn.elu(x, alpha)

# Keras layers can be called on TensorFlow tensors:
x = Dense(128)(img)
# x = Activation(K.relu)(x)
# x = Activation(K.tanh)(x)
print(type(x))
x = selu(x)
print(type(x))
x = Dense(128)(x)
x = Activation(K.relu)(x)
preds = Dense(10, activation=K.softmax)(x)

labels = tf.placeholder(tf.float32, shape=(None, 10))

from keras.objectives import categorical_crossentropy
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Run training loop
print("Training..")
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1]})

from keras.metrics import categorical_accuracy as accuracy
import numpy as np

print("Testing..")
acc_value = accuracy(labels, preds)
with sess.as_default():
    t = acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels})
    print(np.mean(t) * 100)

