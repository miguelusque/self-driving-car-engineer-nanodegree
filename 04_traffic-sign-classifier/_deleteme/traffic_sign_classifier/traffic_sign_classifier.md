
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle

train_file = 'data/train.p'
valid_file = 'data/valid.p'
test_file = 'data/test.p'

with open(train_file, mode='rb') as f:
    train = pickle.load(f)
with open(valid_file, mode='rb') as f:
    valid = pickle.load(f)
with open(test_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
import numpy as np

# Number of training examples
n_train = X_train.shape[0]

# Number of validation examples
n_valid = X_valid.shape[0]

# Number of testing examples.
n_test = X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# How many unique classes/labels there are in the dataset.
n_classes = np.unique(np.concatenate((y_train, y_valid, y_test))).shape[0]

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 34799
    Number of validation examples = 4410
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43



### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?






----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 

Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

As a first step, a [random seed](https://en.wikipedia.org/wiki/Random_seed) will be set for both numpy and tensorflow. This is done for [reproducibility purposes](https://machinelearningmastery.com/reproducible-results-neural-networks-keras/).


```python
import tensorflow as tf

# My humble tribute to Michael Jordan and Magic Johnson, the best BasketBall players ever. 
np.random.seed(23)
tf.set_random_seed(32)
```

    /Users/miguelangel/miniconda2/envs/gpu/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters



```python
# Normalize values between -1 and 1.
def scale(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    x = ((x - x.min())/(255 - x.min()))
    
    # scale to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x

X_train = [scale(x) for x in X_train]
X_valid = [scale(x) for x in X_valid]
X_test = [scale(x) for x in X_test]
```

### Model Architecture


```python
from tensorflow.contrib.layers import flatten

# Please notice this is a modified version of the MNIST LeNet network, 
# previously discussed in this Nanodegree.
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1   = tf.layers.batch_normalization(conv1, training=True)


    # Activation.
    conv1 = tf.nn.leaky_relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2   = tf.layers.batch_normalization(conv2, training=True)   

    # Activation.
    conv2 = tf.nn.leaky_relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        
    
    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 300.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 300), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(300))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1    = tf.nn.leaky_relu(fc1)

    # Layer 4: Fully Connected. Input = 300. Output = 200.
    fc8_W = tf.Variable(tf.truncated_normal(shape=(300, 200), mean = mu, stddev = sigma))
    fc8_b = tf.Variable(tf.zeros(200))
    fc8   = tf.matmul(fc1, fc8_W) + fc8_b
    
    # Activation.
    fc8    = tf.nn.leaky_relu(fc8)
    
    # Layer 4: Fully Connected. Input = 200. Output = 120.
    fc9_W = tf.Variable(tf.truncated_normal(shape=(200, 120), mean = mu, stddev = sigma))
    fc9_b = tf.Variable(tf.zeros(120))
    fc9   = tf.matmul(fc8, fc9_W) + fc9_b
    
    # Activation.
    fc9    = tf.nn.leaky_relu(fc9)
    
    # Layer 5: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc9, fc2_W) + fc2_b
    
    # Activation.
    fc2    = tf.nn.leaky_relu(fc2)
    fc2    = tf.nn.dropout(fc2, 0.8)

    # Layer 5: Fully Connected. Input = 84. Output = 60.
    fc7_W  = tf.Variable(tf.truncated_normal(shape=(84, 60), mean = mu, stddev = sigma))
    fc7_b  = tf.Variable(tf.zeros(60))
    fc7    = tf.matmul(fc2, fc7_W) + fc7_b
    
    # Activation.
    fc7    = tf.nn.leaky_relu(fc7)
    fc7    = tf.nn.dropout(fc7, 0.8)    
    
    # Layer 6: Fully Connected. Input = 60. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(60, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc7, fc3_W) + fc3_b
    
    return logits
```

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

#### Hyperparameters
The following [hyperparameters][1] will be used:
- [learning rate](https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10)
- [number of epochs](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)
- [batch size](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)

  [1]: https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)


```python
# Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
```

#### Training Pipeline
Create a training pipeline that uses the model to classify [German Traffic Sign](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) data.


```python
# x is a placeholder for a batch of input images
# y is a placeholder for a batch of output labels
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
training_operation = optimizer.minimize(loss_operation)
```

    WARNING:tensorflow:From <ipython-input-10-6825e50155a2>:8: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See tf.nn.softmax_cross_entropy_with_logits_v2.
    


#### Model Evaluation
Evaluate the loss and accuracy of the model for a given dataset.


```python
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data, batch_size):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

```

#### Train the Model
Run the training data through the training pipeline to train the model.

Before each epoch, shuffle the training set.

After each epoch, measure the loss and accuracy of the validation set.

Save the model after training.


```python
from sklearn.utils import shuffle

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid, BATCH_SIZE)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
```

    Training...
    
    EPOCH 1 ...
    Validation Accuracy = 0.800
    
    EPOCH 2 ...
    Validation Accuracy = 0.852
    
    EPOCH 3 ...
    Validation Accuracy = 0.867
    
    EPOCH 4 ...
    Validation Accuracy = 0.874
    
    EPOCH 5 ...
    Validation Accuracy = 0.883
    
    EPOCH 6 ...
    Validation Accuracy = 0.880
    
    EPOCH 7 ...
    Validation Accuracy = 0.900
    
    EPOCH 8 ...
    Validation Accuracy = 0.895
    
    EPOCH 9 ...
    Validation Accuracy = 0.895
    
    EPOCH 10 ...
    Validation Accuracy = 0.887
    
    EPOCH 11 ...
    Validation Accuracy = 0.900
    
    EPOCH 12 ...
    Validation Accuracy = 0.903
    
    EPOCH 13 ...
    Validation Accuracy = 0.912
    
    EPOCH 14 ...
    Validation Accuracy = 0.909
    
    EPOCH 15 ...
    Validation Accuracy = 0.911
    
    EPOCH 16 ...
    Validation Accuracy = 0.908
    
    EPOCH 17 ...
    Validation Accuracy = 0.912
    
    EPOCH 18 ...
    Validation Accuracy = 0.918
    
    EPOCH 19 ...
    Validation Accuracy = 0.930
    
    EPOCH 20 ...
    Validation Accuracy = 0.919
    
    EPOCH 21 ...
    Validation Accuracy = 0.915
    
    EPOCH 22 ...
    Validation Accuracy = 0.905
    
    EPOCH 23 ...
    Validation Accuracy = 0.928
    
    EPOCH 24 ...
    Validation Accuracy = 0.936
    
    EPOCH 25 ...
    Validation Accuracy = 0.936
    
    EPOCH 26 ...
    Validation Accuracy = 0.918
    
    EPOCH 27 ...
    Validation Accuracy = 0.922
    
    EPOCH 28 ...
    Validation Accuracy = 0.924
    
    EPOCH 29 ...
    Validation Accuracy = 0.922
    
    EPOCH 30 ...
    Validation Accuracy = 0.911
    
    EPOCH 31 ...
    Validation Accuracy = 0.923
    
    EPOCH 32 ...
    Validation Accuracy = 0.940
    
    EPOCH 33 ...
    Validation Accuracy = 0.912
    
    EPOCH 34 ...
    Validation Accuracy = 0.915
    
    EPOCH 35 ...
    Validation Accuracy = 0.932
    
    EPOCH 36 ...
    Validation Accuracy = 0.928
    
    EPOCH 37 ...
    Validation Accuracy = 0.920
    
    EPOCH 38 ...
    Validation Accuracy = 0.898
    
    EPOCH 39 ...
    Validation Accuracy = 0.928
    
    EPOCH 40 ...
    Validation Accuracy = 0.930
    
    EPOCH 41 ...
    Validation Accuracy = 0.929
    
    EPOCH 42 ...
    Validation Accuracy = 0.924
    
    EPOCH 43 ...
    Validation Accuracy = 0.935
    
    EPOCH 44 ...
    Validation Accuracy = 0.926
    
    EPOCH 45 ...
    Validation Accuracy = 0.917
    
    EPOCH 46 ...
    Validation Accuracy = 0.930
    
    EPOCH 47 ...
    Validation Accuracy = 0.949
    
    EPOCH 48 ...
    Validation Accuracy = 0.925
    
    EPOCH 49 ...
    Validation Accuracy = 0.932
    
    EPOCH 50 ...
    Validation Accuracy = 0.937
    
    Model saved


#### Evaluate the Model


```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test, BATCH_SIZE)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

    INFO:tensorflow:Restoring parameters from ./lenet
    Test Accuracy = 0.933


---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images


```python
import cv2
import glob
import os


# Load images previously downloaded from internet,
# change their color scheme from BGR to RGB, and
# resize them to 32x32.
path = './web_images/'
ext = '*.png'
files = glob.glob(path + ext)
X_web_images = []
y_web_images = []

for file in files:
    X_web_images.append(cv2.resize(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB), (32, 32)))
    y_web_images.append(file.replace(path, '').split('-')[0])
```


```python
# Output the images
num_images = len(X_web_images)
idx = range(num_images)
fig, axes = plt.subplots(1, num_images, sharex=True, sharey=True, figsize=(10,10),)
for ii, ax in zip(idx, axes.flatten()):
    ax.imshow(X_web_images[ii], aspect='equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
plt.subplots_adjust(wspace=0, hspace=0)
```


![png](output_38_0.png)


### Predict the Sign Type for Each Image


```python
# Pre-process the images with the same pre-processing pipeline used earlier.
X_web_images = [scale(x) for x in X_web_images]

# Run predictions using the model
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    graph = sess.run(logits, feed_dict = {x: X_web_images, y: y_web_images})
    labels = list(map(str, np.argmax(graph, axis=1)))
    
print(list(zip(y_web_images, labels)))
```

    INFO:tensorflow:Restoring parameters from ./lenet
    [('13', '10'), ('14', '14'), ('4', '3'), ('1', '1'), ('2', '1'), ('3', '3'), ('33', '33'), ('39', '39')]


### Analyze Performance


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_web_images, y_web_images, BATCH_SIZE)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

    INFO:tensorflow:Restoring parameters from ./lenet
    Test Accuracy = 0.625


### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
# Print out the top five softmax probabilities for the predictions 
# on the German traffic sign images found on the web. 

top_k = tf.nn.top_k(tf.nn.softmax(logits), k = 5)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    top_five = sess.run(top_k, feed_dict={x: X_web_images})
    
print(top_five)
```

    INFO:tensorflow:Restoring parameters from ./lenet
    TopKV2(values=array([[9.42677617e-01, 5.66513985e-02, 5.67347626e-04, 9.22235195e-05,
            4.95853237e-06],
           [9.97672975e-01, 2.31648562e-03, 5.88722696e-06, 3.79589937e-06,
            3.65934625e-07],
           [7.89872885e-01, 1.81987032e-01, 1.41359689e-02, 7.03625008e-03,
            6.95413072e-03],
           [9.99985933e-01, 1.40962975e-05, 3.23867128e-10, 7.44336884e-11,
            1.25958089e-11],
           [9.70629036e-01, 2.93710064e-02, 1.20532884e-09, 3.76975404e-12,
            1.08924631e-12],
           [9.85417485e-01, 1.37327751e-02, 7.68624945e-04, 5.87271425e-05,
            1.95346202e-05],
           [9.88372743e-01, 1.12068132e-02, 3.35038145e-04, 8.38659253e-05,
            1.58142689e-06],
           [9.99667287e-01, 3.32306197e-04, 1.46251253e-07, 7.05028853e-08,
            6.96263669e-08]], dtype=float32), indices=array([[10, 12, 42, 14, 35],
           [14, 10,  9, 15,  3],
           [ 3, 25,  4, 38, 33],
           [ 1,  2,  3,  4,  5],
           [ 1,  2,  4, 31,  0],
           [ 3,  2,  9, 13, 38],
           [33, 24, 16, 39, 40],
           [39, 33,  4, 17, 12]], dtype=int32))


### Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

---

## Step 4 (Optional): Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
```
