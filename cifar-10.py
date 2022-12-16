################################################################################################
# Cifar-10 training & test tutorial

#Created by: Won Seok, Choi
#Commit Date: '17. 8. 24.
#Version: Python3

## Network Configuration
#Input: cifar-10 image: 32x32x3
#Network: 3 Convolution layers, 3 Pooling layers, 3 Fully-connected layers
#Optimizer: AdamOptimizer

## Set Hyper-Parameters
#There are 3 hyper-parameters to choose from. `Epochs, Batch size`, and `Drop out probability`.

#For example, set `Epochs: 10, Batch size: 128, Drop out: 0.75`
#python cifar-10.py -e 10 -b 128 -d 0.75

#If you do not set anything, the default settings will be used.
#python cifar-10.py
#default: `Epochs: 10, Batch size: 128, Drop out: 0.75`
################################################################################################

# Define HyperParameters
EPOCHS = 10
BATCH_SIZE = 128
KEEP_PROB = 0.75

########################################
## Step 0: Parser for Command Setting ##
########################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs', help='Set training epochs, default: 10')
parser.add_argument('-b','--batch_size', help='Set training batch size, default: 128')
parser.add_argument('-d','--drop_out', help='Set drop out keep_prob, default: 0.75')
args = vars(parser.parse_args())
if args['epochs']:
    print("Epochs setting: {}".format(args['epochs']))
    epochs = int(args['epochs'])
else:
    print("Epochs setting: {}".format(EPOCHS))
    epochs = EPOCHS
if args['batch_size']:
    print("Batch Size setting: {}".format(args['batch_size']))
    batch_size = int(args['batch_size'])
else:
    print("Batch Size setting: {}".format(BATCH_SIZE))
    batch_size = BATCH_SIZE
if args['drop_out']:
    if float(args['drop_out']) >= 1:
        print("**Notice: Drop out probability can not exceed 1")
        print("Drop out probability setting: {}".format(KEEP_PROB))
        drop_out = KEEP_PROB
    else:
        print("Drop out probability setting: {}".format(args['drop_out']))
        drop_out = float(args['drop_out'])
else:
    print("Drop out probability setting: {}".format(KEEP_PROB))
    drop_out = KEEP_PROB


##############################
## Step 1: Download Dataset ##
##############################
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer

import os
import tarfile
import pickle
import numpy as np
import tensorflow as tf

cifar10_dataset_folder_path = 'cifar-10-batches-py'
tar_gz_path = 'cifar-10-python.tar.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not os.path.exists(cifar10_dataset_folder_path):
    os.makedirs(cifar10_dataset_folder_path)

if not isfile(cifar10_dataset_folder_path + '/' + tar_gz_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            cifar10_dataset_folder_path + '/' + tar_gz_path,
            pbar.hook)

with tarfile.open(cifar10_dataset_folder_path + '/' + tar_gz_path,) as tar:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar)
    tar.close()


################################
## Step 2: Data Preprocessing ##
################################
def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    x_max = 255
    x_min = 0
    a = 0.0
    b = 0.1
    return a + (((x - x_min ) * (b - a) / (x_max - x_min)))

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    labels = np.array([0,1,2,3,4,5,6,7,8,9])
    lb = preprocessing.LabelBinarizer()
    lb.fit(labels)
    return lb.transform(x)

# Preprocess Training, Validation, and Testing Data
n_batches = 5

for batch_i in range(1, n_batches + 1):
    # Load cifar-10 batch
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_i), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    validation_count = int(len(features) * 0.1)

    # Prprocess and save a batch of training data
    features_training = normalize(features[:-validation_count])
    labels_training = one_hot_encode(labels[:-validation_count])
    pickle.dump((features_training, labels_training), open(cifar10_dataset_folder_path + '/preprocess_batch_' + str(batch_i) + '.p', 'wb'))

    # Preprocess and Save all validation data
    # Use a portion of training batch for validation
    features_validation = normalize(features[-validation_count:])
    labels_validation = one_hot_encode(labels[-validation_count:])
    pickle.dump((features_validation, labels_validation), open(cifar10_dataset_folder_path + '/preprocess_validation.p', 'wb'))


    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

        # load the test data
        test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        test_labels = batch['labels']
        # Preprocess and Save all test data
        features_test = normalize(test_features)
        labels_test = one_hot_encode(test_labels)
        pickle.dump((features_test, labels_test), open(cifar10_dataset_folder_path + '/preprocess_test.p', 'wb'))

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    features, labels = pickle.load(open(cifar10_dataset_folder_path + '/preprocess_batch_' + str(batch_i) + '.p', mode='rb'))

    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

valid_features, valid_labels = pickle.load(open(cifar10_dataset_folder_path + '/preprocess_validation.p', mode='rb'))
test_features, test_labels = pickle.load(open(cifar10_dataset_folder_path + '/preprocess_test.p', mode='rb'))


###################################
## Step 3: Define Neural Network ##
###################################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], mean=0, stddev=0.1, name="wc1")),
    'wc2': tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.1, name="wc2")),
    'wc3': tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], mean=0, stddev=0.1, name="wc3")),
    'wd1': tf.Variable(tf.truncated_normal(shape=[4096, 512], mean=0, stddev=0.1, name="wd1")),
    'wd2': tf.Variable(tf.truncated_normal(shape=[512, 512], mean=0, stddev=0.1, name="wd2")),
    'wd3': tf.Variable(tf.truncated_normal(shape=[512, 10], mean=0, stddev=0.1, name="wd3"))}

biases = {
    'bc1': tf.Variable(tf.zeros([64]), name="bc1"),
    'bc2': tf.Variable(tf.zeros([128]), name="bc2"),
    'bc3': tf.Variable(tf.zeros([256]), name="bc3"),
    'bd1': tf.Variable(tf.zeros([512]), name="bd1"),
    'bd2': tf.Variable(tf.zeros([512]), name="bd2"),
    'bd3': tf.Variable(tf.zeros([10]), name="bd3")}

# Build the Neural Network
def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # 32x32x3
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, biases['bc1'])
    conv1 = tf.nn.relu(conv1)
    # 32x32x64
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 16x16x64
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, biases['bc2'])
    conv2 = tf.nn.relu(conv2)
    # 16x16x128
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 8x8x128
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, biases['bc3'])
    conv3 = tf.nn.relu(conv3)
    # 8x8x256
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 4x4x256
    # For Flatten
    tensor_shape = pool3.get_shape().as_list()
    flat = tf.reshape(pool3, [tf.shape(pool3)[0], tensor_shape[1] * tensor_shape[2] * tensor_shape[3]])

    fc1 = tf.nn.relu(tf.add(tf.matmul(flat, weights['wd1']), biases['bd1']))
    drop1 = tf.nn.dropout(fc1, keep_prob)
    fc2 = tf.nn.relu(tf.add(tf.matmul(drop1, weights['wd2']), biases['bd2']))
    fc3 = tf.nn.bias_add(tf.matmul(fc2, weights['wd3']), biases['bd3'])

    return fc3

# Inputs
x = tf.placeholder(tf.float32, [None, 32, 32, 3], "x")
y = tf.placeholder(tf.float32, [None, 10], "y")
keep_prob = tf.placeholder(tf.float32, None, "keep_prob")

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

#####################################
## Step 4: Training Neural Network ##
#####################################


save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in batch_features_labels(features, labels, batch_size):
                sess.run(optimizer, feed_dict={x: batch_features, y: batch_labels, keep_prob: drop_out})
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            loss = sess.run(cost, feed_dict={x:batch_features, y:batch_labels, keep_prob:1.})
            valid_accuracy = sess.run(accuracy, feed_dict={x:valid_features, y:valid_labels, keep_prob:1.})
            print('Loss: {:.4f} Validation_Accuracy: {:.4f}'.format(loss, valid_accuracy))

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)


#####################################
## Step 5: Test Neural Network ##
#####################################
def test_model():
    """
    Test the saved model against the test dataset
    """
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0

        for test_feature_batch, test_label_batch in batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: test_feature_batch, loaded_y: test_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Test Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

print('Testing...')
test_model()
