import sys
import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define class name for Cifar10 dataset
cifar10_ClassName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class linearClassifier (object):
    def __init__ (self, isTrain=1):

        # Define input
        self.x = tf.placeholder(name="x", shape=[None, 32, 32, 3], dtype=tf.float32)
        self.y = tf.placeholder(name="y", shape=[None], dtype=tf.int64)

        w1 = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.03), name='w1')  # 1st layer weights
        w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.03), name='w2')  # 2nd layer weights
        w3 = tf.get_variable(name="w3", shape=[9216, 1024])  # 3rd layer weights
        w4 = tf.get_variable(name="w4", shape=[1024, 10])  # 4th layer weights

        # --------------------------------
        # 1st Layer - Convolution
        # --------------------------------
        conv1 = tf.nn.conv2d(self.x, w1, [1, 1, 1, 1], padding='VALID')
        t1 = conv1
        # Batch normalization
        scale1 = tf.Variable(tf.ones([32]))
        beta1 = tf.Variable(tf.zeros([32]))
        batch_mean1, batch_var1 = tf.nn.moments(t1, [0])
        bn1 = tf.nn.batch_normalization(t1, batch_mean1, batch_var1, beta1, scale1, 1e-3)
        # Activation
        cnn1 = tf.nn.relu(bn1)

        # --------------------------------
        # 2nd Layer - Convolution
        # --------------------------------
        conv2 = tf.nn.conv2d(cnn1, w2, [1, 1, 1, 1], padding='VALID')
        # biases1 = tf.get_variable('B1', shape=(32))
        # t1 = tf.nn.bias_add(conv1x, biases1)
        t2 = conv2
        # Batch normalization
        scale2 = tf.Variable(tf.ones([64]))
        beta2 = tf.Variable(tf.zeros([64]))
        batch_mean2, batch_var2 = tf.nn.moments(t2, [0])
        bn2 = tf.nn.batch_normalization(t2, batch_mean2, batch_var2, beta2, scale2, 1e-3)
        # Activation
        cnn2 = tf.nn.relu(bn2)

        # Max Pool to reduce the dimmension
        pool2 = tf.nn.max_pool(cnn2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

        # --------------------------------
        # 3rd Layer - Fully Connected
        # --------------------------------
        shape = int(np.prod(pool2.get_shape()[1:]))
        cnn2_flat = tf.reshape(cnn2, [-1, shape])
        t3 = tf.matmul(cnn2_flat, w3)

        # Normalization first layer
        scale3 = tf.Variable(tf.ones([1024]))
        beta3 = tf.Variable(tf.zeros([1024]))
        batch_mean3, batch_var3 = tf.nn.moments(t3, [0])
        bn3 = tf.nn.batch_normalization(t3, batch_mean3, batch_var3, beta3, scale3, 1e-3)

        # Activation Third layer
        h3 = tf.nn.relu(bn3)

        # --------------------------------
        # Output Layer - Fully Connected
        # --------------------------------
        yOut = tf.matmul(h3, w4)

        # Loss
        totalLoss = tf.losses.hinge_loss(tf.one_hot(self.y, 10), logits=yOut)
        self.meanLoss = tf.reduce_mean(totalLoss) + 1e-5*tf.nn.l2_loss(w1) + 1e-5*tf.nn.l2_loss(w2) + 1e-5*tf.nn.l2_loss(w3) + 1e-5*tf.nn.l2_loss(w4)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(1e-5)
        self.trainStep = optimizer.minimize(self.meanLoss)

        # Correct Prediction and accuracy
        correctPrediction = tf.equal(tf.argmax(yOut, 1), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        # Predict index
        self.predict = tf.argmax(yOut, 1)

        # Initialize session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=None)

        # Log directory
        if isTrain == 1:
            if tf.gfile.Exists('./model'):
                tf.gfile.DeleteRecursively('./model')
            tf.gfile.MakeDirs('./model')
        else:
            self.saver.restore(self.sess, './model/model.ckpt')


    def train(self, xTr, yTr, xTe, yTe, maxSteps=1000, batchSize=128):
        print('{0:>7} {1:>12} {2:>12} {3:>12} {4:>12}'.format('Loop', 'Train Loss', 'Train Acc %', 'Test Loss', 'Test Acc %'))
        for i in range(maxSteps):
            # Shuffle data
            s = np.arange(xTr.shape[0])
            np.random.shuffle(s)
            xTr = xTr[s]
            yTr = yTr[s]

            # Train
            losses = []
            accuracies = []
            for j in range(0, xTr.shape[0], batchSize):
                xBatch = xTr[j:j + batchSize]
                yBatch = yTr[j:j + batchSize]
                trainLoss, trainAccuracy, _ = self.sess.run([self.meanLoss, self.accuracy, self.trainStep], feed_dict={self.x: xBatch, self.y: yBatch})
                losses.append(trainLoss)
                accuracies.append(trainAccuracy)
            avgTrainLoss = sum(losses) / len(losses)
            avgTrainAcc = sum(accuracies) / len(accuracies)

            # Test
            losses = []
            accuracies = []
            for j in range(0, xTe.shape[0], batchSize):
                xBatch = xTe[j:j + batchSize]
                yBatch = yTe[j:j + batchSize]
                testLoss, testAccuracy = self.sess.run([self.meanLoss, self.accuracy], feed_dict={self.x: xBatch, self.y: yBatch})
                losses.append(testLoss)
                accuracies.append(testAccuracy)
            avgTestLoss = sum(losses) / len(losses)
            avgTestAcc = sum(accuracies) / len(accuracies)

            # Log Output
            print('{0:>7} {1:>12.4f} {2:>12.4f} {3:>12.4f} {4:>12.4f}'.format(str(i+1)+"/"+str(maxSteps), avgTrainLoss, avgTrainAcc*100, avgTestLoss, avgTestAcc*100))

        savePath = self.saver.save(self.sess, './model/model.ckpt')
        print('Model saved in file: {0}'.format(savePath))

    def predictOutput(self, inputX):
        return self.sess.run([self.predict],feed_dict={self.x: inputX})

    def getAcc(self, inputX, inputY):
        return self.sess.run([self.accuracy],feed_dict={self.x: inputX, self.y: inputY})

def readImage(inputImage, meanValue):
    img = cv2.imread(inputImage)
    h_img, w_img, _ = img.shape
    imgResize = cv2.resize(img, (32, 32))
    imgRGB = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)
    imgResizeNp = np.asarray(imgRGB)
    imgResizeNp = np.expand_dims(imgResizeNp, axis=0)
    imgResizeNp = imgResizeNp.astype(np.float)
    imgResizeNp -= meanValue
    imgResizeNp = np.reshape(imgResizeNp, (imgResizeNp.shape[0], -1))
    return imgResizeNp

def getCifar10():
    (x1, y1), (x2, y2) = cifar10.load_data()

    # Format data
    x1 = x1.astype(np.float)
    x2 = x2.astype(np.float)
    y1 = np.squeeze(y1)
    y2 = np.squeeze(y2)

    # Normalize the data by subtract the mean image
    meanImage = np.mean(x1, axis=0)
    x1 -= meanImage
    x2 -= meanImage

    # Reshape data from channel to rows
    x1 = np.reshape(x1, (x1.shape[0], -1))
    x2 = np.reshape(x2, (x2.shape[0], -1))

    return (x1, y1), (x2, y2), meanImage

# Main function
if sys.argv[1] == "train":
    classify = linearClassifier(isTrain=1)
    (xTrain, yTrain), (xTest, yTest), mV = getCifar10()
    classify.train(xTrain, yTrain, xTest, yTest, maxSteps=10)

elif sys.argv[1] == 'test':
    classify = linearClassifier(isTrain=0)
    (xTrain, yTrain), (xTest, yTest), mV = getCifar10()
    print(cifar10_ClassName[np.squeeze(classify.predictOutput(readImage(sys.argv[2], mV)))])