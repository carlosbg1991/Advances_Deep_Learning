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
        self.x = tf.placeholder(name="x", shape=[None, 3072], dtype=tf.float32)
        self.y = tf.placeholder(name="y", shape=[None], dtype=tf.int64)

        # First layer variable
        w1 = tf.get_variable(name="w1", shape=[3072, 1000])
        b1 = tf.get_variable(name="b1", shape=[1000])

        # Second layer variable
        w2 = tf.get_variable(name="w2", shape=[1000, 100])
        b2 = tf.get_variable(name="b2", shape=[100])

        # Third layer variable
        w3 = tf.get_variable(name="w3", shape=[100, 10])
        b3 = tf.get_variable(name="b3", shape=[10])

        # Compute first layer
        t1 = tf.matmul(self.x, w1) + b1

        # Normalization first layer
        scale1 = tf.Variable(tf.ones([1000]))
        beta1 = tf.Variable(tf.zeros([1000]))
        batch_mean1, batch_var1 = tf.nn.moments(t1, [0])
        bn1 = tf.nn.batch_normalization(t1, batch_mean1, batch_var1, beta1, scale1, 1e-3)

        # Activation first layer
        h1 = tf.nn.relu(bn1)

        # Compute second layer
        t2 = tf.matmul(h1, w2) + b2

        # Normalization second layer
        # scale2 = tf.Variable(tf.ones([1000]))
        # beta2 = tf.Variable(tf.zeros([1000]))
        # batch_mean2, batch_var2 = tf.nn.moments(t2, [0])
        # bn2 = tf.nn.batch_normalization(t2, batch_mean2, batch_var2, beta2, scale2, 1e-3)
        bn2 = t2  # No need to normalize the second layer

        # Activation second layer
        h2 = tf.nn.relu(bn2)

        # Last Output
        yOut = tf.matmul(h2, w3) + b3

        # Loss
        totalLoss = tf.losses.hinge_loss(tf.one_hot(self.y, 10), logits=yOut)
        self.meanLoss = tf.reduce_mean(totalLoss) + 1e-5*tf.nn.l2_loss(w1) + 1e-5*tf.nn.l2_loss(w2) + 1e-5*tf.nn.l2_loss(w3)

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