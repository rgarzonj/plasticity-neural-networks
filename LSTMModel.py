#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 21:28:05 2018

@author: rgarzon
"""
import tensorflow as tf

from BinaryPatterns import BinaryPatterns
from tensorflow.contrib import rnn
import numpy as np
# Training Parameters
learning_rate = 0.001
training_steps = 6000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 1 # MNIST data input (img shape: 28*28)
timesteps = 5 # timesteps
num_hidden = 64 # hidden layer num of features
num_classes = timesteps+1 # MNIST total classes (0-9 digits)

class LSMTModel():

    def __init__(self,timesteps,batch_size,num_input):        
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.num_input = num_input
        
    def RNN(self,x, weights, biases):
    
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    
        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, self.timesteps, 1)
    
        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    
        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        print (outputs)
        print (states)
        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    
    def train(self,trainingDataObject):
        ''' Used to train the model with few-shot learning 
        Args:
                    
    
        Returns:    
        '''
        # tf Graph input
        X = tf.placeholder("float", [None, self.timesteps, self.num_input])
        Y = tf.placeholder("float", [None, num_classes])
        
        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([num_classes]))
        }        
        
        logits = self.RNN(X, weights, biases)
        prediction = tf.nn.softmax(logits)
        
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        train_op = optimizer.minimize(loss_op)
        
        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        
        # Start training
        with tf.Session() as sess:
        
            # Run the initializer
            sess.run(init)
        
            for step in range(1, training_steps+1):
                batch_x,batch_y = trainingDataObject.generateBatchWithZeros(self.batch_size)
                batch_x = np.array(batch_x)
                batch_y = np.array(batch_y)
                #print (batch_x.shape)
                #print (batch_y.shape)
#                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Reshape data to get 28 seq of 28 elements
                batch_x = batch_x.reshape((self.batch_size, self.timesteps, num_input))
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                         Y: batch_y})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))
        
            print("Optimization Finished!")
            
            test_len = 1
            test_data,test_label = bp.generateBatchWithZeros(batch_size)
            test_data = np.array(test_data)
            test_data = test_data.reshape((-1, timesteps, num_input))
            test_label = np.array(test_label)
                
            print("Testing Accuracy:")

            pred,acc = sess.run(correct_pred,accuracy, feed_dict={X: test_data, Y: test_label})
            print (pred)
            print(acc)

            # Save the variables to disk.
            save_path = saver.save(sess, "./models/model.ckpt")
            print("Model saved in file: %s" % save_path)
            
    def load_model():
        # Create some variables.

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Later, launch the model, use the saver to restore variables from disk, and
        # do some work with the model.
        with tf.Session() as sess:
          # Restore variables from disk.
          saver.restore(sess, "/models/model.ckpt")
          print("Model restored.")
          # Do some work with the model


lstmM = LSMTModel(timesteps,batch_size,num_input)
bp = BinaryPatterns(timesteps)
lstmM.train(bp)





