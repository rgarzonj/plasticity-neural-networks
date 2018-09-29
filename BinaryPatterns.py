#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 15:43:28 2018

@author: rgarzon
"""
import numpy as np
import random

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class BinaryPatterns():
    
    def __init__(self,seqLength):
        self.SEQUENCE_LENGTH = seqLength
        self.patterns=[]
        self.sequence =[]

    def _generatePattern(self):
        """Generates a series of -1 and 1 of SEQUENCE_LENGTH.
    Args:
        -
    Returns:
        the sequence of 1s and -1s values

        """
        oneSeq = np.empty(self.SEQUENCE_LENGTH)
        for j in range(self.SEQUENCE_LENGTH):
            oneSeq[j]= random.choice((-1,1))
        return oneSeq

    def generateBatchWithZeros(self,batch_size):
        data = []
        labels = []
        for oneItem in range(batch_size):    
            pat = self._generatePatternWithZeros()
            #print (np.sum(pat))
            label = self._getOneHotEncoding(np.sum(pat))
            data.append(pat)
            labels.append(label)            
        return data,labels

    def _getNumberOfOnes(self,batch):
        labels = []
        for oneItem in batch:
            labels.append(np.sum(oneItem))
        return labels
    
    def _getOneHotEncoding(self,integer):
        integer = int(integer)
        #print ('input')
        #print (integer)
        data = np.arange(self.SEQUENCE_LENGTH+1)        
        values = array(data)
#        print(values)
        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
#        print(integer_encoded)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        #print (onehot_encoded)
        return onehot_encoded[integer]
    
    def _generatePatternWithZeros(self):
        """Generates a series of 0 and 1 of SEQUENCE_LENGTH.
    Args:
        -
    Returns:
        the sequence of 1s and -1s values
        """
        oneSeq = np.empty(self.SEQUENCE_LENGTH)
        for j in range(self.SEQUENCE_LENGTH):
            oneSeq[j]= random.choice((0,1))
        return oneSeq
        
    def generateSequence(self,numPatterns):
        """Generates the experiment sequence according to the paper.

    Args:
        numPatterns: Number of patterns to generate in the sequence (default is 3)

    Returns:
        sequence: The sequence of patterns

        """

        #Generate the sequence, each pattern is shown for 10 time steps
        #between representations 3 steps of zero input
        #The whole sequence is presented 3 times in random order
        TIME_STEPS_TO_SHOW = 10
        TIME_STEPS_BETWEEN_PATTERNS = 3
        NUMBER_OF_TIMES_IN_RANDOM_ORDER = 3
        for a in range (numPatterns):
            self.patterns.append(self._generatePattern())            
        shown = []
        for onePattern in range(NUMBER_OF_TIMES_IN_RANDOM_ORDER): 
            which = random.randint(0,NUMBER_OF_TIMES_IN_RANDOM_ORDER-1)
            while (which in shown):                    
                which = random.randint(0,NUMBER_OF_TIMES_IN_RANDOM_ORDER-1)
            shown.append(which)    
        for which in shown:
            for i in range (TIME_STEPS_TO_SHOW):
                self.sequence.append(self.patterns[which])
            for i in range(TIME_STEPS_BETWEEN_PATTERNS):
                self.sequence.append(np.zeros(self.SEQUENCE_LENGTH))
        return self.sequence
        
    def getPatterns(self):
        return self.patterns