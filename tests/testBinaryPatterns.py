#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 15:09:21 2018

@author: rgarzon
"""

import unittest
import sys
import numpy as np

sys.path.append('../')

from BinaryPatterns import BinaryPatterns

SEQUENCE_LENGTH = 10
NUM_PATTERNS = 3
BATCH_SIZE = 4

class TestBinaryPatterns(unittest.TestCase):

    def test_generatePattern(self):
        bp = BinaryPatterns(seqLength=SEQUENCE_LENGTH)
        pat = bp._generatePattern()
        self.assertTrue(len(pat == SEQUENCE_LENGTH))

    def test_generateSequence(self):
        bp = BinaryPatterns(seqLength=SEQUENCE_LENGTH)
        seq = bp.generateSequence(numPatterns= NUM_PATTERNS)
        self.assertEqual(len(seq),(NUM_PATTERNS*10 + NUM_PATTERNS*3))
        self.assertEqual(seq[0].size,SEQUENCE_LENGTH)

    def test_generateBatchWithZeros(self):
        bp = BinaryPatterns(seqLength=SEQUENCE_LENGTH)
        batch,labels = bp.generateBatchWithZeros(BATCH_SIZE)
        print (batch)
        print (labels)
        print (type(batch))
        print (type(labels))
        labelsArray = np.array(labels)
        print (labelsArray.shape)
        self.assertEqual(len(batch),BATCH_SIZE)
        self.assertEqual(len(batch[0]),SEQUENCE_LENGTH)
    
    def test_getNumberOfOnes(self):
        bp = BinaryPatterns(seqLength=SEQUENCE_LENGTH)
        batch,labels = bp.generateBatchWithZeros(BATCH_SIZE)
        # Labels comes in OneHotEncoding format
        labels2 = np.array(bp._getNumberOfOnes(batch))
        # Labels 2 comes as an array of integers
        testOneHotEnc = bp._getOneHotEncoding(labels2[0])
        self.assertTrue(testOneHotEnc.all()==labels[0].all())

    def test__getOneHotEncoding(self):
        bp = BinaryPatterns(seqLength=SEQUENCE_LENGTH)
        oneHotEnc = bp._getOneHotEncoding(5)
        self.assertEqual(len(oneHotEnc),SEQUENCE_LENGTH+1)
        
        
if __name__ == '__main__':
    unittest.main()
