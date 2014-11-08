#coding: utf-8
import numpy as np

from exception.inputexception import *

class LinearNetwork(object):
    def __init__(self, l = None, n = None, maxw = 0.01, W = None, rng = np.random):
        if l != len(n):
            raise InputException

        self.l = l
        self.n = n

        if W == None:
            self.W = []
            for i in xrange(l - 1):
                self.W.append( rng.rand(n[i+1], n[i]) * maxw )
        else:
            self.W = W

    def run(self, input = None):
        tmpans = input
        for i in xrange(self.l - 1):
            tmpans = np.dot(self.W[i], tmpans)

        return tmpans

    def set_W(self, W):
        self.W = W
