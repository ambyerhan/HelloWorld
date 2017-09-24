# ambyer
# 2017.06.08

import theano
import theano.tensor as T
import numpy as np
import numpy.random as rnd
import copy
from datetime import datetime
from layers import CEmbeddingLayer
from utils import get_maxlen, \
					set_eos, \
					get_Hout_3DMask, \
					get_Oout_3DMask
	
# ABORD			
class CGRU(object):
	def __init__(self, edim, sdim, vsize, minibatch, threshold = 0.1, bptt_trun = -1):
		"""
		>>
		> edim : the dim of the pre-layer
		> sdim : size of hidden dim
		> vsize : the size of vocabulary
		> minibatch : the batch size
		> threshold : the threshold
		> bptt_trun : used in scan
		"""
		self.sdim = sdim
		self.vsize = vsize
		self.minibatch = minibatch
		self.bptt_trun = bptt_trun
				
		# hidden layer-1
		# the z-gate vals
		ZU1 = self.threshold * rnd.randn(edim + sdim, sdim).astype(theano.config.floatX)
		ZW1 = self.threshold * rnd.randn(sdim, sdim).astype(theano.config.floatX)
		ZB1 = self.threshold * rnd.randn(sdim).astype(theano.config.floatX)
		
		# the r-gate vals
		RU1 = self.threshold * rnd.randn(edim + sdim, sdim).astype(theano.config.floatX)
		RW1 = self.threshold * rnd.randn(sdim, sdim).astype(theano.config.floatX)
		RB1 = self.threshold * rnd.randn(sdim).astype(theano.config.floatX)
		
		# the g-gate vals
		GU1 = self.threshold * rnd.randn(edim + sdim, sdim).astype(theano.config.floatX)
		GW1 = self.threshold * rnd.randn(sdim, sdim).astype(theano.config.floatX)
		GB1 = self.threshold * rnd.randn(sdim).astype(theano.config.floatX)
		
		# set shared vals
		self.ZU1 = theano.shared(value = ZU1, name = "ZU1")
		self.ZW1 = theano.shared(value = ZW1, name = "ZW1")
		self.ZB1 = theano.shared(value = ZB1, name = "ZB1")
		self.RU1 = theano.shared(value = RU1, name = "RU1")
		self.RW1 = theano.shared(value = RW1, name = "RW1")
		self.RB1 = theano.shared(value = RB1, name = "RB1")
		self.GU1 = theano.shared(value = GU1, name = "GU1")
		self.GW1 = theano.shared(value = GW1, name = "GW1")
		self.GB1 = theano.shared(value = GB1, name = "GB1")
		
		# hidden layer-2
		# the z-gate vals
		ZU2 = self.threshold * rnd.randn(sdim, sdim).astype(theano.config.floatX)
		ZW2 = self.threshold * rnd.randn(sdim, sdim).astype(theano.config.floatX)
		ZB2 = self.threshold * rnd.randn(sdim).astype(theano.config.floatX)
		
		# the r-gate vals
		RU2 = self.threshold * rnd.randn(sdim, sdim).astype(theano.config.floatX)
		RW2 = self.threshold * rnd.randn(sdim, sdim).astype(theano.config.floatX)
		RB2 = self.threshold * rnd.randn(sdim).astype(theano.config.floatX)
		
		# the g-gate vals
		GU2 = self.threshold * rnd.randn(sdim, sdim).astype(theano.config.floatX)
		GW2 = self.threshold * rnd.randn(sdim, sdim).astype(theano.config.floatX)
		GB2 = self.threshold * rnd.randn(sdim).astype(theano.config.floatX)
		
		# set shared vals
		self.ZU2 = theano.shared(value = ZU2, name = "ZU2")
		self.ZW2 = theano.shared(value = ZW2, name = "ZW2")
		self.ZB2 = theano.shared(value = ZB2, name = "ZB2")
		self.RU2 = theano.shared(value = RU2, name = "RU2")
		self.RW2 = theano.shared(value = RW2, name = "RW2")
		self.RB2 = theano.shared(value = RB2, name = "RB2")
		self.GU2 = theano.shared(value = GU2, name = "GU2")
		self.GW2 = theano.shared(value = GW2, name = "GW2")
		self.GB2 = theano.shared(value = GB2, name = "GB2")
		
		# set params
		self.params = [self.ZU1, self.ZW1, self.ZB1,
						self.RU1, self.RW1, self.RB1,
						self.GU1, self.GW1, self.GB1,
						self.ZU2, self.ZW2, self.ZB2,
						self.RU2, self.RW2, self.RB2,
						self.GU2, self.GW2, self.GB2]
						
	def __activate(self):
		"""
		>>
		> return : the scan output
		"""
		def recur(y_1, s1_pre, s2_pre):
			y_1_con = T.concatinate((y_1, ht_pre), axis = 1)
			
			# hidden layer-1
			zg_1 = T.nnet.sigmoid(T.dot(y_1_con, self.ZU1) + T.dot(s1_pre, self.ZW1) + self.ZB1)
			rg_1 = T.nnet.sigmoid(T.dot(y_1_con, self.RU1) + T.dot(s1_pre, self.RW1) + self.RB1)
			gg_1 = T.tanh(T.dot(y_1_con, self.GU1) + T.dot((rg_1 * s1_pre), self.GW1) + self.GB1)
			h_1 = ((1 - zg_1) * s1_pre) + (zg_1 * gg_1)
			
			# hidden layer-2
			y_2 = h_1
			zg_2 = T.nnet.sigmoid(T.dot(y_2, self.ZU2) + T.dot(s2_pre, self.ZW2) + self.ZB2)
			rg_2 = T.nnet.sigmoid(T.dot(y_2, self.RU2) + T.dot(s2_pre, self.RW2) + self.RB2)
			gg_2 = T.tanh(T.dot(y_2, self.GU2) + T.dot((rg_2 * s2_pre), self.GW2) + self.GB2)
			h_2 = ((1 - zg_2) * s2_pre) + (zg_2 * gg_2)
			
			return [h_1, h_2]
		[s1, s2], _ = theano.scan(fn = recur,
									sequences = self.inputs,
									outputs_info = [np.zeros(sdim), np.zeros(sdim)],
									truncate_gradient = self.bptt_trun)
		return [s1, s2]
		
	
