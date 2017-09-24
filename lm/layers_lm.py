# ambyer
# 2017.06.08

import theano
import theano.tensor as T
import numpy as np
import numpy.random as rnd
from utils import getThreshold
from datetime import datetime

class CEmbeddingLayer(object):
	def __init__(self, edim, vsize, threshold = 0.1):
		"""
		>>
		> edim : the embedding dim
		> vsize : size of the vocabulary
		> threshold : the threshold
		"""
		self.edim = edim
		self.vsize = vsize
		self.threshold = getThreshold(vsize, edim)
		
		vocab = self.threshold * rnd.randn(vsize, edim).astype(theano.config.floatX)
		self.voc = theano.shared(value = vocab, name = 'vocab')
		self.params = [self.voc]
		
	def activate(self, inputs):
		"""
		>>
		> inputs : the indices of the sequences
		> return : return a 3D matrix that contains a batch-size sequences' wordembedding
		"""
		outputs = self.voc[inputs]
		return outputs
		
	def resetBatch(self, minibatch):
		"""
		>>
		> minibatch : the new minibatch
		"""
		self.minibatch = minibatch
		
	def savemodel(self, nep, dirname = '../model/'):
		"""
		>>
		> nep : the num of the epoch
		> dirname : the dir (have to cantain the '/' at the end of the dir)
		"""
		date = datetime.now().strftime('%Y.%m.%d')
		if dirname[-1] != '/':
			dirname += '/'
		filename = dirname + date + '-' + str(nep) + ('.voc.npz')
		print '    >>[Debug] saving the model %s.' % filename
		np.savez(filename, voc = self.voc.get_value())
		print '    >>[Debug] model is saved...'
		
	def readmodel(self, filename):
		"""
		>>
		> filename : the model's filename
		"""
		filename = filename + ('.voc.npz')
		print '    >>[Debug] reading the model from file : %s' % filename
		models = np.load(filename)
		self.voc.set_value(models['voc'])
		print '    >>[Debug] model is read...'
		

class CHiddenLayer(object):
	def __init__(self, idim, hdim, minibatch, layern, direct, threshold = 0.1, bptt_trun = -1):
		"""
		>>
		> idim : the input dim (from embedding or last hidden layer)
		> hdim : the hidden dim
		> minibatch : the minibatch
		> layern : which layer is this layer
		> direct : indicate the layer is for foreward or backward
		> threshold : the threshold
		> bptt_trun : used in scan
		"""
		self.idim = idim
		self.hdim = hdim
		self.minibatch = minibatch
		self.direct = direct
		self.layern = layern
		self.threshold = getThreshold(idim, hdim)
		self.bptt_trun = bptt_trun
		
		# gru z-gate vals
		ZU = getThreshold(idim, hdim) * rnd.randn(idim, hdim).astype(theano.config.floatX)
		ZW = getThreshold(hdim, hdim) * rnd.randn(hdim, hdim).astype(theano.config.floatX)
		ZB = 0. * rnd.randn(hdim).astype(theano.config.floatX)
		
		# gru r-gate vals
		RU = getThreshold(idim, hdim) * rnd.randn(idim, hdim).astype(theano.config.floatX)
		RW = getThreshold(hdim, hdim) * rnd.randn(hdim, hdim).astype(theano.config.floatX)
		RB = 0. * rnd.randn(hdim).astype(theano.config.floatX)
		
		# gru g-gate vals
		GU = getThreshold(idim, hdim) * rnd.randn(idim, hdim).astype(theano.config.floatX)
		GW = getThreshold(hdim, hdim) * rnd.randn(hdim, hdim).astype(theano.config.floatX)
		GB = 0. * rnd.randn(hdim).astype(theano.config.floatX)
		
		# set shared vals
		self.ZU = theano.shared(value = ZU, name = "ZU")
		self.ZW = theano.shared(value = ZW, name = "ZW")
		self.ZB = theano.shared(value = ZB, name = "ZB")
		self.RU = theano.shared(value = RU, name = "RU")
		self.RW = theano.shared(value = RW, name = "RW")
		self.RB = theano.shared(value = RB, name = "RB")
		self.GU = theano.shared(value = GU, name = "GU")
		self.GW = theano.shared(value = GW, name = "GW")
		self.GB = theano.shared(value = GB, name = "GB")
		
		# set self params
		self.params = [self.ZU, self.ZW, self.ZB,
						self.RU, self.RW, self.RB,
						self.GU, self.GW, self.GB]
						
	def activate(self, inputs):
		"""
		>>
		> inputs : the inputs from seq
		> return : return the scan output
		"""
		self.inputs = inputs
		def recur(x, h_pre):
			zg = T.nnet.sigmoid(T.dot(x, self.ZU) + T.dot(h_pre, self.ZW) + self.ZB)
			rg = T.nnet.sigmoid(T.dot(x, self.RU) + T.dot(h_pre, self.RW) + self.RB)
			gg = T.tanh(T.dot(x, self.GU) + T.dot((rg * h_pre), self.GW) + self.GB)
			h = ((1 - zg) * h_pre) + (zg * gg)
			return h
		hrslt, _ = theano.scan(fn = recur,
								sequences = self.inputs,
								outputs_info = [T.zeros((self.minibatch, self.hdim))],
								truncate_gradient = self.bptt_trun)
		return hrslt
		
	def resetBatch(self, minibatch):
		"""
		>>
		> minibatch : the new minibatch
		"""
		self.minibatch = minibatch
		
	def savemodel(self, nep, dirname = '../model/'):
		"""
		>>
		> nep : the num of the epoch
		> nh : indicate which hidden layer
		> dirname : name of the dir
		"""
		date = datetime.now().strftime('%Y.%m.%d')
		if dirname[-1] != '/':
			dirname += '/'
		filename = dirname + date + '-' + str(nep) + '.hid' + self.layern + self.direct + '.npz'
		print '    >>[Debug] saving the model %s.' % filename
		np.savez(filename, ZU = self.ZU.get_value(),
							ZW = self.ZW.get_value(),
							ZB = self.ZB.get_value(),
							RU = self.RU.get_value(),
							RW = self.RW.get_value(),
							RB = self.RB.get_value(),
							GU = self.GU.get_value(),
							GW = self.GW.get_value(),
							GB = self.GB.get_value())
		print '    >>[Debug] model is saved...'
							
	def readmodel(self, filename):
		"""
		>>
		> filename : the model's name
		> nh : which hidden layer
		"""
		filename = filename + '.hid' + self.layern + self.direct + '.npz'
		print '    >>[Debug] reading the model from file : %s' % filename
		models = np.load(filename)
		self.ZU.set_value(models['ZU'])
		self.ZW.set_value(models['ZW'])
		self.ZB.set_value(models['ZB'])
		self.RU.set_value(models['RU'])
		self.RW.set_value(models['RW'])
		self.RB.set_value(models['RB'])
		self.GU.set_value(models['GU'])
		self.GW.set_value(models['GW'])
		self.GB.set_value(models['GB'])
		print '    >>[Debug] model is read...'
		

class COutputLayer(object):
	def __init__(self, idim, vsize, minibatch, threshold = 0.1):
		"""
		>>
		> idim : the pre-layer's dim
		> vsize : the size of the vocabulary
		> minibatch : the minibatch
		> threshold : the threshold
		"""
		self.idim = idim
		self.vsize = vsize
		self.minibatch = minibatch
		self.threshold = 0.
		
		HO = self.threshold * rnd.randn(idim, vsize).astype(theano.config.floatX)
		BO = self.threshold * rnd.randn(vsize).astype(theano.config.floatX)
		self.HO = theano.shared(value = HO, name = "HO")
		self.BO = theano.shared(value = BO, name = "BO")
		self.params = [self.HO, self.BO]
		
	def activate(self, inputs):
		"""
		>>
		> inputs : the inputs should be 3D matrix(usually is shaped <mxlen, minibatch, 2 * hdim>), and contains all timesteps' hidden2 output
		"""
		tmp = T.dot(inputs, self.HO) + self.BO
		out = -T.log(T.nnet.softmax(tmp.reshape((tmp.shape[0] * tmp.shape[1], tmp.shape[2])))) # when calculate the softmax, the matrix must be 2D or 1D
		return (out.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2])))
		
	def resetBatch(self, minibatch):
		"""
		>>
		> minibatch : the new minibatch
		"""
		self.minibatch = minibatch
		
	def savemodel(self, nep, dirname = '../model/'):
		"""
		>>
		> nep : the num of the epoch
		> dirname : name of the dir
		"""
		date = datetime.now().strftime('%Y.%m.%d')
		if dirname[-1] != '/':
			dirname += '/'
		filename = dirname + date + '-' + str(nep) + '.out.npz'
		print '    >>[Debug] saving the model %s.' % filename
		np.savez(filename, HO = self.HO.get_value(), BO = self.BO.get_value())
		print '    >>[Debug] model is saved...'
		
	def readmodel(self, filename):
		"""
		>>
		> filename : the model's filename
		"""
		filename = filename + '.out.npz'
		print '    >>[Debug] reading the model from file : %s' % filename
		models = np.load(filename)
		self.HO.set_value(models['HO'])
		self.BO.set_value(models['BO'])
		print '    >>[Debug] model are read...'
		

