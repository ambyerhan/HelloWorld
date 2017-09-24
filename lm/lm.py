# ambyer
# 2017.06.08

import theano
import theano.tensor as T
import numpy as np

from layers_lm import CEmbeddingLayer, CHiddenLayer, COutputLayer
from utils import *
from ErrorMsg import CErrorMsg

class LMModel(object):
	def __init__(self, edim, hdim, vsize, minibatch, lrate = 0.1, method = 0, threshold = 0.1, clip = 5):
		"""
		>>
		> edim : the embedding dim
		> hdim : the hidden dim
		> vsize : the size of vocabulary
		> minibatch : the batch size
		> lrate : the learning rate
		> method : the optimize methods
		> threshold : the threshold
		> clip : used in gredient clip
		"""
		self.edim = edim
		self.hdim = hdim
		self.vsize = vsize
		self.minibatch = minibatch
		self.lrate = lrate
		self.method = method
		self.threshold = threshold
		self.clip = clip
		
		self.embedding = CEmbeddingLayer(self.edim, self.vsize, self.threshold)

		self.hidden1_foreward = CHiddenLayer(self.edim, self.hdim, self.minibatch, '1', 'foreward', self.threshold)

		self.output = COutputLayer(self.hdim, self.vsize, self.minibatch, self.threshold)
		
		# all params
		self.params = []
		self.params.extend(self.embedding.params)

		self.params.extend(self.hidden1_foreward.params)

		self.params.extend(self.output.params)
		
		if self.method == 1:
			# init Eg2 and Ex2
			self.rho = 0.95
			self.epsilon = 1e-6
			self.Eg2 = [theano.shared(np.zeros(param.get_value(borrow = True).shape, dtype = theano.config.floatX)) for param in self.params]
			self.Ex2 = [theano.shared(np.zeros(param.get_value(borrow = True).shape, dtype = theano.config.floatX)) for param in self.params]
			
		if self.method == 2:
			# initialize  constant
			self.alpha = 0.001
			self.beta1 = 0.9
			self.beta2 = 0.999
			self.epsilon = 1e-8
			self.ts = 0.
			
			# initialize tensor shared variable
			self.mt = [theano.shared(np.zeros(param.get_value(borrow = True).shape, dtype = theano.config.floatX)) for param in self.params]
			self.vt = [theano.shared(np.zeros(param.get_value(borrow = True).shape, dtype = theano.config.floatX)) for param in self.params]
		
	def __gradient(self):
		"""
		>>clip the gradients
		"""
		if self.clip < 0:
			return [T.grad(self.cost, param) for param in self.params]
		elif self.clip == 0:
			grads = [T.grad(self.cost, param) for param in self.params]
			return [(3 * (grad / (1. + T.sqrt((grad ** 2).sum())))) for grad in grads]
		else:
			return theano.grad(theano.gradient.grad_clip(self.cost, -1.0 * self.clip, self.clip), self.params)
	
	def __concatenate(self, h, rh, lastpos):
		"""
		>>
		> h : the orignal matrix or vector
		> rh : the reversed matrix or vector
		> lastpos : the positions that refers to last legal elem
		"""
		def reverse(rh, pos):
			if pos == -1:
				return rh[:][::-1]
			else:
				return T.concatenate(((rh[:pos + 1])[::-1], rh[pos + 1:]), axis = 0)
		nrh, _ = theano.scan(fn = reverse,
							sequences = [rh.dimshuffle(1, 0, 2), lastpos],
							outputs_info = None)
		nrh = nrh.dimshuffle(1, 0, 2)
		return (T.concatenate((h, nrh), axis = 2))
	
	def __SGD(self):
		"""
		>>
		"""
		grads = theano.grad(theano.gradient.grad_clip(self.cost, -1.0 * self.clip, self.clip), self.params)
		update = [(param, param + self.lrate * grad) for param, grad in zip(self.params, grads)]
		return update
		
	def __Adadelta(self):
		"""
		>>
		"""
		# calculate grads
		grads = theano.grad(theano.gradient.grad_clip(self.cost, -1.0 * self.clip, self.clip), self.params)

		# update g2 first
		Eg2_update = [(g2, self.rho * g2 + (1.0 - self.rho) * (g ** 2)) for g2, g in zip(self.Eg2, grads)]
		
		# calculate the delta_x by RMS [RMS(x) = sqrt(x + epsilon)]
		delta_x = [-1.0 * (T.sqrt(x2_last + self.epsilon) / T.sqrt(g2_now[1] + self.epsilon)) * g for x2_last, g2_now, g in zip(self.Ex2, Eg2_update, grads)]
		
		# update Ex2 and params
		Ex2_update = [(x2, self.rho * x2 + (1.0 - self.rho) * (x ** 2)) for  x2, x in zip(self.Ex2, delta_x)] # the delta_x's each elem in for cannot be same in two for(or else, there is a error[the length is not known]), here i use name 'x' and 'delta'.
		delta_x_update = [(param, param - delta) for param, delta in zip(self.params, delta_x)]
		return Eg2_update + Ex2_update + delta_x_update

	def __Adam(self):
		"""
		>>
		"""
		# update timestep
		self.ts += 1.

		# calculate grads
		grads = theano.grad(theano.gradient.grad_clip(self.cost, -1.0 * self.clip, self.clip), self.params)

		# update m and v
		mt_upd = [(m, self.beta1 * m + (1. - self.beta1) * grad) for m, grad in zip(self.mt, grads)]
		vt_upd = [(v, self.beta2 * v + (1. - self.beta2) * (grad ** 2)) for v, grad in zip(self.vt, grads)]
		
		# calc mt^ and vt^ (here we note mt^ and vt^ as mt_ and vt_)
		mt_ = [m[1] / (1. - (self.beta1 ** self.ts)) for m in mt_upd]
		vt_ = [v[1] / (1. - (self.beta2 ** self.ts)) for v in vt_upd]
		
		# update params
		pa_upd = [(param, param + self.alpha * (m_ / (T.sqrt(v_) + self.epsilon))) for param, m_, v_ in zip(self.params, mt_, vt_)]
		
		return mt_upd + vt_upd + pa_upd

	def calcCost(self, y, y_gold, oMask):
		"""
		>>
		> y : the output of the output layer
		> y_gold : the expected value of the tgt
		> oMask : the mask of the output, and shaped <maxlen, minibath>
		"""
		cost = T.sum(y[T.arange(y.shape[0]), y_gold] * oMask) # y.shape[0] is minibatch, and y is a 2D matrix
		return cost

	def calcCostPre(self, y, y_gold):
		cost = y[y_gold]
		return cost

	def build(self, isTrain = True):
		"""
		>>
		> isTrain : build for train or predict
		"""
		if isTrain:
			x = T.imatrix('x')
			y_gold = T.imatrix('y_gold')
			#hm = T.tensor3('hm')
			hm = T.imatrix('hm')
			om = T.imatrix('om')
			
			# the l2r direction
			em_out = self.embedding.activate(x.T)
			h1_out = self.hidden1_foreward.activate(em_out)

			h_out = h1_out * hm[:, :, None]
			o_out = self.output.activate(h_out)

			#tmp_cost, _ = theano.scan(fn = self.calcCost,
			#						  sequences = [o_out, y_gold, om])
			y_gold_flat = y_gold.flatten()
			y_gold_flat_idx = T.arange(y_gold_flat.shape[0]) * self.vsize + y_gold_flat
			o_out_flat = o_out.flatten()
			cost = o_out_flat[y_gold_flat_idx]
			cost = cost.reshape((y_gold.shape[0], y_gold.shape[1]))

			self.cost = T.sum(cost * om) / T.sum(om)

			if self.method == 0:
				param_update = self.__SGD()
			elif self.method == 1:
				param_update = self.__Adadelta()
			else:
				param_update = self.__Adam()
			
			self.__train = theano.function(inputs = [x, y_gold, hm, om],
											outputs = self.cost,
											updates = param_update,
											allow_input_downcast = True)
		else:
			self.resetBatch(1)
			x = T.ivector('x')
			#y_gold = T.ivector('y_gold')
			om = T.tensor3('om')
			
			# the l2r direction
			em_out = self.embedding.activate(x)
			h1_out = self.hidden1_foreward.activate(em_out)
			
			h_out = h1_out
			tmp_o_out = self.output.activate(h_out)
			o_out = tmp_o_out * om

			scores = T.sum(o_out, axis = 2) # the scores should be a vector that each elem of it is a score of a word(so axis is 1)
			sum_s = T.sum(o_out)
			# though socres's batch is 1, it still a 3D matrix shaped (mxlen, minibatch, vsize)
			self.__predict = theano.function(inputs = [x, om],
											 outputs = [scores, sum_s],
											 allow_input_downcast = True)
			
	def train(self, src_seq):
		"""
		>>
		> src_seq : the sequences
		"""
		sindex = src_seq
		rindex = get_rindex(src_seq)
		mxlen, lastpos = get_maxlen(sindex, True)
		set_pad(sindex, mxlen, rindex)
		#hMask = get_Hout_3DMask(sindex, self.minibatch, self.hdim, mxlen)
		hMask = get_Hout_2DMask(sindex, self.minibatch, mxlen)
		oMask, tgtIdx = get_Oout_2DMask(sindex, self.minibatch, lastpos)
		
		return (self.__train(sindex, tgtIdx.T, hMask.T, oMask.T))
		
	def predict(self, src_seq):
		"""
		>>
		> src_seq : the sequences
		"""
		sindex = src_seq
		rindex = get_rindex(src_seq)
		mxlen, lastpos = get_maxlen(sindex, True)
		set_pad(sindex, mxlen, rindex)
		oMask = get_Oout_3DMask(sindex, 1, self.vsize, mxlen)
		
		return (self.__predict(sindex[0], oMask))
		
	def resetBatch(self, minibatch):
		"""
		>>
		> minibatch : the new minibatch
		"""
		self.embedding.resetBatch(minibatch)
		self.hidden1_foreward.resetBatch(minibatch)
		self.output.resetBatch(minibatch)
		
	def savemodel(self, nep, dirname = '../mode/'):
		"""
		>>
		> nep : the num of the epoch
		> dirname : the model's dir-name that the last elem is '/'
		"""
		print '[Debug] saving the models of epoch<%d>...' % nep
		self.embedding.savemodel(nep, dirname)
		self.hidden1_foreward.savemodel(nep, dirname)
		self.output.savemodel(nep, dirname)
		print '[Debug] models are saved...'
		
	def readmodel(self, filename):
		"""
		>>
		> filename : the model's name
		"""
		print '[Debug] reading the models from file : %s.*' % filename
		self.embedding.readmodel(filename)
		self.hidden1_foreward.readmodel(filename)
		self.output.readmodel(filename)
		print '[Debug] models are read...'
		
