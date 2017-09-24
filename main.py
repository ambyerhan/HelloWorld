# ambyer uploaded on Sep. 2017
# ambyer
# 2017.06.09

import theano
import theano.tensor as T
import numpy as np
import argparse
import time
import os

from train_lm import train_func
			  


if __name__ == "__main__":
	# defining the arguments
	parser = argparse.ArgumentParser(description = 'train Neural Machine Translation System or predict...')
	parser.add_argument('--src', metavar = 'sfile', dest = 'src', default = None, help = 'source training set')
	parser.add_argument('--test', metavar = 'file', dest = 'tst', default = None, help = 'testing set')
	parser.add_argument('--hdim', metavar = 'H', dest = 'hdim', type = int, default = 12, help = 'size of hidden level')
	parser.add_argument('--lrate', metavar = 'lr', dest = 'lrate', type = float, default = 0.1, help = 'learning rate')
	parser.add_argument('--epochs', metavar = 'ep', dest = 'epochs', type = int, default = 3, help = 'size of epoch')
	parser.add_argument('--minibatch', metavar = 'mb', dest = 'minibatch', type = int, default = 3, help = 'size of minibatch')
	parser.add_argument('--embedding', metavar = 'em', dest = 'embedding', type = int, default = 8, help = 'size of embedding')
	parser.add_argument('--swvocab', metavar = 'swv', dest = 'swvocab', help = 'source lang word vocabulary')
	parser.add_argument('--mode', metavar = 'm', dest = 'mode', type = int, default = 0, help = 'mode: 0 means train and 1 means continue, 2 means exe')
	parser.add_argument('--model', metavar = 'md', dest = 'model', default = None, help = 'model: there should be a model that already trained')
	parser.add_argument('--save_nepoch', metavar = 'se', dest = 'save_nepoch', type = int, default = 1, help = 'save the model every n epoch(s)')
	parser.add_argument('--from_n', metavar = 'fn', dest = 'from_n', type = int, default = 1, help = 'start from the n(st/nd/rd/th) model')
	parser.add_argument('--threshold', metavar = 'thr', dest = 'thr', type = float, default = 0.0541, help = 'the threshold')
	parser.add_argument('--clip', metavar = 'clip', dest = 'clip', type = float, default = 5, help = 'the gradient clip')
	parser.add_argument('--method', metavar = 'method', dest = 'method', type = int, default = 0, help = 'the Optimum method of the deep learning [0:SGD, 1:ADADELTA, 2:ADAM]')
	parser.add_argument('--shuffle', metavar = 'shuff', dest = 'shuff', type = int, default = 0, help = 'whether shuffle the training set or not [0:No, 1:Yes]')
	parser.add_argument('--batch_n', metavar = 'batch_n', dest = 'batch_n', type = int, default = 2, help = 'indicates that extract how many batches at one time')
	args = parser.parse_args()
	
	train_func(args)
