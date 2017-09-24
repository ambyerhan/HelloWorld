# ambyer 
# 2017.06.08

# ambyer
# 2017.04.05
# train from zero : 	--src --swvocab
# train from n epoch : 	--src --swvocab --mode --model
# predicting : 			--test --swvocab --mode --model

import theano
import theano.tensor as T
import numpy as np
import argparse
import time
import os
import copy

from lm.utils import *
from lm.ErrorMsg import CErrorMsg
from lm.lm import LMModel
	

def train_func(args):
	# initializing
	srcfin 		= args.src
	tstfin		= args.tst
	sfdict		= args.swvocab
	hdim		= args.hdim
	lrate 		= args.lrate
	epochs		= args.epochs
	minibatch 	= args.minibatch
	embedding	= args.embedding
	mode 		= args.mode
	modelname 	= args.model
	savenep   	= args.save_nepoch
	fromn 		= args.from_n
	thr			= args.thr
	clip		= args.clip
	method		= args.method
	shuff 		= args.shuff
	batch_n		= args.batch_n
	svsize 		= 0
	# 
	svsize = read_vocab_json(sfdict)
	show_Info(args, svsize)
	# 
	lm = LMModel(embedding, hdim, svsize, minibatch, lrate, method, thr, clip)
	if mode == 0 or mode == 1: # train
		lm.build()
		if mode == 1: # from n model, and read the model
			print '[Read] Reading the model...'
			lm.readmodel(modelname)
			fromn = get_N(modelname) + 1
			print '[Read] Done, and begin the train from epoch <%d>' % fromn
		src_str_lines, src_n = read_set(srcfin)
		src_lines = strIdx2intIdx(src_str_lines) # change string index to int index
		total_time = []
		sent_n_bl = batch_n * minibatch # the sentence num in the batch list
		CErrorMsg.showErrExMsg((sent_n_bl <= src_n), 'The training set is little than n mini-batches!')
		fill_set(src_lines, src_n, sent_n_bl)
		CErrorMsg.showErrExMsg((len(src_lines) % sent_n_bl) == 0, 'The training sets failed to fill!')
		print '[Train] Begin the training...'
		for epoch in range(fromn - 1, epochs):
			e_beg = time.time()
			s_ln = []
			############################## shuffle the set ##############################
			CErrorMsg.showErrExMsg((shuff in [0, 1, 2]), 'The value of param shuff is unvalid!')
			if shuff == 1:
				print '[Shuff] Shuffling and sorting the training sets...'
				sh_beg = time.time()
				s_ln = shuf_sort(src_lines, sent_n_bl)
				sh_end = time.time()
				print '[Shuff] End shuffling and sorting, cost %.3f sec...' % (sh_end - sh_beg)
			elif shuff == 2:
				print '[Shuff] Shuffling the training set, but no sort...'
				sh_beg = time.time()
				s_ln = shuf_no_sort(src_lines, sent_n_bl)
				sh_end = time.time()
				print '[Shuff] Ending shuffling without sort, cost %.3f sec...' % (sh_end - sh_beg)
			else:
				s_ln = [s[:] for s in src_lines]
			print '[Epoch] Begining of the epoch %d' % (epoch + 1)
			
			############################## train the model ##############################
			b_n_to = 0 # batch num in total set
			total_win = len(s_ln) / sent_n_bl
			total_batch = len(s_ln) / minibatch
			for ii in range((total_win)): # every n batchs
				print '\n    <------------------------------------ Starting a window of batchs ------------------------------------>'
				win = s_ln[ii * sent_n_bl : (ii + 1) * sent_n_bl]
				for jj in range(batch_n): # every minibatch
					b_n_to += 1
					sub_t = win[jj * minibatch : (jj + 1) * minibatch]
					err = lm.train(sub_t)
					print '    >>[Cost] Epoch<%d / %d>::Batch<%d / %d>::{%d / %d} | {Win<%d / %d>, Batch<%d / %d>} >> Cost = %f' % ((epoch + 1), epochs, b_n_to, total_batch, (epoch + 1), b_n_to, ii + 1, total_win, jj + 1, batch_n, err)
			e_end = time.time()
			print '[Epoch] Ending of the epoch %d, and costs %.3f sec' % ((epoch + 1), (e_end - e_beg))
			total_time.append((e_end - e_beg))
			
			############################## saving the model ##############################
			if (epoch + 1) % savenep == 0:
				modelfile = './model/' + str(epoch + 1)
				if not os.path.exists(modelfile): # os.path.join('dir', 'sub_dir')
					os.makedirs(modelfile)
				lm.savemodel(epoch + 1, modelfile)
		print '[Train] End the training, costs %f sec totally, average %.3f sec per epoch ' % (sum(total_time), (sum(total_time) / len(total_time)))
	elif mode == 2: # predict
		frs = open(tstfin + ".rslt", "w")
		lm.build(False)
		lm.readmodel(modelname)
		model_num = get_N(modelname) + 1
		print '[Test]  Begin the testing...'
		tst_str_lines, tst_n = read_set(tstfin)
		tst_lines = strIdx2intIdx(tst_str_lines)

		total_score = []
		total_words = []
		for line in tst_lines:
			l = []
			l.append(line) # let the dimension increas
			rs, sum_s = lm.predict(l)
			############################## calculate the perplexity ##############################
			total_score.append(sum_s)
			total_words.append(len(l[0]))
			############################## save the every word's score ##############################
			ss = rs.tolist() # ss = [[score1], [score2], ..., [scoreN]]
			str_ss = ""
			for s in ss:
				str_ss += (str(s[0]) + ' ')
			print >> frs, str_ss[:-1]
		score = sum(total_score)
		words = sum(total_words)
		print '[PPL]   The perplexity is >> %.3f' % (np.exp(-1.0 * score / words))
		print '[Test]  End the testing...'
		frs.close()
	else:
		CErrorMsg.showErrMsg('The mode is wrong, please set value 0-2')
