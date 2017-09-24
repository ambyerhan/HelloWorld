# ambyer
# 2017.03.27

import numpy as np
import sys
import theano
import theano.tensor as T
import copy
import json
import random

def show_Info(args, s):
	"""
	>>show the info of the configer
	> args : the argparse, is a model of python
	> s : the size of source vacabulary
	"""
	print "|------------------------------------------------------|"
	print "|                    Configuration                     |"
	print "|------------------------------------------------------|"
	
	print "|--%-10s = %-7d[0:train, 1:continue, 2:predict]" % ('mode', args.mode)
	print "|--%-10s = %-7d[the hidden dimensionality]" % ('hdim', args.hdim)
	print "|--%-10s = %-7d[the dimensionality of the word representation]" % ('edim', args.embedding)
	print "|------------------------------------------------------|"
	if args.mode == 0 or args.mode == 1:
		print "|--%-10s = %-7d[the size of minibatch]" % ('minibatch', args.minibatch)
		print "|--%-10s = %-7d[how many batchs will be extracted at one time]" % ('batch_win', args.batch_n)
		print "|--%-10s = %-7d[the training epoch]" % ('epoch', args.epochs)
		print "|--%-10s = %-7.3f[used in parameters' initialization]" % ('threshold', args.thr)
		print "|--%-10s = %-7.3f[used in gradient clip]" % ('clip', args.clip)
		print "|--%-10s = %-7.3f[the learning rate]" % ('lrate', args.lrate)
		print "|--%-10s = %-7d[the Optimum method [0:SGD, 1:ADADELTA, 2:ADAM]]" % ('method', args.method)
		print "|--%-10s = %-7d[whether shuffle the training set or not [0:No, 1:Yes, 2:Shuffle but not Sort]]" % ('shuffle', args.shuff)
		print "|--%-10s = %-7d[save a model every n epoch]" % ('savenep', args.save_nepoch)
		print "|--%-10s = %-7d[size of source vocabulary]" % ('ssize', s)
		if args.mode == 1:
			print "|--%-10s = %-7d[continue to train the model from n-th model]" % ('from_n', args.from_n)
			print "|--%-10s = %s [the model that start from]" % ('model', args.model)
	else:
		print "|--%-10s = %s [the set that need to be predicted]" % ('tstfile', args.tst)
	print "|------------------------------------------------------|"
	print "|--%-10s = %s [source training set]" % ('srcfile', args.src)
	print "|--%-10s = %s [source lang vocabulary]" % ('swvocab', args.swvocab)
	print "|------------------------------------------------------|"
	print "|                 Configuration  End                   |"
	print "|------------------------------------------------------|"
	#exit()

def getThreshold(idim, odim):
	return np.sqrt(6. / (idim + odim))

def strIdx2intIdx(lines):
	"""
	>>
	>lines : the index matrix that typed string
	"""
	idx = []
	for ln in lines:
		units = ln.strip().split(' ')
		tmp_r = []
		for unit in units:
			if unit == '\n' or unit == '':
				print 'wrongs'
				continue
			tmp_r.append(int(unit))
		idx.append(tmp_r)
	return idx

def get_rindex(sub_arr):
	"""
	>>
	> sub_arr : a mini-batch sentences that typed int
	> return : the indexes and rindexes
	"""
	rindexes = []
	for arr in sub_arr:
		rindex = []
		rindex = arr[: : -1]
		rindexes.append(rindex)
	return rindexes

def get_maxlen(sub_arr, isSrc = True):
	"""
	>>
	> sub_arr : a mini-batch sentences that typed int
	> isSrc : if true then return the last_pos, else just return maxlen
	> return : the maxlen in a mini-batch sentences and the vector contains the last used word's pos
	"""
	sent_lens = [len(sent) for sent in sub_arr]
	maxlen = max(length for length in sent_lens)
	last_pos = [(length - (maxlen + 1)) for length in sent_lens]
	if isSrc:
		return [maxlen, last_pos]
	else:
		return maxlen

def get_N(modelname):
	"""
	>>get the model's number, that refers to the model is epoch n's model
	> modelname : the name of the model, it contains the number of the epoch
	"""
	ii = modelname.rfind('-')
	nn = int(modelname[ii + 1:])
	return nn

def set_pad(indexes, maxlen, rindexes = None):
	"""
	>>
	> sub_arr : a mini-batch sentences that typed int
	> maxlen : the longest sentence's len in a mini-batch sentences
	> return : the indexes and rindexes that already add the <pad> tag
	"""
	for ii in range(len(indexes)):
		if len(indexes[ii]) < maxlen:
			for jj in range(maxlen - len(indexes[ii])):
				indexes[ii].append(3)
				if rindexes != None:
					rindexes[ii].append(3)

def set_tgt_pad(indices, maxlen):
	"""
	>>
	> indices : the tgt indices seq
	> maxlen : the max-length of the tgt seq
	"""
	for ii in range(len(indices)):
		if len(indices[ii]) < maxlen:
			indices[ii].append(1)
			for jj in range(maxlen - len(indices) - 1):
				indices[ii].append(3)

def get_Hout_2DMask(ein, minibatch, maxlen):
	"""
	>>
	> ein : the src inputs, is a 2d matrix
	> minibatch : batch size
	> maxlen : the max seq_len
	"""
	"""
	_2DMask = np.zeros((maxlen, minibatch)) # ein.shape <minibatch, maxlen>
	for ii in range(minibatch):
		for jj in range(maxlen):
			if ein[ii][jj] == 1:
				break
			_2DMask[jj][ii] = 1
	return _2DMask
	"""
	_2DMask = np.not_equal(ein, 3).astype("int32")
	return _2DMask

def get_Hout_3DMask(ein, minibatch, hdim, maxlen):
	"""
	>>
	> ein : the src inputs, is a 2d matrix
	> minibatch : batch size
	> hdim : size of hidden layer
	> maxlen : the max seq_len
	"""
	_3DMask = np.zeros((maxlen, minibatch, hdim))
	for ii in range(minibatch):
		for jj in range(maxlen):
			if ein[ii][jj] == 3:
				break
			_3DMask[jj][ii][:] = 1
	return _3DMask

def get_Oout_2DMask(y, minibatch, lp):
	"""
	>>
	> y : the target seqs
	"""
	con = np.ones((minibatch, 1), dtype = np.int) * 3
	ty = np.array(y)
	ty = ty[:, 1:]
	ty = np.concatenate((ty, con), axis = 1)
	ty[np.arange(minibatch), lp] = 1 # let the sentences add <eos> tag by using lastpos
	_2DMask = np.not_equal(ty, 3).astype("int32")
	return _2DMask, ty

def get_Oout_3DMask(y, minibatch, tvsize, maxlen):
	"""
	>>
	> y : the tgt sentences' indexes
	> minibatch : the minibatch
	> tvsize : size of the tgt vocabulary
	> maxlen : the longest sentence's len in a mini-batch sentences
	> return : get the 3DMask that used in loss function
	"""
	con = np.ones((minibatch, 1), dtype = np.int) * 3
	ty = np.array(y)
	ty = ty[:, 1:]
	ty = np.concatenate((ty, con), axis = 1)
	_3DMask = np.zeros((maxlen, minibatch, tvsize))
	for ii in range(minibatch):
		firstPadFlag = True
		for jj in range(maxlen):
			if ty[ii][jj] == 3:
				if firstPadFlag: # if meet the first <pad>, we have to change it to <eos>
					ty[ii][jj] = 1
					firstPadFlag = False
				else:
					break
			_3DMask[jj][ii][ty[ii][jj]] = 1
	return _3DMask

def read_set(filename):
	"""
	>>read the set from the file
	> filename : the filename or filedir+filename
	> return : return a list that contains all lines and a num that indecate the num of the lines
	"""
	lines = []
	try:
		with open(filename, "r") as f:
			lines = f.readlines()
			line_n = len(lines)
			f.close()
	except:
		print '[Error] Cannot open the file %s, No such file or directory!' % filename
		exit()
	return [lines, line_n]

def read_vocab_json(filename):
	"""
	>>read vocabulary from the file
	> filename : the file name
	> return the num of the vocab
	"""
	v_n = 0
	try:
		with open(filename, "r") as f:
			dic = json.load(f)
			v_n = len(dic)
			f.close()
	except:
		print '[Error] Cannot open the file %s, No such file or directory!' % filename
		exit()
	return v_n

def shuffle(src, tgt, length):
	"""
	>>
	> src : the source sentences' indices, and type is 'list'
	> tgt : the target sentences' indices, and type is 'list'
	> length : the len of the src and tgt
	"""
	beg = 0
	for ii in range(beg, length):
		r = random.randint(beg, length - 1)
		src[ii], src[r] = src[r], src[ii]
		tgt[ii], tgt[r] = tgt[r], tgt[ii]
		beg += 1

def shuffle_lst(lst, length):
	"""
	>>
	> lst : the list that need to be shuffled
	> length : the len of the lst
	"""
	beg = 0
	for ii in range(beg, length):
		r = random.randint(beg, length - 1)
		lst[ii], lst[r] = lst[r], lst[ii]
		beg += 1
		
def shuf_sort(src, win_len):
	"""
	>>
	> src : the source sentences
	> batch_n : indicate how many batch in a window
	> minibatch : the batch num
	"""
	# make a shuffled list and zipped with sentence lens
	src_n = len(src)
	permu = np.random.permutation(src_n)
	src_zip = [(permu[i], len(src[permu[i]])) for i in range(src_n)]
	if src_n % win_len > 0:
		src_rest_zip = [(j, len(src[j])) for j in range(win_len - (src_n % win_len))] # the rest
		src_zip.extend(src_rest_zip)
	assert(((len(src_zip) % win_len) == 0))
	# sort list within a batch-window
	rslt = []
	for i in range((len(src_zip) / win_len)):
		segment = src_zip[i * win_len : (i + 1) * win_len]
		rslt = rslt + sorted(segment, key = lambda s : s[1])
	
	del permu
	del src_zip
	
	final = [src[r[0]][:] for r in rslt]
	return final

def shuf_no_sort(src, win_len):
	"""
	>>
	> src : the source sentences
	> batch_n : indicate how many batch in a window
	> minibatch : the batch num
	"""
	# make a shuffled list and zipped with sentence lens
	src_n = len(src)
	permu = np.random.permutation(src_n)
	src_zip = [(permu[i], len(src[permu[i]])) for i in range(src_n)]
	if src_n % win_len > 0:
		src_rest_zip = [(j, len(src[j])) for j in range(win_len - (src_n % win_len))]  # the rest
		src_zip.extend(src_rest_zip)
	assert (((len(src_zip) % win_len) == 0))
	# sort list within a batch-window
	rslt = []
	for i in range((len(src_zip) / win_len)):
		segment = src_zip[i * win_len: (i + 1) * win_len]
		rslt = rslt + segment

	del permu
	del src_zip

	final = [src[r[0]][:] for r in rslt]
	return final

def no_shuf_sort(src, src_n, win_len):
	"""
	>>
	> src : the source sentences
	> src_n : the num of the source training set
	> batch_n : indicate how many batch in a window
	> minibatch : the batch num
	"""
	if src_n % win_len > 0:
		tmp = src[0 : win_len - (src_n % win_len)]
		src.extend(tmp)
	return src

def quickSort(src, tgt, low, high):
	"""
	>>
	> src : the src matrix
	> tgt : the tgt matrix
	> low : the pointer that point to lower position
	> hight : the pointer that point to higher position
	"""
	i = low
	j = high
	if i >= j:
		return src, tgt
	key = max(len(src[i]), len(tgt[i]))
	stmp = src[i]
	ttmp = tgt[i]
	while i < j:
		while i < j and max(len(src[j]), len(tgt[j])) >= key:
			j -= 1
		src[i] = src[j]
		tgt[i] = tgt[j]
		while i < j and max(len(src[i]), len(tgt[i])) <= key:
			i += 1
		src[j] = src[i]
		tgt[j] = tgt[i]
	src[i] = stmp
	tgt[i] = ttmp
	
	quickSort(src, tgt, low, i - 1)
	quickSort(src, tgt, j + 1, high)
	return src, tgt

def quickSort_single(src, low, high):
	"""
	>>
	> src : the src matrix
	> low : the pointer that point to lower position
	> hight : the pointer that point to higher position
	"""
	i = low
	j = high
	if i >= j:
		return src
	key = len(src[i])
	stmp = src[i]
	while i < j:
		while i < j and len(src[j]) >= key:
			j -= 1
		src[i] = src[j]
		while i < j and len(src[i]) <= key:
			i += 1
		src[j] = src[i]
	src[i] = stmp
	
	quickSort_single(src, low, i - 1)
	quickSort_single(src, j + 1, high)
	return src

def sort(src):
	"""
	>>
	> src : the source sentences
	"""
	permu = np.random.permutation(len(src))
	zip_length = [(permu[i], len(src[permu[i]])) for i in range(len(src))]
	
	n_src = []
	nn = sorted(zip_length, key = lambda s:s[1])
	n_src = [src[ii[0]] for ii in nn]
	
	del permu
	del zip_length
	
	return n_src

def fill_set(src, src_n, win_len):
	"""
	>>
	> src : the source set
	> src_n : the size of set
	> win_len : indicate how many sentences in a window (i.e. win_len = minibatch * n_batch)
	"""
	if src_n % win_len > 0:
		stmp = [src[s] for s in range(win_len - (src_n % win_len))]
		src.extend(stmp)
