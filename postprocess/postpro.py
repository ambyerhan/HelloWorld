# ambyer
# 2017.06.09

import sys
import math
def pospro(fp):
	"""
	>>
	> fp : file pointer
	"""
	total_w = 0 # total word num
	total_s = 0 # total score
	for ln in fp:
		scores = ln.strip().split(' ')
		total_w += len(scores)
		for score in scores:
			total_s += float(score)
	
	ave = total_s / total_w
	ppl = math.exp(-1 * ave)
	print 'the perplexity = %.4f' % ppl


if __name__ == "__main__":
	fin = open(sys.argv[1], 'r')
	#fot = open(sys.argv[1] + ".post", 'w')
	pospro(fin)
