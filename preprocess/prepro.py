# ambyer 
# 2017.02.24
# 
# params:
#	argv[1] : the src file
#	argv[2] : the vocabulary
#

import sys
import json
import re
import time
import codecs

fin = codecs.open(sys.argv[1], 'rb', 'utf-8')
fv = open(sys.argv[2], 'rb')
fout = open(sys.argv[1] + '.out', 'w')

print 'Loading vocabulary...', fv
dict = json.load(fv)
print 'Done...'

print 'Processing...'
start = time.time()

for line in fin:
	nline = re.sub('\n', '', line) 
	words = nline.strip().split(' ')
	newline = ''
	for word in words:
		if dict.has_key(word):
			#print '>>>', word
			newline += str(dict[word]) + ' '
		else:
			newline += str(dict['<unk>']) + ' '
	fout.write(newline[0: len(newline) - 1] + '\n') # delete the last space ' '

end = time.time()
print ('Done, use %fs totally...') % (end - start)

fin.close()
fv.close()
fout.close()
