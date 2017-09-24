# ambyer
# 2017.02.24
# params:
# argv[1] : the file that need to add tail

import sys
import time

fin = open(sys.argv[1], 'r')
fout = open(sys.argv[1] + '.sos', 'w')

print 'Processing...'
start = time.time()
for line in fin:
	if line == '\n':
		continue
	nline = '<sos> ' + line[0: len(line) - 1] + ' \n'
	fout.write(nline)
end = time.time()
print ('Done, used %fs totally...') % (end - start)

fin.close()
fout.close()
