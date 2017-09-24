#
#ambyer
#

import sys
import json
from collections import OrderedDict


def build(cutoff = 2):
   fin = open(sys.argv[1], 'r')
   #fout = open(sys.argv[1] + '.norder', 'w')
   forder = open(sys.argv[1] + '.dict', 'w')
   
   wdict = {}
   odict = OrderedDict()

   print 'building dictionaries...'
   for line in fin:
       words = line.strip().split(' ')
       for word in words:
           if wdict.has_key(word):
	        wdict[word] += 1
           else:
	        wdict[word] = 1

   tmpdict = sorted(wdict.iteritems(), key = lambda d:d[1], reverse = True) 

   odict['<sos>'] = 0
   odict['<eos>'] = 1
   odict['<unk>'] = 2
   odict['<pad>'] = 3
   i = 4
   for k, v in enumerate(tmpdict):##tmpdict={0:['key', 'value'], ...}
       if i > 30002:
	  print 'the maximum...'
	  break
       if v[1] > cutoff:
          odict[v[0]] = i
          i += 1
       #else:
          # print v[0], v[1]

   #odict = sorted(odict.iteritems(), key = lambda d:d[1], reverse = True)

   print 'storing...'
   #json.dump(wdict, fout, indent = 2, ensure_ascii = False)
   json.dump(odict, forder, indent = 2, ensure_ascii = False)
   print 'done...'
   fin.close()
   forder.close()

if __name__ == "__main__":
   build()

