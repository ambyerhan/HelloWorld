# ambyer
# 2017.05.22

import sys
import inspect

class CErrorMsg(object):
	def __init__(self):
		self.frame = inspect.currentframe()
		
	@staticmethod
	def showErrExMsg(ex = False, emsg = 'Err msg not define yet!'):
		if not ex:
			print '[Error] An error ocurred in File : %s, line %d:' % (__file__, self.frame.f_lineno)
			print '\t[ErrMsg] %s' % (emsg)
			exit()
			
	@staticmethod
	def showErrMsg(emsg = 'Err msg not define yet!'):
		print '[Error] %s' % emsg
		exit()
