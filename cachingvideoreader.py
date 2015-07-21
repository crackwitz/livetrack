from __future__ import division
import numpy as np
import cv2
from multiprocessing.pool import ThreadPool
from collections import deque

class RateChangedVideo(object):
	"reads the last of every n frames"
	
	def __init__(self, vid, decimate=1):
		self.vid = vid
		self.decimate = decimate
	
	def grab(self):
		for i in xrange(self.decimate):
			rv = self.vid.grab()
			if not rv: return False

		return True
	
	def retrieve(self):
		(rv, frame) = self.vid.retrieve()
		return (rv, frame)

	def read(self):
		rv = self.grab()
		if not rv: return (False, None)
		
		(rv, frame) = self.retrieve()
		return (rv, frame)
	
	def seek(self, pos):
		pos = (int(pos)+1) * self.decimate - 1
		self.vid.set(cv2.CAP_PROP_POS_FRAMES, pos)
	
	def tell(self):
		pos = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
		pos //= self.decimate
		return pos


class CachingVideoReader(object):
	def __init__(self):
		pass
