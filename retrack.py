from __future__ import division
import os
import sys
import time
import numpy as np
import cv2
import json
import pprint; pp = pprint.pprint

########################################################################
# http://code.activestate.com/recipes/577231-discrete-pid-controller/

class PID:
	"Discrete PID control"
	def __init__(self, P=2.0, I=0.0, D=1.0, Derivator=0, Integrator=0, Integrator_max=500, Integrator_min=-500):
		self.Kp=P
		self.Ki=I
		self.Kd=D
		self.Derivator=Derivator
		self.Integrator=Integrator
		self.Integrator_max=Integrator_max
		self.Integrator_min=Integrator_min

		self.set_point=0.0
		self.error=0.0

	def update(self,current_value):
		"""
		Calculate PID output value for given reference input and feedback
		"""

		self.error = self.set_point - current_value

		self.P_value = self.Kp * self.error
		self.D_value = self.Kd * ( self.error - self.Derivator)
		self.Derivator = self.error

		self.Integrator = self.Integrator + self.error

		if self.Integrator > self.Integrator_max:
			self.Integrator = self.Integrator_max
		elif self.Integrator < self.Integrator_min:
			self.Integrator = self.Integrator_min

		self.I_value = self.Integrator * self.Ki

		PID = self.P_value + self.I_value + self.D_value

		return PID

########################################################################


def getslice(frame, anchor):
	return None
	
	if anchor is not None:
		(x,y) = anchor
	else:
		y = iround(frame.shape[0] * 0.333)
	#return (frame.sum(axis=0) / frame.shape[1]).astype(np.uint8)
	return frame[y]

class VideoSource(object):
	def __init__(self, vid, numcache=100, numstep=25):
		self.vid = vid
		self.index = -1 # just for relative addressing
		self.numcache = numcache
		self.numstep = numstep
		self.cache = {} # index -> frame
		#self.stripes = {} # index -> row
		self.mru = [] # oldest -> newest
	
	def cache_range(self, start, stop):
		requested = range(start, stop)
		requested = [i for i in requested if i not in self.cache]
		if not requested: return
		start = min(requested)
		stop = max(requested)+1
		requested = range(start, stop)
		self.vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, start)
		for i in requested:
			if i in self.cache:
				rv = self.vid.grab()
			else:
				(rv, frame) = self.vid.read()
				if rv:
					self.cache[i] = frame
		
		self.mru = [i for i in self.mru if i not in requested] + requested

	def _prefetch(self, newindex):
		rel = newindex - self.index

		if rel < 0:
			do_prefetch = not all(i in self.cache for i in xrange(newindex-1, newindex+1))
		
			if do_prefetch:
				upcoming = range(newindex-self.numstep, newindex+1)
				print "prefetching"
				self.vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, newindex-self.numstep)
				for i in upcoming:
					if i in self.cache:
						#print "grabbing frame {0}".format(i)
						self.vid.grab()
					else:
						#print "reading frame {0}".format(i)
						(rv, frame) = self.vid.read()
						if rv:
							self.cache[i] = frame
							#self.stripes[i] = getslice(frame, keyframes.get(i, None))
				
				self.mru = [i for i in self.mru if i not in upcoming] + upcoming
		
		if newindex not in self.cache:
			vidpos = int(self.vid.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
			if vidpos != newindex:
				print "seeking to {0}".format(newindex)
				self.vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, newindex)
			
			#print "reading frame {0}".format(newindex)
			(rv,frame) = self.vid.read()
			if rv:
				self.cache[newindex] = frame
				#self.stripes[newindex] = getslice(frame, keyframes.get(newindex, None))
			
			self.mru = [i for i in self.mru if i != newindex] + [newindex]
		
		
		self.mru = self.mru[-self.numcache:]
		self.cache = { i: frame for i,frame in self.cache.iteritems() if i in self.mru }
		#self.stripes = {i: self.stripes[i] for i in self.mru}
			
	def read(self, newindex=None):
		if newindex is None:
			newindex = self.index + 1

		self._prefetch(newindex)
		self.index = newindex
		
		return self.cache[self.index]

########################################################################

VK_LEFT = 2424832
VK_RIGHT = 2555904
VK_SPACE = 32

VK_PGUP = 2162688
VK_PGDN = 2228224

def iround(x):
	return int(round(x))

def sgn(x):
	return (x > 0) - (x < 0)

def redraw_display():
	#print "redraw"
	if mousedown:
		cursorcolor = (255, 0, 0)
	else:
		cursorcolor = (255, 255, 0)


	# anchor is animated
	(ax,ay) = anchor
	ay = meta['anchor'][1]
	Anchor = np.matrix([
		[1, 0, -ax],
		[0, 1, -ay],
		[0, 0, 1.0],
	])
	InvAnchor = np.linalg.inv(Anchor)
	scale = meta['scale']
	Scale = np.matrix([
		[scale, 0, 0],
		[0, scale, 0],
		[0, 0, 1.0]
	])
	
	# position is fixed in meta
	Translate = np.matrix([
		[1, 0, position[0]],
		[0, 1, position[1]],
		[0, 0, 1.0]
	])
	
	M = Translate * Scale * Anchor
	InvM = np.linalg.inv(M)

	viewbox = meta['viewbox']
	
	if draw_output:
		surface = cv2.warpAffine(curframe, M[0:2,:], (screenw, screenh), flags=cv2.INTER_AREA)
		
		cv2.rectangle(surface, tuple(viewbox[0:2]), tuple(viewbox[2:4]), (0,255,255), thickness=2)

		cv2.line(surface,
			(position[0]-10, position[1]-10),
			(position[0]+10, position[1]+10), 
			cursorcolor,  thickness=2)
		cv2.line(surface,
			(position[0]+10, position[1]-10),
			(position[0]-10, position[1]+10), 
			cursorcolor,  thickness=2)

		cv2.imshow("output", surface)


	source = curframe.copy()
	
	cv2.line(source,
		(anchor[0]-10, anchor[1]-10),
		(anchor[0]+10, anchor[1]+10), 
		cursorcolor,  thickness=2)
	cv2.line(source,
		(anchor[0]+10, anchor[1]-10),
		(anchor[0]-10, anchor[1]+10), 
		cursorcolor,  thickness=2)
	
	# todo: transform using inverse M
	TL = InvM * np.matrix([[viewbox[0], viewbox[1], 1]]).T
	BR = InvM * np.matrix([[viewbox[2], viewbox[3], 1]]).T
	
	TL = tuple(map(iround, np.array(TL[0:2,0].T).tolist()[0]))
	BR = tuple(map(iround, np.array(BR[0:2,0].T).tolist()[0]))
	
	cv2.rectangle(source,
		TL, BR,
		(255, 0, 0), thickness=2)

	cv2.imshow("source", source)
	
	if draw_graph:
		global graphbg, graphbg_head, graphbg_indices
		# graphslices/2 is midpoint
		
		# draw this range
		imax = src.index + graphslices//2
		imin = imax - graphslices
		indices = range(imax, imin, -1)

		if graphbg is None: # full redraw
			t0 = time.clock()
			graphbg = [
				src.cache[i][get_keyframe(i)[1]] if i in src.cache else emptyrow
				for i in indices
			]
			t1 = time.clock()
			graphbg = np.array(graphbg, dtype=np.uint8)
			t2 = time.clock()
			graphbg_head = imax
			graphbg_indices = set(indices) & set(src.cache)

			print "graphbg redraw {0:.3f} {1:.3f}".format(t1-t0, t2-t1)
		
		if graphbg_head != imax: # scrolling to current position
			t0 = time.clock()
			shift = imax - graphbg_head
			graphbg = np.roll(graphbg, shift, axis=0)
			oldhead = graphbg_head
			graphbg_head = imax
			t1 = time.clock()
			
			# replace rolled-over lines
			ashift = min(graphslices, abs(shift))
			
			if shift > 0:
				#import pdb; pdb.set_trace()
				newindices = xrange(imax, imax-shift, -1)
				graphbg_indices = set(i for i in graphbg_indices if i > imin)
			elif shift < 0:
				#import pdb; pdb.set_trace()
				newindices = xrange(imin-shift, imin, -1)
				graphbg_indices = set(i for i in graphbg_indices if i <= imax)
			
			t2 = time.clock()
			replacements = [
				src.cache[i][get_keyframe(i)[1]] if i in src.cache else emptyrow
				for i in newindices
			]
			graphbg_indices.update( set(newindices) & set(src.cache) )

			t3 = time.clock()

			if shift > 0:
				graphbg[:ashift] = replacements
			elif shift < 0:
				graphbg[-ashift:] = replacements
			
			t4 = time.clock()

			#print "graphbg roll {0:.3f} {1:.3f} {1:.3f} {1:.3f}".format(t1-t0, t2-t1, t3-t2, t4-t3)

		updates = (set(indices) & set(src.cache)) - graphbg_indices
		if updates:
			for i in updates:
				graphbg[graphbg_head - i] = src.cache[i][get_keyframe(i)[1]]
			graphbg_indices.update(updates)
		
		graph = cv2.resize(graphbg, (screenw, graphheight), interpolation=cv2.INTER_NEAREST)

		lines = np.array([
			( iround(keyframes[index][0]), (imax - index) * (graphscale / framerate) )
			for index in xrange(imin, imax+1)
			if index in keyframes
		], dtype=np.int32)

		now = iround((imax - src.index) * graphscale / framerate)
		cv2.line(graph,
			(0, now), (screenw, now), (255, 255, 255), thickness=2)

		if lines.shape[0] > 0:
			cv2.polylines(
				graph,
				[lines],
				False,
				(255, 255, 0),
				thickness=2
			)
			for pos in lines:
				x,y = pos
				cv2.line(
					graph,
					(x-10, y), (x+10, y),
					(0, 255, 255),
					thickness=1)

		cv2.imshow("graph", graph)


def onmouse(event, x, y, flags, userdata):
	global mousedown
	
	if event == cv2.EVENT_MOUSEMOVE:
		#print "move", event, (x,y), flags
		
		if flags == cv2.EVENT_FLAG_LBUTTON:
			#print "onmouse move lbutton", (x,y), flags, userdata
			set_cursor(x,y)

	elif event == cv2.EVENT_LBUTTONDOWN:
		#print "onmouse buttondown", (x,y), flags, userdata
		mousedown = True
		set_cursor(x, y)

	elif event == cv2.EVENT_LBUTTONUP:
		#print "onmouse buttonup", (x,y), flags, userdata
		set_cursor(x, y)
		mousedown = False

def onmouse_output(event, x, y, flags, userdata):
	global redraw, curframe, anchor
	
	if (event == cv2.EVENT_LBUTTONDOWN) or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
		newindex = iround(totalframes * x / screenw)
		curframe = src.read(newindex)
		if mousedown:
			keyframes[src.index] = anchor
		else:
			#anchor = keyframes.get(src.index, anchor)
			anchor = get_keyframe(src.index)
		redraw = True

def onmouse_graph(event, x, y, flags, userdata):
	global redraw, curframe, anchor
	
	delta = (graphdepth/2 - y / graphscale) * framerate
	newindex = iround(src.index + delta)
	
	
	if graphdraw:
		if (event == cv2.EVENT_LBUTTONDOWN) or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
			(ax,ay) = get_keyframe(src.index)
			ax = x
			keyframes[newindex] = (ax, ay)
			redraw = True

	else:
		if (event == cv2.EVENT_LBUTTONDOWN):
			curframe = src.read(newindex)
			if mousedown:
				keyframes[src.index] = anchor
			else:
				#anchor = keyframes.get(src.index, anchor)
				anchor = get_keyframe(src.index)
			redraw = True
		else:
			pass
			#print "graph", event, (x,y), flags
	
def set_cursor(x, y):
	global anchor, redraw
	anchor = (x,y)
	keyframes[src.index] = anchor
	redraw = True

def save():
	json.dump(meta, open(metafile, "w"), indent=2, sort_keys=True)
	json.dump(keyframes, open(meta['keyframes'], 'w'), indent=2, sort_keys=True)

def get_keyframe(index):
	if index in keyframes:
		return keyframes[index]
	else:
		prev = [i for i in keyframes if i < index]
		next = [i for i in keyframes if i > index]
		prev = max(prev) if prev else None
		next = min(next) if next else None
		
		if prev is None and next is None:
			return meta['anchor']
		
		if prev is None:
			return keyframes[next]
		
		if next is None:
			return keyframes[prev]
		
		alpha = (index - prev) / (next-prev)
		#print "alpha", alpha, index, prev, next
		u = np.array(keyframes[prev])
		v = np.array(keyframes[next])
		return np.int32(0.5 + u + alpha * (v-u))

draw_output = True
draw_graph = True

graphbg = None
graphbg_head = None
graphbg_indices = set()
graphdraw = False

graphdepth = 5.0
graphscale = 150 # pixels per second
# graphslices

graphheight = iround(graphdepth * graphscale)


if __name__ == '__main__':
	metafile = sys.argv[1]
	
	if os.path.exists(metafile):
		meta = json.load(open(metafile))
	else:
		meta = {}

#	for pair in sys.argv[2:]:
#		(key, value) = pair.split("=", 1)
#		print key, value
#		meta[key] = value

	print json.dumps(meta, indent=2, sort_keys=True)
	
	assert os.path.exists(meta['source'])
	if os.path.exists(meta['keyframes']):
		keyframes = json.load(open(meta['keyframes']))	
		keyframes = {int(k): keyframes[k] for k in keyframes}
	else:
		keyframes = {}
	
	screenw, screenh = meta['screen']
	position = meta['position']
	anchor = meta['anchor']

	emptyrow = np.uint8([(0,0,0)] * screenw)
	
	# TODO: range of anchor such that viewbox inside video

	# smoothed anchor on top of that, extra value
	# needs state/history of anchor values, at least for back stepping
	#controller = PID(P=1.0, I=1.0, D=1.0)
	#controller.set_point = anchor[0]
	
	srcvid = cv2.VideoCapture(meta['source'])
	
	framerate = srcvid.get(cv2.cv.CV_CAP_PROP_FPS)
	totalframes = srcvid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
	print "{0} fps".format(framerate)
	srcw = srcvid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
	srch = srcvid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
	
	graphslices = iround(graphdepth * framerate)
	src = VideoSource(srcvid, numcache=graphslices)

	firstpos = max(keyframes)
	curframe = src.read(firstpos)
	
	try:
		cv2.namedWindow("source", cv2.WINDOW_NORMAL)
		if draw_output: cv2.namedWindow("output", cv2.WINDOW_NORMAL)
		if draw_graph: cv2.namedWindow("graph", cv2.WINDOW_NORMAL)

		cv2.resizeWindow("source", int(srcw/2), int(srch/2))
		if draw_output: cv2.resizeWindow("output", int(screenw/2), int(screenh/2))
		if draw_graph: cv2.resizeWindow("graph", int(screenw/2), graphheight)

		cv2.setMouseCallback("source", onmouse) # keys are handled by all windows
		if draw_output: cv2.setMouseCallback("output", onmouse_output) # for seeking
		if draw_graph: cv2.setMouseCallback("graph", onmouse_graph)
		
		mousedown = False # override during mousedown

		running = True
		sched = None
		playspeed = 0
		
		redraw = True # init
		while running:
			if redraw:
				redraw = False
				redraw_display()

			key = cv2.waitKey(1)
			
			if abs(playspeed) > 1e-3:
				now = time.clock()
				dt = sched - now
				if dt > 0:
					time.sleep(dt)
				else:
					sched = now

				newindex = src.index + sgn(playspeed)
				curframe = src.read(newindex)
				assert curframe is not None
				redraw = True

				if mousedown:
					keyframes[src.index] = anchor
				else:
					#anchor = keyframes.get(src.index, anchor)
					anchor = get_keyframe(src.index)

				if (src.index == 0 and playspeed < 0):
					playspeed = 0
				else:
					dt = 1 / (framerate * abs(playspeed))
					sched += dt

			if key == -1: continue
			
			if key == 27:
				running = False
				break

			print "key", key

			if key in (VK_LEFT, VK_PGUP):
				delta = 1
				if key == VK_PGUP: delta = 25
				curframe = src.read(src.index - delta)
				assert curframe is not None

				if mousedown:
					keyframes[src.index] = anchor
				else:
					#anchor = keyframes.get(src.index, anchor)
					anchor = get_keyframe(src.index)

				redraw = True

			if key in (VK_RIGHT, VK_PGDN):
				delta = 1
				if key == VK_PGDN: delta = 25
				curframe = src.read(src.index + delta)
				assert curframe is not None

				if mousedown:
					keyframes[src.index] = anchor
				else:
					#anchor = keyframes.get(src.index, anchor)
					anchor = get_keyframe(src.index)
				
				redraw = True
			
			if key == ord('x'):
				if src.index in keyframes:
					del keyframes[src.index]
					#lasti = max(i for i in keyframes if i < src.index)
					#anchor = keyframes[lasti]
					anchor = get_keyframe(src.index)
					redraw = True
			
			if key == ord('c'): # cache all frames in the graph
				imin = iround(src.index - graphdepth/2 * framerate)
				imax = iround(src.index + graphdepth/2 * framerate)
				src.cache_range(imin, imax+1)
				redraw = True
				graphbg = None
			
			if key == ord('g'):
				draw_graph = not draw_graph
				if draw_graph:
					redraw = True
				
			if key == ord('o'):
				draw_output = not draw_output
				if draw_output:
					redraw = True
				
			if key == ord('s'):
				save()
				print "saved"
			
			if key == ord('d'):
				graphdraw = not graphdraw
				print "graphdraw", graphdraw
			
			if key == ord('l'):
				playspeed += 0.2
				print "speed: {0}".format(playspeed)
				sched = time.clock()

			if key == ord('j'):
				playspeed -= 0.2
				print "speed: {0}".format(playspeed)
				sched = time.clock()

			if key in (VK_SPACE, ord('k')):
				if abs(playspeed) > 1e-3:
					playspeed = 0.0
				else:
					playspeed = 1.0
					sched = time.clock()


	# space -> stop/play
	# left/right -> frame step

	# need a frame cache. read-ahead for k frames (10? 25?), for rev

	# need s-proportional
	# need v-proportional
	# switchable

	finally:
		cv2.destroyWindow("source")
		if draw_output: cv2.destroyWindow("output")
		if draw_graph: cv2.destroyWindow("graph")
		save()
