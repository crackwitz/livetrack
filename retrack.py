from __future__ import division
import os
import sys
import time
import numpy as np
import cv2
import json
import pprint; pp = pprint.pprint

from mosse import MOSSE
from opencv_common import RectSelector

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
		requested = range(start, stop+1)
		requested = [i for i in requested if i not in self.cache]
		if not requested: return
		start = min(requested)
		stop = max(requested)
		requested = range(start, stop+1)
		vidpos = int(self.vid.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
		if start != vidpos:
			print "cache_range: seeking from {0} to {1}".format(vidpos, start)
			self.vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, start)
		for i in requested:
			rv = self.vid.grab()
			if not rv: continue
			if i not in self.cache:
				(rv, frame) = self.vid.retrieve()
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
				
				self.mru = [i for i in self.mru if i not in upcoming] + upcoming
		
		if newindex not in self.cache:
			vidpos = int(self.vid.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
			if vidpos != newindex:
				print "seeking to {0}".format(newindex)
				self.vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, newindex)
			
			#print "reading frame {0}".format(newindex)
			(rv,frame) = self.vid.read()
			assert rv
			self.cache[newindex] = frame
			
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

def clamp(low, high, value):
	if value < low: return low
	if value > high: return high
	return value

def redraw_display():
	#print "redraw"
	if mousedown:
		cursorcolor = (255, 0, 0)
	else:
		cursorcolor = (255, 255, 0)


	# anchor is animated
	(ax, ay) = anchor
	(axi, ayi) = map(iround, anchor)

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
		
		timepos = iround(screenw * src.index / totalframes)
		cv2.line(surface,
			(timepos, 0), (timepos, 20),
			(255, 255, 0), thickness=4)

		cv2.imshow("output", surface)


	source = curframe.copy()
	
	cv2.line(source,
		(axi-10, ayi-10),
		(axi+10, ayi+10), 
		cursorcolor,  thickness=2)
	cv2.line(source,
		(axi+10, ayi-10),
		(axi-10, ayi+10), 
		cursorcolor,  thickness=2)
	
	# todo: transform using inverse M
	TL = InvM * np.matrix([[viewbox[0], viewbox[1], 1]]).T
	BR = InvM * np.matrix([[viewbox[2], viewbox[3], 1]]).T
	
	TL = tuple(map(iround, np.array(TL[0:2,0].T).tolist()[0]))
	BR = tuple(map(iround, np.array(BR[0:2,0].T).tolist()[0]))
	
	cv2.rectangle(source,
		TL, BR,
		(255, 0, 0), thickness=2)

	secs = src.index / framerate
	hours, secs = divmod(secs, 3600)
	mins, secs = divmod(secs, 60)
	cv2.rectangle(source,
		(0, screenh), (screenw, screenh-70),
		(0,0,0),
		cv2.cv.CV_FILLED
		)
	text = "{h:.0f}:{m:02.0f}:{s:06.3f} / frame {frame}".format(h=hours, m=mins, s=secs, frame=src.index)
	cv2.putText(source,
		text,
		(10, screenh-10), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 3)
	
	if use_tracker:
		tracker_rectsel.draw(source)
	
	if tracker:
		tracker.draw_state(source, trackerscale)
		cv2.imshow('tracker state', tracker.state_vis)
	
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
				src.cache[i][clamp(0, screenh-1, get_keyframe(i)[1])] if (i in src.cache) else emptyrow
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
				newindices = xrange(imax, imax-ashift, -1)
				graphbg_indices = set(i for i in graphbg_indices if i > imin)
			elif shift < 0:
				#import pdb; pdb.set_trace()
				newindices = xrange(imin+ashift, imin, -1)
				graphbg_indices = set(i for i in graphbg_indices if i <= imax)
			
			t2 = time.clock()
			replacements = [
				src.cache[i][clamp(0, screenh-1, get_keyframe(i)[1])] if (i in src.cache) else emptyrow
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
				graphbg[graphbg_head - i] = src.cache[i][clamp(0, screenh-1, get_keyframe(i)[1])]
			graphbg_indices.update(updates)
		
		graph = cv2.resize(graphbg, (screenw, graphheight), interpolation=cv2.INTER_NEAREST)

		lineindices = [i for i in range(imin, imax+1) if keyframes[i] is not None]
		lines = np.array([
			( iround(keyframes[index][0]), (imax - index) * (graphscale / framerate) )
			for index in lineindices
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
			for i,pos in zip(lineindices,lines):
				x,y = pos
				
				points = np.array(map(get_keyframe, [i-1, i, i+1]))
				
				d2 = np.linalg.norm((points[0]+points[2])/2 - points[1])
				
				spread = 10 # todo: some way to tell if this was raw or smoothed
				spread = 5 + iround(d2 * 20)
				
				thickness = 1
				color = (0, 255, 255)
				if graphsmooth_start is not None:
					if graphsmooth_start <= i <= graphsmooth_stop:
						thickness = 3
						color = (255,255,255)
						spread += 5
					
				cv2.line(
					graph,
					(x-spread, y), (x+spread, y),
					color,
					thickness=thickness)

		secs = src.index / framerate
		hours, secs = divmod(secs, 3600)
		mins, secs = divmod(secs, 60)
		#cv2.rectangle(graph,
		#	(0, screenh), (screenw, screenh-70),
		#	(0,0,0),
		#	cv2.cv.CV_FILLED
		#	)
		text = "{h:.0f}:{m:02.0f}:{s:06.3f} / frame {frame}".format(h=hours, m=mins, s=secs, frame=src.index)
		cv2.putText(graph,
			text,
			(10, graphheight-10), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), 3)

		cv2.imshow("graph", graph)


def onmouse(event, x, y, flags, userdata):
	global mousedown, redraw
	
	if use_tracker:
		tracker_rectsel.onmouse(event, x, y, flags, userdata)
		redraw = True
		return
	
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
	if (event == cv2.EVENT_LBUTTONDOWN) or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
		newindex = iround(totalframes * x / screenw)
		load_this_frame(newindex)

def onmouse_graph(event, x, y, flags, userdata):
	global redraw
	
	curindex = graphbg_head - iround(y / graphscale * framerate)

#	delta = (graphdepth/2 - y / graphscale) * framerate
#	newindex = iround(src.index + delta)
	
	if True:
		global graphsmooth_start, graphsmooth_stop
		
		# implement some selection dragging
		if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN)[1:]:
			graphsmooth_start = curindex
			graphsmooth_stop = curindex
			redraw = True
		
		elif (event == cv2.EVENT_MOUSEMOVE and flags in (cv2.EVENT_FLAG_LBUTTON, cv2.EVENT_FLAG_RBUTTON)[1:]):
			graphsmooth_stop = curindex
			redraw = True
			
		elif graphsmooth_start is not None and event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP)[1:]:
			graphsmooth_stop = curindex
			redraw = True
			
			indices = range(graphsmooth_start, graphsmooth_stop+1)
			
			# prepare to undo this
			oldkeyframes = {i: keyframes[i] for i in indices if keyframes[i] is not None}
			def undo():
				for i in indices: keyframes[i] = None
				for i in oldkeyframes:
					keyframes[i] = oldkeyframes[i]
			undoqueue.append(undo)
			while len(undoqueue) > 100:
				undoqueue.pop(0)
			
			support = 2
			support = range(-support, +support+1)
			
			#import pdb; pdb.set_trace()
			for i in indices:
				keyframes[i] = smoothed_keyframe(i)
			
			graphsmooth_start = None

	if graphdraw:
		if (event == cv2.EVENT_LBUTTONDOWN) or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
			(ax,ay) = get_keyframe(src.index)
			ax = x
			keyframes[curindex] = (ax, ay)
			redraw = True

	else:
		if (event == cv2.EVENT_LBUTTONDOWN):
			load_this_frame(curindex)

smoothing_radius = 2
smoothing_kernel = range(-smoothing_radius, +smoothing_radius+1)

def smoothed_keyframe(i):
	#import pdb; pdb.set_trace()
	return (np.sum([get_keyframe(i+j) for j in smoothing_kernel], axis=0, dtype=np.float32) / len(smoothing_kernel)).tolist()
	
def set_cursor(x, y):
	global anchor, redraw
	anchor = (x,y)
	keyframes[src.index] = anchor
	redraw = True
	#print "set cursor", anchor

def save():
	output = json.dumps(meta, indent=2, sort_keys=True)
	open(metafile, "w").write(output)
	
	output = json.dumps(keyframes, indent=2, sort_keys=True)
	open(meta['keyframes'], 'w').write(output)

def scan_nonempty(keyframes, pos, step):
	if step < 0:
		while pos >= 0:
			if keyframes[pos] is not None:
				return pos
			pos += step

	elif step > 0:
		while pos < len(keyframes):
			if keyframes[pos] is not None:
				return pos
			pos += step
	
	return None

def get_keyframe(index):
	if keyframes[index] is not None:
		return keyframes[index]

	else:
		prev = scan_nonempty(keyframes, index-1, -1)
		next = scan_nonempty(keyframes, index+1, +1)
		
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

def on_tracker_rect(rect):
	global tracker, use_tracker
	print "rect selected:", rect
	tracker = MOSSE(curframe_gray, tracker_downscale(rect))
	set_cursor(*tracker_upscale(tracker.pos))

def load_delta_frame(delta):
	result = None
	
	if delta in (-1, +1):
		load_this_frame(src.index + delta, False)
		
		if trailing_smooth:
			target = src.index + -5 * delta
			keyframes[target] = smoothed_keyframe(target)
	
		if tracker:
			global playspeed
			tracker.update(curframe_gray, 0.2) # tweakable parameter, default 0.125
			#print "tracker updated: xy", tpos
			if tracker.good:
				tpos = tracker_upscale(tracker.pos)
				set_cursor(*tpos)
			else:
				result = True # stop
				print "tracking bad, aborting"

	else: # big jump
		load_this_frame(src.index + delta, bool(tracker))

	if (delta > 0) and (graphbg_head is not None) and (draw_graph):
		imax = graphbg_head
		imin = imax - graphslices//2
		src.cache_range(imin, imax)
	
	return result

def load_this_frame(index=None, update_tracker=True):
	global curframe, curframe_gray, redraw, anchor
	
	if index is not None:
		pass
	else:
		index = src.index

	delta = index - src.index
		
	curframe = src.read(index) # sets src.index
	curframe_gray = cv2.pyrDown(cv2.cvtColor(curframe, cv2.COLOR_BGR2GRAY))
	
	anchor = get_keyframe(src.index)
	
	#print "frame", src.index, "anchor {0:8.3f} x {1:8.3f}".format(*anchor)

	if update_tracker and tracker:
		print "set tracker to", tracker.pos
		tracker.pos = tracker_downscale(anchor)

	redraw = True

def tracker_upscale(point):
	return tuple(v * trackerscale for v in point)

def tracker_downscale(point):
	return tuple(iround(v / trackerscale) for v in point)
	
draw_output = True
draw_graph = True

graphbg = None
graphbg_head = None
graphbg_indices = set()

graphdraw = False

graphsmooth_start = None
graphsmooth_stop = None

trailing_smooth = False

graphdepth = 5.0
graphscale = 150 # pixels per second
# graphslices

graphheight = iround(graphdepth * graphscale)

tracker = None
use_tracker = False
tracker_rectsel = RectSelector(on_tracker_rect)
trackerscale = 2.0

undoqueue = []

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
	totalframes = int(srcvid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	print "{0} fps".format(framerate)
	srcw = srcvid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
	srch = srcvid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
	
	assert os.path.exists(meta['source'])
	if os.path.exists(meta['keyframes']):
		keyframes = json.load(open(meta['keyframes']))	
		#keyframes = {int(k): keyframes[k] for k in keyframes}
	else:
		keyframes = [None] * totalframes
	
	graphslices = iround(graphdepth * framerate)
	src = VideoSource(srcvid, numcache=graphslices+10)

	if not all(k is None for k in keyframes):
		lastkey = max(k for k in xrange(totalframes) if keyframes[k] is not None)
		load_this_frame(lastkey+1)
	else:
		load_this_frame(0)

	print "frame", src.index
	
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
			#assert not any(isinstance(v, np.ndarray) for v in keyframes.itervalues())
				
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

				do_stop = load_delta_frame(sgn(playspeed))

				if mousedown:
					keyframes[src.index] = anchor
				else:
					anchor = get_keyframe(src.index)

				if (src.index == 0 and playspeed < 0):
					playspeed = 0
				else:
					dt = 1 / (framerate * abs(playspeed))
					sched += dt
				
				if do_stop:
					playspeed = 0

			if key == -1: continue
			
			if key == 27:
				running = False
				break

			#print "key", key

			if key in (VK_LEFT, VK_PGUP):
				delta = 1
				if key == VK_PGUP: delta = 25
				
				if mousedown:
					keyframes[src.index] = anchor
				else:
					anchor = get_keyframe(src.index)

				load_delta_frame(-delta)

			if key in (VK_RIGHT, VK_PGDN):
				delta = 1
				if key == VK_PGDN:
					delta = 25
					src.cache_range(src.index, src.index+delta+1)
				
				if mousedown:
					keyframes[src.index] = anchor
				else:
					anchor = get_keyframe(src.index)
			
				load_delta_frame(delta)

			if key == ord('x'):
				if keyframes[src.index] is not None:
					keyframes[src.index] = None
					anchor = get_keyframe(src.index)
					redraw = True
			
			if key == ord('c'): # cache all frames in the graph
				imax = graphbg_head
				imin = imax - graphslices
				src.cache_range(imin, imax)
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

			if key == ord('m'):
				trailing_smooth = not trailing_smooth
				print "trailing smooth", trailing_smooth
			
			if key == 26: # ctrl-z
				if undoqueue:
					print "undoing..."
					item = undoqueue.pop()
					item()
					print "undone"
					redraw = True
				else:
					print "nothing to be undone"
			
			if key == ord('l'):
				playspeed += 0.5
				print "speed: {0}".format(playspeed)
				sched = time.clock()

			if key == ord('j'):
				playspeed -= 0.5
				print "speed: {0}".format(playspeed)
				sched = time.clock()

			if key == ord('k'):
				if abs(playspeed) > 1e-3:
					playspeed = 0.0
				else:
					playspeed = 1.0
					sched = time.clock()
			
			if key == VK_SPACE:
				if abs(playspeed) > 1e-3:
					playspeed = 0.0
				else:
					playspeed = 10
					sched = time.clock()
			
			if key == ord('t'):
				use_tracker = not use_tracker
				print "use tracker:", use_tracker
				tracker = None
				cv2.destroyWindow('tracker state')
				tracker_rectsel.enabled = use_tracker
				redraw = True


	# space -> stop/play
	# left/right -> frame step

	# need a frame cache. read-ahead for k frames (10? 25?), for rev

	# need s-proportional
	# need v-proportional
	# switchable

	finally:
		cv2.destroyWindow('tracker state')
		cv2.destroyWindow("source")
		cv2.destroyWindow("output")
		cv2.destroyWindow("graph")
		save()
