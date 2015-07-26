from __future__ import division
import os
import sys
import time
import numpy as np
import scipy.ndimage
import cv2
import json
from multiprocessing.pool import ThreadPool
from collections import deque
import pprint; pp = pprint.pprint

from mosse import MOSSE
from opencv_common import RectSelector

import ffwriter
from cachingvideoreader import RateChangedVideo

########################################################################


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
		if start < 0: start = 0
		if stop >= totalframes: stop = totalframes-1
		assert start <= stop
		requested = range(start, stop+1)
		requested = [i for i in requested if i not in self.cache]
		if not requested: return
		start = min(requested)
		stop = max(requested)
		requested = range(start, stop+1)
		vidpos = self.vid.tell()
		if start != vidpos:
			print "cache_range: seeking from {0} to {1}".format(vidpos, start)
			self.vid.seek(start)
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
				self.vid.seek(newindex-self.numstep)
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
			vidpos = self.vid.tell()
			if vidpos != newindex:
				print "seeking to {0}".format(newindex)
				self.vid.seek(newindex)
			
			#print "reading frame {0}".format(newindex)
			(rv,frame) = self.vid.read()
			if rv:
				self.cache[newindex] = frame
			else:
				return
			
		self.mru = [i for i in self.mru if i != newindex] + [newindex]
			
		self.mru = self.mru[-self.numcache:]
		self.cache = { i: frame for i,frame in self.cache.iteritems() if i in self.mru }
		#self.stripes = {i: self.stripes[i] for i in self.mru}
			
	def read(self, newindex=None):
		if newindex is None:
			newindex = self.index + 1

		if not (0 <= newindex < totalframes):
			return None

		self._prefetch(newindex)
		self.index = newindex
		
		if self.index not in self.cache:
			return None
		
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

def fix8(a):
	if isinstance(a, (int, float)):
		return a * 256
	else:
		return tuple((np.array(a) * 256).round().astype(np.int32))

def redraw_display():
	#print "redraw"
	if mousedown:
		cursorcolor = (255, 0, 0)
	else:
		cursorcolor = (255, 255, 0)

	# anchor is animated

	(xmin, xmax) = meta['anchor_x_range']
	(ymin, ymax) = meta['anchor_y_range']

	# anchor within bounds
	(ax, ay) = canchor = np.clip(anchor, [xmin, ymin], [xmax, ymax])

	# anchor cross will be updated
	cpos = np.float32(position) + (anchor - canchor) * meta['scale']
	# *scale to compensate the offset in screen space

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

	if draw_output:
		surface = cv2.warpAffine(curframe, M[0:2,:], (screenw, screenh), flags=cv2.INTER_AREA)
		
		cv2.line(surface,
			fix8(cpos + (+10, +10)),
			fix8(cpos - (+10, +10)),
			cursorcolor,  thickness=iround(1/dispscale), shift=8, lineType=cv2.LINE_AA)
		cv2.line(surface,
			fix8(cpos + (+10, -10)),
			fix8(cpos - (+10, -10)),
			cursorcolor,  thickness=iround(1/dispscale), shift=8, lineType=cv2.LINE_AA)
		
		timepos = (screenw * src.index / totalframes)
		cv2.line(surface,
			fix8([timepos, 0]),
			fix8([timepos, 20]),
			(255, 255, 0), thickness=iround(2/dispscale), shift=8, lineType=cv2.LINE_AA)

		cv2.imshow("output", surface)

	if draw_input and curframe is not None:
		source = curframe.copy()

		cv2.line(source,
			fix8(anchor - 10),
			fix8(anchor + 10), 
			cursorcolor,  thickness=iround(1/dispscale), shift=8, lineType=cv2.LINE_AA)
		cv2.line(source,
			fix8(anchor + (+10, -10)),
			fix8(anchor - (+10, -10)),
			cursorcolor,  thickness=iround(1/dispscale), shift=8, lineType=cv2.LINE_AA)

		TL = InvM * np.matrix([[0, 0, 1]]).T
		BR = InvM * np.matrix([[screenw, screenh, 1]]).T

		cv2.rectangle(source,
			fix8(np.array(TL)[0:2,0]),
			fix8(np.array(BR)[0:2,0]),
			(255, 0, 0), thickness=iround(1/dispscale), shift=8, lineType=cv2.LINE_AA)

		secs = src.index / framerate
		hours, secs = divmod(secs, 3600)
		mins, secs = divmod(secs, 60)
		cv2.rectangle(source,
			(0, srch),
			(srcw, srch-70),
			(0,0,0), cv2.FILLED)

		text = "{h:.0f}:{m:02.0f}:{s:06.3f} / frame {frame}".format(h=hours, m=mins, s=secs, frame=src.index)
		cv2.putText(source,
			text,
			(10, srch-10), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 3)

		if use_faces:
			# faces are in source coordinates and scale
			if faces_roi is not None:
				cv2.rectangle(source,
					fix8(faces_roi[0:2]), fix8(faces_roi[2:4]),
					(0, 0, 255), thickness=iround(1/dispscale), shift=8, lineType=cv2.LINE_AA)

			for face in faces:
				facewh = face[2:4] - face[0:2]
				fanchor = face[0:2] + facewh * face_anchor

				cv2.polylines(source,
					fix8([
						[fanchor + facewh * 0.05, fanchor - facewh * 0.05],
						[fanchor + facewh * (+1,-1) * 0.05, fanchor - facewh * (+1,-1) * 0.05]
					]),
					False,
					(0, 255, 0), thickness=iround(1/dispscale), shift=8, lineType=cv2.LINE_AA)

				cv2.rectangle(source,
					fix8(face[0:2]), fix8(face[2:4]),
					(0, 255, 0), thickness=iround(1/dispscale), shift=8, lineType=cv2.LINE_AA)

		if use_tracker:
			tracker_rectsel.draw(source)

		if tracker:
			# scale to source resolution
			tracker.draw_state(source, 1 / trackerscale)

		cv2.imshow("source", source)
	

	if tracker and draw_tracker:
		cv2.imshow('tracker state', tracker.state_vis)

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
				src.cache[i][np.clip(get_keyframe(i)[1], 0, srch-1)] if (i in src.cache) else emptyrow
				for i in indices
			]
			t1 = time.clock()
			graphbg = np.array(graphbg, dtype=np.uint8)
			t2 = time.clock()
			graphbg_head = imax
			graphbg_indices = set(indices) & set(src.cache)
			
			print "graphbg redraw {0:.3f} {1:.3f}".format(t1-t0, t2-t1)
		
		if graphbg_head != imax: # scrolling to current position
			shift = imax - graphbg_head
			graphbg = np.roll(graphbg, shift, axis=0)
			oldhead = graphbg_head
			graphbg_head = imax
			
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
			
			replacements = [
				src.cache[i][np.clip(get_keyframe(i)[1], 0, srch-1)] if (i in src.cache) else emptyrow
				for i in newindices
			]
			graphbg_indices.update( set(newindices) & set(src.cache) )

			if shift > 0:
				graphbg[:ashift] = replacements
			elif shift < 0:
				graphbg[-ashift:] = replacements

		updates = (set(indices) & set(src.cache)) - graphbg_indices
		if updates:
			for i in updates:
				graphbg[graphbg_head - i] = src.cache[i][np.clip(get_keyframe(i)[1], 0, srch-1)]
			graphbg_indices.update(updates)
		
		graph = cv2.resize(graphbg, (srcw, graphheight), interpolation=cv2.INTER_NEAREST)

		lineindices = [i for i in range(imin, imax+1) if (0 <= i < totalframes) and keyframes[i] is not None]
		lines = np.int32([
			fix8([ keyframes[index][0], (imax - index) * graphscale ])
			for index in lineindices
		])

		now = (imax - src.index) * graphscale
		cv2.line(graph,
			fix8([0, now]),
			fix8([srcw, now]),
			(255, 255, 255), thickness=1, shift=8, lineType=cv2.LINE_AA)

		if len(lines) > 0:
			cv2.polylines(
				graph,
				[lines],
				False,
				(255, 255, 0),
				thickness=1,
				shift=8, lineType=cv2.LINE_AA
			)
			for i,pos in zip(lineindices, lines):
				x,y = pos
				
				points = np.array(map(get_keyframe, [i-1, i, i+1]))
				
				d2 = (points[0]+points[2])/2.0 - points[1]
				d2 *= 100
				
				spread = d2[0]
				spread = np.array([max(-spread, 0), max(spread, 0)]) + 5
				spread += abs(d2[1])

				thickness = 1
				color = (0, 255, 255)
				if graphsel_start is not None:
					if graphsel_start <= i <= graphsel_stop:
						thickness = 3
						color = (255,255,255)
						spread += 3
					
				cv2.line(
					graph,
					(x-int(spread[0] * 256), y), (x+int(spread[1] * 256), y),
					color,
					thickness=thickness, shift=8, lineType=cv2.LINE_AA)

		secs = src.index / framerate
		hours, secs = divmod(secs, 3600)
		mins, secs = divmod(secs, 60)
		#cv2.rectangle(graph,
		#	(0, screenh), (screenw, screenh-70),
		#	(0,0,0),
		#	cv2.FILLED
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
			set_cursor([x,y])

	elif event == cv2.EVENT_LBUTTONDOWN:
		#print "onmouse buttondown", (x,y), flags, userdata
		mousedown = True
		set_cursor([x,y])

	elif event == cv2.EVENT_LBUTTONUP:
		#print "onmouse buttonup", (x,y), flags, userdata
		set_cursor([x,y])
		mousedown = False

def onmouse_output(event, x, y, flags, userdata):
	if (event == cv2.EVENT_LBUTTONDOWN) or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
		newindex = iround(totalframes * x / screenw)
		load_this_frame(newindex)

def onmouse_graph(event, x, y, flags, userdata):
	global redraw
	
	curindex = graphbg_head - iround(y / graphscale)

	global graphsel_start, graphsel_stop

	# implement some selection dragging (for smoothing and deleting)
	if event in (cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
		graphsel_start = curindex
		graphsel_stop = curindex
		redraw = True

	elif event == cv2.EVENT_MOUSEMOVE and flags in (cv2.EVENT_FLAG_MBUTTON, cv2.EVENT_FLAG_RBUTTON):
		graphsel_stop = curindex
		redraw = True
		
	elif graphsel_start is not None and event in (cv2.EVENT_MBUTTONUP, cv2.EVENT_RBUTTONUP):
		graphsel_stop = curindex
		redraw = True

		indices = range(graphsel_start, graphsel_stop+1)

		# prepare to undo this
		oldkeyframes = {i: keyframes[i] for i in indices if keyframes[i] is not None}
		def undo():
			for i in indices: keyframes[i] = None
			for i in oldkeyframes:
				keyframes[i] = oldkeyframes[i]
		undoqueue.append(undo)
		while len(undoqueue) > 100:
			undoqueue.pop(0)

		graphsel_start = None

		### graph smoothing
		if event is cv2.EVENT_RBUTTONUP:
			updates = { i: smoothed_keyframe(i) for i in indices }
			for i in indices: keyframes[i] = updates[i]

		### graph smoothing
		if event is cv2.EVENT_MBUTTONUP:
			for i in indices: keyframes[i] = None

	if graphdraw:
		if (event == cv2.EVENT_LBUTTONDOWN) or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
			(ax,ay) = get_keyframe(curindex)
			ax = x
			keyframes[curindex] = np.float32([ax, ay])
			redraw = True

	else:
		if (event == cv2.EVENT_LBUTTONDOWN):
			load_this_frame(curindex)

smoothing_radius = 2
smoothing_kernel = range(-smoothing_radius, +smoothing_radius+1)

def smoothed_keyframe(i):
	#import pdb; pdb.set_trace()
	return np.sum([get_keyframe(i+j) for j in smoothing_kernel], axis=0, dtype=np.float32) / len(smoothing_kernel)
	
def set_cursor(newanchor):
	global anchor, redraw
	if not isinstance(newanchor, np.float32):
		newanchor = np.float32(newanchor)
	keyframes[src.index] = anchor = newanchor
	redraw = True
	#print "set cursor", anchor

def save(do_query=False):
	# meta file
	output = json.dumps(meta, indent=2, sort_keys=True)

	do_write = True
	exists = os.path.exists(metafile)
	
	if exists:
		do_write &= (open(metafile).read() != output)

	if do_query and do_write:
		do_write &= (raw_input("write meta file? (y/n) ").lower().startswith('y'))

	if do_write:
		if exists:
			bakfile = "{0}.bak".format(metafile)
			if os.path.exists(bakfile):
				os.unlink(bakfile)
			os.rename(metafile, bakfile)

		open(metafile, "w").write(output)
		print "wrote metafile"
	
	# keyframes
	output = json.dumps(
		[None if (x is None) else x.tolist() for x in keyframes],
		indent=2,
		sort_keys=True)
	
	do_write = True
	exists = os.path.exists(meta['keyframes'])
	
	if exists:
		do_write &= (open(meta['keyframes']).read() != output)
		
	if do_query and do_write:
		do_write &= (raw_input("write keyframes? (y/n) ").lower().startswith('y'))

	if do_write:
		if exists:
			bakfile = "{0}.bak".format(meta['keyframes'])
			if os.path.exists(bakfile):
				os.unlink(bakfile)
			os.rename(meta['keyframes'], bakfile)

		open(meta['keyframes'], "w").write(output)
		print "wrote keyframes"

def scan_nonempty(keyframes, pos, step):
	if step < 0:
		while pos >= 0 and step < 0:
			if keyframes[pos] is not None:
				return pos
			pos -= 1
			step += 1
		else:
			return None

	elif step > 0:
		while pos < len(keyframes) and step > 0:
			if keyframes[pos] is not None:
				return pos
			pos += 1
			step -= 1
		else:
			return None
	
	return None

def get_keyframe(index):
	if not (0 <= index < totalframes):
		return np.float32(meta['anchor'])
	
	if keyframes[index] is not None:
		return np.float32(keyframes[index])

	else:
		prev = scan_nonempty(keyframes, index-1, -100)
		next = scan_nonempty(keyframes, index+1, +100)
		
		if prev is None and next is None:
			return np.float32(meta['anchor'])
		
		if prev is None:
			return np.float32(keyframes[next])
		
		if next is None:
			return np.float32(keyframes[prev])
		
		alpha = (index - prev) / (next-prev)
		#print "alpha", alpha, index, prev, next
		u = np.array(keyframes[prev])
		v = np.array(keyframes[next])
		return np.float32(0.5 + u + alpha * (v-u))

def on_tracker_rect(rect):
	global tracker, use_tracker
	print "rect selected:", rect
	tracker = MOSSE(curframe_gray, tracker_downscale(rect))
	set_cursor(tracker_upscale(tracker.pos))

def load_delta_frame(tdelta):
	global redraw
	result = None # True ~ stop
	
	if tdelta in (-1, +1):
		oldanchor = anchor
		load_this_frame(src.index + tdelta, False) # changes anchor
		
		if curframe is not None:
			newanchor = oldanchor.copy()

			if tracker:
				xydelta = tracker.track(curframe_gray, pos=(oldanchor * trackerscale)) / trackerscale

				if tracker.good:
					newanchor += xydelta
				else:
					result = True # stop
					print "tracking bad, aborting"

			if use_faces: # and tracker and tracker.good:
				global faces_roi # will be set

				if tracker:
					trackersize = np.float32(tracker.size) / trackerscale # from tracker scale to source scale
					faces_roi = np.hstack([
						newanchor + trackersize * faces_rel_roi[0:2],
						newanchor + trackersize * faces_rel_roi[2:4]
					])
				else:
					faces_roi = None

				global faces
				faces = detect_faces(subrect=faces_roi)

				if tracker and tracker.good and len(faces) >= 1:
					faces.sort(key=(lambda face:
						np.linalg.norm(
							newanchor - \
								(face[0:2] + (face[2:4] - face[0:2]) * face_anchor))))

					face = faces[0]
					facesize = face[2:4] - face[0:2]
					faceanchor = face[0:2] + facesize * face_anchor

					newanchor += (faceanchor - newanchor) * face_attract_rate

					redraw = True

			if tracker and tracker.good:
				# use (dx,dy) from above, possibly updated by face pos

				tracker.adapt(curframe_gray, rate=tracker_adapt_rate, pos=(newanchor * trackerscale))

				set_cursor(newanchor)

				# update xt
				if draw_graph:
					graphbg[graphbg_head - src.index] = \
						src.cache[src.index][ np.clip(newanchor[1], 0, srch-1) ]

	else: # big jump
		load_this_frame(src.index + tdelta, bool(tracker))

	if curframe is None:
		return True # stop

	if (tdelta > 0) and (graphbg_head is not None) and (draw_graph):
		imax = graphbg_head
		imin = imax - graphslices//2
		src.cache_range(imin, imax)
	
	return result

def load_this_frame(index=None, update_tracker=True, only_decode=False):
	global curframe, curframe_gray, redraw, anchor
	
	if index is not None:
		pass
	else:
		index = src.index
	
	if not (0 <= index < totalframes):
		curframe = None
		return

	delta = index - src.index
		
	curframe = src.read(index) # sets src.index
	if curframe is None:
		return

	if not only_decode:
		curframe_gray = cv2.resize(
			cv2.cvtColor(curframe, cv2.COLOR_BGR2GRAY),
			dsize=None,
			fx=trackerscale, fy=trackerscale,
			interpolation=cv2.INTER_AREA)
		
	anchor = get_keyframe(src.index)
	
	#print "frame", src.index, "anchor {0:8.3f} x {1:8.3f}".format(*anchor)

	if update_tracker and tracker and not only_decode:
		print "set tracker to", tracker.pos
		tracker.pos = tracker_downscale(anchor)

	redraw = True

def tracker_upscale(point):
	return tuple(v / trackerscale for v in point)

def tracker_downscale(point):
	return tuple(v * trackerscale for v in point)

def dump_video(videodest):
	output = np.zeros((totalframes, 2), dtype=np.float32)

	prevgood = None
	nextgood = None
	for i in xrange(totalframes):
		output[i] = get_keyframe(i)

	(xmin, xmax) = meta['anchor_x_range']
	(ymin, ymax) = meta['anchor_y_range']
	
	output[output[:,0] < xmin, 0] = xmin
	output[output[:,0] > xmax, 0] = xmax
	output[output[:,1] < ymin, 1] = ymin
	output[output[:,1] > ymax, 1] = ymax
	
	sigma = meta.get('sigma', 0)
	
	if sigma > 0:
		sigma *= framerate
		output[:,0] = scipy.ndimage.filters.gaussian_filter(output[:,0], sigma)

	do_pieces = ('%' in videodest)
	outseq = 1
	outvid = None

	for i,k in enumerate(output):
		if do_pieces and (i % int(600 * framerate) == 0) and (outvid is not None):
			outvid.release()
			outvid = None
			outseq += 1

		if outvid is None:
			if i == 0:
				fourcc = -1 # user config
			else:
				fourcc = cv2.VideoWriter_fourcc(*"X264")
				
			outvid = ffwriter.FFWriter(
				videodest,
				framerate, (screenw, screenh),
				codec='libx264', pixfmt='bgr24',
				moreflags='-loglevel 32 -pix_fmt yuv420p -crf 15 -preset ultrafast')
			#outvid = cv2.VideoWriter(videodest % outseq, fourcc, framerate, (screenw, screenh))
			#assert outvid.isOpened()

		load_this_frame(i, only_decode=True)

		# anchor is animated
		(ax,ay) = k

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

		#surface = cv2.warpAffine(curframe, M[0:2,:], (screenw, screenh), flags=cv2.INTER_CUBIC)
		surface = cv2.warpAffine(curframe, M[0:2,:], (screenw, screenh), flags=cv2.INTER_LINEAR)
		
		outvid.write(surface)

		if i % 10 == 0:
			#sys.stdout.write("\rframe {0} of {1} written ({2:.3f}%)".format(i, totalframes, 100.0 * i/totalframes))
			sys.stdout.flush()
			cv2.imshow("rendered", cv2.pyrDown(surface))
			key = cv2.waitKey(1)
			if key == 27: break

	cv2.destroyWindow("rendered")
	outvid.release()
	print "done"

def detect_faces(subrect=None):
	# http://docs.opencv.org/modules/objdetect/doc/cascade_classification.html#cascadeclassifier-detectmultiscale
	# expects U8 input (gray)

	facesize = minfacesize # full region

	image = curframe_gray

	if subrect is not None:
		cliprect = np.clip(subrect, [0,0,0,0], [srcw, srch, srcw, srch])
		trackrect = cliprect * trackerscale
		(x0,y0,x1,y1) = trackrect.round().astype(np.int32)

		facesize = min(facesize, int(min(x1-x0, y1-y0) * 0.3))

		image = image[y0:y1, x0:x1]


	faces = face_cascade.detectMultiScale(
		image,
		scaleFactor=1.3, minNeighbors=2, minSize=(facesize, facesize), flags=cv2.CASCADE_SCALE_IMAGE)

	if len(faces) == 0:
		return []

	if subrect is not None:
		faces[:,0:2] += (x0,y0)

	faces[:,2:4] += faces[:,0:2]

	faces *= (1 / trackerscale)

	return list(faces)

face_cascade = cv2.CascadeClassifier(
	os.path.join(
		os.getenv('OPENCV_DIR'),
		"../sources/data/haarcascades/haarcascade_frontalface_alt.xml"))

draw_input = True
draw_output = True
draw_graph = True
draw_tracker = True
dispscale = 0.5

graphbg = None
graphbg_head = None
graphbg_indices = set()

graphdraw = False

graphsel_start = None
graphsel_stop = None

graphslices = 125
graphscale = 6 # pixels per frame
# graphslices

graphheight = iround(graphslices * graphscale)

tracker = None
use_tracker = False
trackerscale = 0.5
tracker_adapt_rate = 0.2
tracker_rectsel = RectSelector(on_tracker_rect)

use_faces = False
minfacesize = 30 # for a full region (less if the tracker region is smaller)
faces_rel_roi = [-0.5, -1.0, +0.5, +0.5]
faces = []
faces_roi = None
face_attract_rate = 0.02
face_anchor = np.float32([0.5, 0.75])

undoqueue = []

if __name__ == '__main__':
	do_dump = False
	if sys.argv[1] == 'dump':
		do_dump = True
		videodest = sys.argv[3]
		sys.argv.pop(3)
		sys.argv.pop(1)
		
		print sys.argv
	
	metafile = sys.argv[1]
	
	assert os.path.exists(metafile)
	meta = json.load(open(metafile))

	screenw, screenh = meta['screen']
	position = np.float32(meta['position'])
	anchor = np.float32(meta['anchor'])

	if 'trackerscale' in meta:
		trackerscale = float(meta['trackerscale'])

	if 'dispscale' in meta:
		dispscale = float(meta['dispscale'])
		tracker_rectsel.scale = dispscale

	if 'tracker_adapt_rate' in meta:
		tracker_adapt_rate = float(meta['tracker_adapt_rate'])

	if 'face_attract_rate' in meta:
		face_attract_rate = float(meta['face_attract_rate'])

	if 'face_anchor' in meta:
		face_anchor = np.float32(meta['face_anchor'])

	if 'faces_rel_roi' in meta:
		faces_rel_roi = np.float32(meta['faces_rel_roi'])

	assert os.path.exists(meta['source'])
	srcvid = cv2.VideoCapture(meta['source'])
	
	framerate = srcvid.get(cv2.CAP_PROP_FPS)
	totalframes = int(srcvid.get(cv2.CAP_PROP_FRAME_COUNT))
	print "{0} fps".format(framerate)
	srcw = int(srcvid.get(cv2.CAP_PROP_FRAME_WIDTH))
	srch = int(srcvid.get(cv2.CAP_PROP_FRAME_HEIGHT))

	emptyrow = np.uint8([(0,0,0)] * srcw)

	decimate = 1
	while framerate / decimate > 30:
		decimate += 1
	srcvid = RateChangedVideo(srcvid, decimate=decimate)
	
	framerate /= decimate
	
	totalframes //= decimate
	totalframes -= (decimate > 1) # shouldn't be needed... debug?

	print "{0} fps effective".format(framerate)

	meta['source_fps'] = framerate
	meta['source_framecount'] = totalframes
	meta['source_wh'] = (srcw, srch)
	
	print json.dumps(meta, indent=2, sort_keys=True)
	
	keyframes = []
	
	if os.path.exists(meta['keyframes']):
		keyframes = [
			None if keyframe is None else np.float32(keyframe)
			for keyframe in json.load(open(meta['keyframes']))
		]
	
	if len(keyframes) < totalframes:
		keyframes += [None] * (totalframes - len(keyframes))
	
	if do_dump:
		src = VideoSource(srcvid, numcache=10)
		dump_video(videodest)
		sys.exit(0)
	
	src = VideoSource(srcvid, numcache=graphslices+10)
	
	if not all(k is None for k in keyframes):
		lastkey = scan_nonempty(keyframes, len(keyframes)-1, -totalframes)
		#lastkey = max(k for k in xrange(totalframes) if keyframes[k] is not None)
		load_this_frame(lastkey)
	else:
		load_this_frame(0)

	print "frame", src.index
	
	try:
		cv2.namedWindow("source", cv2.WINDOW_NORMAL)
		cv2.namedWindow("output", cv2.WINDOW_NORMAL)
		cv2.namedWindow("graph", cv2.WINDOW_NORMAL)

		cv2.resizeWindow("source", int(srcw*dispscale), int(srch*dispscale))
		cv2.resizeWindow("output", int(screenw*dispscale), int(screenh*dispscale))
		cv2.resizeWindow("graph", int(srcw*dispscale), graphheight)

		cv2.setMouseCallback("source", onmouse) # keys are handled by all windows
		cv2.setMouseCallback("output", onmouse_output) # for seeking
		cv2.setMouseCallback("graph", onmouse_graph)
		
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
					def undo(index):
						point = keyframes[src.index]
						def sub():
							print "restoring keyframe {0} (was {1})".format(index, keyframes[index] if index in keyframes else None)
							keyframes[index] = point
							print "keyframe restored to {1}".format(point)
						return sub
					undoqueue.append(undo(src.index))

					keyframes[src.index] = None
					anchor = get_keyframe(src.index)
					redraw = True

					print "keyframe {0} deleted".format(src.index)
			
			if key == ord('c'): # cache all frames in the graph
				draw_graph = True
				imax = graphbg_head
				imin = imax - graphslices
				src.cache_range(imin, imax)
				redraw = True
				graphbg = None
				print "graph cached."
			
			if key == ord('1'):
				draw_input = not draw_input
				if draw_input:
					redraw = True
				print "draw input:", draw_input

			if key == ord('2'):
				draw_output = not draw_output
				if draw_output:
					redraw = True
				print "draw output:", draw_output
				
			if key == ord('3'):
				draw_graph = not draw_graph
				if draw_graph:
					redraw = True
				print "draw graph:", draw_graph

			if key == ord('4'):
				draw_tracker = not draw_tracker
				if draw_tracker:
					redraw = True
				print "draw tracker:", draw_tracker
				
			if key == ord('s'):
				save()
				print "saved"
			
			if key == ord('d'):
				graphdraw = not graphdraw
				print "manual drawing in graph:", graphdraw

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
					print "stopped"
				else:
					playspeed = 10
					sched = time.clock()
					print "playing..."

			if key == ord('f'):
				use_faces = not use_faces
				print "use faces:", use_faces
				if not use_faces:
					faces = []
					faces_roi = None

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
		save(do_query=True)
