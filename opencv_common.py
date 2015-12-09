import numpy as np
import cv2

def draw_str(dst, (x, y), s):
	cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), thickness = 4, lineType=cv2.LINE_AA)
	cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

class RectSelector:
	def __init__(self, callback):
		self.callback = callback
		self.drag_center = None
		self.drag_rect = None
		self.drag_radius = None
		self.enabled = False
	def onmouse(self, event, x, y, flags, param):
		if not self.enabled: return
		x, y = np.int16([x, y]) # BUG
		if event == cv2.EVENT_LBUTTONDOWN:
			self.drag_center = (x, y)
		if self.drag_center:
			if flags & cv2.EVENT_FLAG_LBUTTON:
				xo, yo = self.drag_center
				xr, yr = x - xo, y - yo
				x0, y0 = np.minimum([xo-xr, yo-yr], [xo+xr, yo+yr])
				x1, y1 = np.maximum([xo-xr, yo-yr], [xo+xr, yo+yr])
				self.drag_rect = None
				if x1-x0 > 0 and y1-y0 > 0:
					self.drag_rect = (x0, y0, x1, y1)
					self.drag_radius = (xr, yr)
			else:
				rect = self.drag_rect
				#self.drag_center = None
				self.drag_rect = None
				#self.drag_radius = None
				if rect:
					self.callback(rect)
	def draw(self, vis):
		if not self.drag_rect:
			return False
		x0, y0, x1, y1 = self.drag_rect
		cx = (x0+x1) * 0.5
		cy = (y0+y1) * 0.5
		fix8 = lambda *s: tuple(int(round(x * 256)) for x in s)
		cv2.rectangle(vis,
			fix8(x0, y0),
			fix8(x1, y1),
			(0, 255, 0), 2, shift=8)
		cv2.line(vis,
			fix8(cx-5, cy-5),
			fix8(cx+5, cy+5),
			(0, 255, 0), 2, shift=8)
		cv2.line(vis,
			fix8(cx+5, cy-5),
			fix8(cx-5, cy+5),
			(0, 255, 0), 2, shift=8)
		return True
	@property
	def dragging(self):
		return self.drag_rect is not None

