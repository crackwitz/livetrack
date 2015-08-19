import numpy as np
import cv2
import subprocess

class NullFile(object):
	def fileno(self):
		return 0

	def write(self, data):
		pass

class FFWriter(object):
	def __init__(self, fname, fps, (width, height), codec='libx264', pixfmt=None, moreflags=''):
		self.width = width
		self.height = height

		self.proc = subprocess.Popen([
			"ffmpeg",
			'-loglevel', 'warning',
			'-f', 'rawvideo',
			'-pix_fmt', pixfmt or 'bgr24',
			'-s', '{0}x{1}'.format(width, height),
			'-r', '{0}'.format(fps),
			'-i', 'pipe:0',
			'-c:v', codec,
		] + (moreflags.split() if isinstance(moreflags, str) else moreflags) + [
			'-y',
			fname
		], stdin=subprocess.PIPE) #, stderr=NullFile())

	def isOpened(self):
		return True

	def write(self, frame):
		assert frame.dtype == np.uint8
		assert frame.shape[2] in (3, 4)
		assert frame.shape[0] == self.height
		assert frame.shape[1] == self.width
		frame.tofile(self.proc.stdin)
	
	def close(self):
		self.proc.stdin.close()
		return self.proc.wait()

	def release(self):
		self.close()

	def __del__(self):
		self.close()


if __name__ == '__main__':
	#frame = np.zeros((1080, 1920,3), dtype=np.uint8)

	cam = cv2.VideoCapture(0)
	rv,frame = cam.read()

	# https://www.ffmpeg.org/ffmpeg-codecs.html#libx264_002c-libx264rgb
	
	vid = FFWriter("test.mov", 25, frame.shape[1::-1], '-crf 15 -preset ultrafast')
	vid.write(frame)

	try:
		while True:
			rv,frame = cam.read()
			vid.write(frame)

	finally:
		vid.close()
	
	
