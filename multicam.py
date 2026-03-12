""" Multi-camera capture with OpenCV, multiprocessing, and per-frame metadata.

This script demonstrates how to capture video from multiple cameras simultaneously using OpenCV 
and Python's multiprocessing module. Each camera runs in its own process, writing video frames 
to disk along with per-frame metadata (timestamp, elapsed time) in a CSV file. 
The main process can preview the latest frame from each camera in real-time, and the recording 
can be stopped by pressing 'q' or after a specified duration.

Output:
- For each camera, an MP4 video file and a corresponding CSV metadata file are saved in
  an output directory named with the current timestamp.
  
CSV Metadata Columns:
- camera_id: Identifier for the camera (integer).
- frame_number: Sequential number of the frame captured from that camera.
- datetime: Timestamp of when the frame was captured (ISO format).
- elapsed_sec: Elapsed time in seconds since the start of recording for that camera.

Author: Anthropic's Claude (modified by Jacob Gronemeyer)
Date: 2026-03-12
"""

import cv2
import numpy as np
import multiprocessing as mp
from datetime import datetime
import time
import csv
import os
from queue import Full

# ── Configuration ──────────────────────────────────────────────
NUM_CAMERAS = 3
FPS = 30
FRAME_SIZE = (640, 480)     # (width, height)
CODEC = 'mp4v'
OUTPUT_ROOT = 'output'
DURATION_SEC = None         # None = unlimited, press 'q' to stop
PREVIEW = True
SHOW_FPS = True              # overlay measured FPS on preview windows
QUEUE_SIZE = 100

# Per-camera overrides: source (int index, path, or URL), backend, use_fake.
# If omitted, cameras default to fake frames with index as ID.
CAMERAS = [
	{'source': 0, 'use_fake': True},
	{'source': 1, 'use_fake': True},
	{'source': 2, 'use_fake': True},
]

# Fake-frame colors (BGR) cycled across cameras
_COLORS = [(200, 50, 50), (50, 200, 50), (50, 50, 200)]


def _fake_frame(cam_id, n, w, h):
	"""Quick synthetic frame with moving dot and text overlay."""
	frame = np.full((h, w, 3), _COLORS[cam_id % len(_COLORS)], dtype=np.uint8)
	frame = cv2.add(frame, np.random.randint(0, 30, frame.shape, dtype=np.uint8))
	cx = int((n * 3) % w)
	cy = h // 2 + int(50 * np.sin(n * 0.05))
	cv2.circle(frame, (cx, cy), 20, (255, 255, 255), -1)
	cv2.putText(frame, f"CAM {cam_id} | {n}", (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
	cv2.putText(frame, datetime.now().strftime("%H:%M:%S.%f")[:-3],
				(10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
	return frame


class Writer:
	"""Writes video + per-frame CSV metadata for one camera."""

	def __init__(self, cam_id, output_dir):
		self.cam_id = cam_id
		self._vid = None
		self._csv_f = None
		self._csv = None
		self.frame_count = 0

		os.makedirs(output_dir, exist_ok=True) # make output directory passed during instantiation 
		fourcc = cv2.VideoWriter.fourcc(*CODEC) # set codec

		self._vid = cv2.VideoWriter(
			os.path.join(output_dir, f'camera_{cam_id}.mp4'),
			fourcc, FPS, FRAME_SIZE,
		)
  
		self._csv_f = open(
			os.path.join(output_dir, f'camera_{cam_id}_metadata.csv'), 'w', newline='',
		)
		self._csv = csv.writer(self._csv_f)
		self._csv.writerow(['camera_id', 'frame_number', 'datetime', 'elapsed_sec'])

	def write(self, frame, ts, elapsed):
		self._vid.write(frame) # write frame to video file
		self._csv.writerow([self.cam_id, self.frame_count, ts.isoformat(), f"{elapsed:.6f}"]) # write metadata to CSV
		self.frame_count += 1 # increment frame count

	def close(self):
		if self._vid:
			self._vid.release()
		if self._csv_f:
			self._csv_f.close()


class Camera:
	"""One camera: spawns a worker process, queues frames for preview."""

	def __init__(self, cam_id, output_dir, source=0, backend=None, use_fake=True):
		self.cam_id = cam_id
		self.output_dir = output_dir
		self.source = source
		self.backend = backend
		self.use_fake = use_fake
		# Note: underscores indicate these are "private" attributes 
  		# not meant for external use outside of the class methods.
		self._queue = mp.Queue(maxsize=QUEUE_SIZE)
		self._start = mp.Event()
		self._stop = mp.Event()
		self._proc = None
		self.latest_frame = None
		self._last_show_time = None
		self._display_fps = 0.0

	def start(self): # start the MultiProcess worker for this camera
		self._proc = mp.Process(target=self._run, daemon=True)
		self._proc.start()

	def begin(self):
		self._start.set() #setting the start event to signal the worker process to begin capturing frames

	def stop(self):
		self._stop.set() #setting the stop event to signal the worker process to stop 
  
		# Dequeue any remaining frames to unblock the worker if it's waiting to put frames in the queue
		while not self._queue.empty():
			try:
				self._queue.get_nowait() # attempt to get a frame from the queue without blocking; if the queue is empty, it raises an exception which we catch to break the loop
			except Exception:
				break
		if self._proc:
			self._proc.join(timeout=3)
			if self._proc.is_alive():
				self._proc.terminate()

	def poll(self):
		while not self._queue.empty():
			self.latest_frame = self._queue.get()

	def show(self):
		if self.latest_frame is not None:
			now = time.perf_counter()
			if self._last_show_time is not None:
				dt = now - self._last_show_time
				if dt > 0:
					self._display_fps = 0.9 * self._display_fps + 0.1 * (1.0 / dt)
			self._last_show_time = now

			frame = self.latest_frame
			if SHOW_FPS:
				frame = frame.copy()
				cv2.putText(frame, f"{self._display_fps:.1f} fps",
							(frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX,
							0.7, (0, 255, 255), 2)
			cv2.imshow(f"Camera {self.cam_id}", frame)

	# ── worker (runs in child process) ─────────────────────────
	def _run(self):
		self._start.wait()
		t0 = time.perf_counter_ns()
		w, h = FRAME_SIZE
		cap = None
		fake = self.use_fake

		if not fake: # ie if using real camera source, attempt to open it with OpenCV
			cap = cv2.VideoCapture(self.source) if self.backend is None \
				else cv2.VideoCapture(self.source, self.backend)
			if cap.isOpened():
				cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
				cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
				cap.set(cv2.CAP_PROP_FPS, FPS)
			else:
				cap.release()
				cap = None
				fake = True   # fallback

		writer = Writer(self.cam_id, self.output_dir)
		n = 0 # frame counter for the capture loop
		try:
			# Capture loop: read frames, write to video+CSV, queue for preview, sleep to target FPS
			while not self._stop.is_set(): 
				if fake:
					frame = _fake_frame(self.cam_id, n, w, h)
				else:
					ok, frame = cap.read()
					if not ok:
						fake = True
						continue
					if (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
						frame = cv2.resize(frame, FRAME_SIZE)

				elapsed = (time.perf_counter_ns() - t0) / 1_000_000_000
				writer.write(frame, datetime.now(), elapsed)
				n += 1

				try:
					self._queue.put_nowait(frame)
				except Full:
					pass

				#time.sleep(1.0 / FPS) # crude way to limit capture rate
		finally:
			if cap:
				cap.release()
			writer.close()


def main():
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	output_dir = os.path.join(OUTPUT_ROOT, timestamp)

	cameras = [
		Camera(
			cam_id=i,
			output_dir=output_dir,
			source=cfg.get('source', i),
			backend=cfg.get('backend'),
			use_fake=cfg.get('use_fake', True),
		)
		for i, cfg in enumerate(CAMERAS[:NUM_CAMERAS])
	]

	for cam in cameras:
		cam.start()

	input("Press Enter to start capturing...")
	for cam in cameras:
		cam.begin()
	t_start = time.perf_counter()

	print(f"Recording{f' for {DURATION_SEC}s' if DURATION_SEC else ''} — press 'q' to stop")

	try:
		while True:
			if DURATION_SEC and (time.perf_counter() - t_start) >= DURATION_SEC:
				print(f"{DURATION_SEC}s reached.")
				break

			for cam in cameras:
				cam.poll()
				if PREVIEW:
					cam.show()

			if PREVIEW and (cv2.waitKey(1) & 0xFF == ord('q')):
				break
	except KeyboardInterrupt:
		pass
	finally:
		for cam in cameras:
			cam.stop()
		cv2.destroyAllWindows()
		print(f"Saved to {output_dir}/")


if __name__ == '__main__':
	main()
