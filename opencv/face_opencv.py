"""CSI camera -> OpenCV Haar face detector -> annotated Flask MJPEG."""
import argparse
import logging
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify
from werkzeug.serving import make_server

import math


def find_cascade(explicit=None):
    """兼容pip OpenCV和树莓派apt安装的级联文件位置。"""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    filename = 'haarcascade_frontalface_default.xml'
    candidates = [Path(__file__).resolve().parent / filename]
    data = getattr(cv2, 'data', None)
    if data is not None:
        candidates.append(Path(data.haarcascades) / filename)
    candidates.extend(Path(base) / filename for base in (
        '/usr/share/opencv4/haarcascades', '/usr/share/opencv/haarcascades',
        '/usr/local/share/opencv4/haarcascades'))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        'OpenCV cascade XML not found. On Raspberry Pi install opencv-data '
        '(sudo apt install opencv-data), or pass --cascade /path/to/haarcascade_frontalface_default.xml')


class OpenCVFaceDetector:
    def __init__(self, cascade=None, detect_width=320, scale_factor=1.15,
                 min_neighbors=5, min_face=40, threads=2):
        if detect_width < 64 or not math.isfinite(scale_factor) or scale_factor <= 1:
            raise ValueError('detect-width must be >=64 and scale-factor must be finite and >1')
        if min_neighbors < 1 or min_face < 1 or threads < 1:
            raise ValueError('min-neighbors, min-face and threads must be positive')
        if not hasattr(cv2, 'CascadeClassifier'):
            raise RuntimeError(
                'This OpenCV build lacks the objdetect module (CascadeClassifier). '
                'Use the full python3-opencv package on Raspberry Pi.')
        self.cascade_path = find_cascade(cascade)
        self.classifier = cv2.CascadeClassifier(str(self.cascade_path))
        if self.classifier.empty():
            raise RuntimeError(f'Cannot load OpenCV cascade: {self.cascade_path}')
        self.detect_width = detect_width
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face = min_face
        cv2.setNumThreads(threads)

    def detect(self, frame):
        """缩小灰度图后检测，再将人脸框还原到原始相机分辨率。"""
        height, width = frame.shape[:2]
        ratio = min(1.0, self.detect_width / width)
        small_width, small_height = round(width * ratio), max(1, round(height * ratio))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ratio < 1:
            gray = cv2.resize(gray, (small_width, small_height), interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(gray)
        # 级联训练窗口为24x24；min-face参数以原图像素为单位。
        minimum = max(24, round(self.min_face * ratio))
        rectangles = self.classifier.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
            minSize=(minimum, minimum))
        sx, sy = width / small_width, height / small_height
        faces = []
        for x, y, w, h in rectangles:
            x1, y1 = max(0, round(x * sx)), max(0, round(y * sy))
            x2, y2 = min(width - 1, round((x + w) * sx)), min(height - 1, round((y + h) * sy))
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2))
        return faces

LOG = logging.getLogger('face')


class LatestFrame:
    """只保存最新一帧，不堆积旧帧；采集、推理、推流互不等待。"""
    def __init__(self):
        self.condition = threading.Condition()
        self.value = None
        self.sequence = 0
        self.timestamp = 0.0

    def publish(self, value):
        with self.condition:
            self.value = value
            self.sequence += 1
            self.timestamp = time.monotonic()
            self.condition.notify_all()

    def newer(self, sequence, timeout=1.0):
        with self.condition:
            ready = self.condition.wait_for(lambda: self.sequence > sequence, timeout)
            return (self.value if ready else None, self.sequence, self.timestamp)


class CsiCamera:
    def __init__(self, width, height, fps, index, stop):
        self.width, self.height = width, height
        self.stop = stop
        self.frames = LatestFrame()
        self.command = ['rpicam-vid', '--camera', str(index), '--timeout', '0',
                        '--nopreview', '--codec', 'yuv420', '--width', str(width),
                        '--height', str(height), '--framerate', str(fps), '--output', '-']
        self.thread = threading.Thread(target=self.run, name='csi-camera', daemon=True)

    def run(self):
        size = self.width * self.height * 3 // 2
        while not self.stop.is_set():
            process = None
            try:
                process = subprocess.Popen(self.command, stdout=subprocess.PIPE, bufsize=0)
                fd = process.stdout.fileno()
                while not self.stop.is_set():
                    data = bytearray()
                    deadline = time.monotonic() + 3.0
                    while len(data) < size and not self.stop.is_set():
                        if time.monotonic() > deadline:
                            raise TimeoutError('CSI camera produced no complete frame for 3 seconds')
                        if not select.select([fd], [], [], .2)[0]:
                            continue
                        chunk = os.read(fd, size - len(data))
                        if not chunk:
                            raise RuntimeError('rpicam-vid exited or camera disconnected')
                        data.extend(chunk)
                    if self.stop.is_set():
                        break
                    yuv = np.frombuffer(data, np.uint8).reshape(self.height * 3 // 2, self.width)
                    self.frames.publish(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420))
            except Exception:
                LOG.exception('Camera failed; retrying in 1 second')
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2)
                    process.stdout.close()
            self.stop.wait(1.0)


def create_app(jpegs, status, stop):
    app = Flask(__name__)

    @app.get('/')
    def index():
        return ('<!doctype html><html><meta name="viewport" content="width=device-width">'
                '<title>Face detection</title><body style="background:#161616;color:white;'
                'text-align:center;font-family:sans-serif"><h2>Live face detection</h2>'
                '<img src="/video_feed" style="max-width:100%;height:auto" '
                'alt="Waiting for CSI camera"><p>OpenCV Haar / Raspberry Pi 5</p></body></html>')

    @app.get('/healthz')
    def health():
        with jpegs.condition:
            age = time.monotonic() - jpegs.timestamp if jpegs.sequence else None
        healthy = age is not None and age < 5 and not status.get('error')
        return jsonify(dict(status, healthy=healthy, frame_age_s=age)), (200 if healthy else 503)

    @app.get('/video_feed')
    def video_feed():
        def generate():
            sequence = 0
            while not stop.is_set():
                data, sequence, timestamp = jpegs.newer(sequence)
                if data is not None and time.monotonic() - timestamp < 5:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n'
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame',
                        headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})
    return app


def inference_loop(detector, camera, jpegs, status, stop, quality):
    sequence = 0
    previous = time.monotonic()
    while not stop.is_set():
        frame, sequence, captured_at = camera.frames.newer(sequence)
        if frame is None or time.monotonic() - captured_at > 1:
            continue
        try:
            started = time.monotonic()
            faces = detector.detect(frame)
            elapsed_ms = (time.monotonic() - started) * 1000
            annotated = frame.copy()
            for x1, y1, x2, y2 in faces:
                p1, p2 = (round(x1), round(y1)), (round(x2), round(y2))
                cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
                cv2.putText(annotated, 'Face', (p1[0], max(20, p1[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            now = time.monotonic()
            fps = 1 / max(now - previous, 1e-6)
            previous = now
            cv2.putText(annotated, f'Faces: {len(faces)} | {fps:.1f} FPS | {elapsed_ms:.0f} ms',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            ok, encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError('JPEG encoding failed')
            status.update(faces=len(faces), inference_ms=round(elapsed_ms, 1),
                          fps=round(fps, 1), detector='opencv-haar', error=None)
            jpegs.publish(encoded.tobytes())
        except Exception as exc:
            status['error'] = str(exc)
            LOG.exception('OpenCV detection failed; stopping')
            stop.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--camera-fps', type=int, default=30)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cascade', type=Path, help='Path to the official OpenCV cascade XML')
    parser.add_argument('--detect-width', type=int, default=320, help='Downscaled detection width')
    parser.add_argument('--scale-factor', type=float, default=1.15)
    parser.add_argument('--min-neighbors', type=int, default=5)
    parser.add_argument('--min-face', type=int, default=40, help='Minimum face size in source pixels')
    parser.add_argument('--jpeg-quality', type=int, default=80)
    parser.add_argument('--check-detector', action='store_true', help='Check OpenCV cascade without camera/server')
    args = parser.parse_args()
    if args.width <= 0 or args.width % 32 or args.height <= 0 or args.height % 2:
        parser.error('I420 capture requires width divisible by 32 and even height')
    if args.camera_fps < 1 or not 1 <= args.jpeg_quality <= 100:
        parser.error('Invalid camera FPS or JPEG quality')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    detector = OpenCVFaceDetector(args.cascade, args.detect_width, args.scale_factor,
                                  args.min_neighbors, args.min_face, args.threads)
    # 启动时检查XML加载及真实OpenCV检测调用。
    detector.detect(np.zeros((args.height, args.width, 3), np.uint8))
    LOG.info('OpenCV %s | cascade=%s | detect width=%s', cv2.__version__,
             detector.cascade_path, detector.detect_width)
    if args.check_detector:
        return
    if os.name != 'posix' or shutil.which('rpicam-vid') is None:
        parser.error('Camera mode requires Raspberry Pi Linux and rpicam-vid')
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    camera = CsiCamera(args.width, args.height, args.camera_fps, args.camera, stop)
    jpegs, status = LatestFrame(), {'error': None, 'faces': 0, 'fps': 0, 'detector': 'opencv-haar'}
    app = create_app(jpegs, status, stop)
    # 多线程服务：一个长连接视频流不会阻塞其他浏览器或healthz。
    server = make_server(args.host, args.port, app, threaded=True)
    server.timeout = .5
    worker = threading.Thread(target=inference_loop,
                              args=(detector, camera, jpegs, status, stop, args.jpeg_quality),
                              name='face-inference', daemon=True)
    try:
        camera.thread.start()
        worker.start()
        LOG.info('Open http://<Raspberry-Pi-IP>:%s/ on your computer', args.port)
        while not stop.is_set():
            server.handle_request()
    finally:
        stop.set()
        camera.thread.join(timeout=5)
        worker.join(timeout=5)
        server.server_close()
    if status['error']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
