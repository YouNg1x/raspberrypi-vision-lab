"""CSI camera -> native NCNN animal detector -> annotated Flask MJPEG."""
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

from animal_detector import AnimalDetector

LOG = logging.getLogger('animal')


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
                '<title>Animal detection</title><body style="background:#161616;color:white;'
                'text-align:center;font-family:sans-serif"><h2>Live animal detection</h2>'
                '<img src="/video_feed" style="max-width:100%;height:auto" '
                'alt="Waiting for CSI camera"><p>NCNN / Raspberry Pi 5</p></body></html>')

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
            animals = detector.detect(frame)
            elapsed_ms = (time.monotonic() - started) * 1000
            annotated = frame.copy()
            for x1, y1, x2, y2, score, class_id in animals:
                p1, p2 = (round(x1), round(y1)), (round(x2), round(y2))
                cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
                cv2.putText(annotated, f'{detector.names[class_id]} {score:.2f}', (p1[0], max(20, p1[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            now = time.monotonic()
            fps = 1 / max(now - previous, 1e-6)
            previous = now
            cv2.putText(annotated, f'Animals: {len(animals)} | {fps:.1f} FPS | {elapsed_ms:.0f} ms',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            ok, encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError('JPEG encoding failed')
            counts = {}
            for *_, class_id in animals:
                name = detector.names[class_id]
                counts[name] = counts.get(name, 0) + 1
            status.update(animals=len(animals), counts=counts, inference_ms=round(elapsed_ms, 1),
                          fps=round(fps, 1), output_shape=list(detector.last_shape), error=None)
            jpegs.publish(encoded.tobytes())
        except Exception as exc:
            status['error'] = str(exc)
            LOG.exception('Inference failed; stopping (check model output and metadata)')
            stop.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--camera-fps', type=int, default=30)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--conf', type=float, default=.4)
    parser.add_argument('--iou', type=float, default=.45)
    parser.add_argument('--jpeg-quality', type=int, default=80)
    parser.add_argument('--all-classes', action='store_true', help='Show all model classes instead of animals only')
    parser.add_argument('--check-model', action='store_true', help='Run native inference without camera/server')
    args = parser.parse_args()
    if args.width <= 0 or args.width % 32 or args.height <= 0 or args.height % 2:
        parser.error('I420 capture requires width divisible by 32 and even height')
    if (args.threads < 1 or args.camera_fps < 1 or not 0 < args.conf < 1
            or not 0 < args.iou < 1 or not 1 <= args.jpeg_quality <= 100):
        parser.error('Invalid threads/FPS/confidence/IoU/JPEG quality')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cv2.setNumThreads(1)
    detector = AnimalDetector(args.model_dir, args.threads, args.conf, args.iou, args.all_classes)
    # 启动时用真实模型预热，提前检查输入尺寸和输出格式。
    detector.detect(np.zeros((args.height, args.width, 3), np.uint8))
    LOG.info('Model OK: input=%sx%s, output=%s, blobs=%s -> %s', detector.size,
             detector.size, detector.last_shape, detector.input_name, detector.output_name)
    LOG.info('Enabled classes: %s', [detector.names[i] for i in sorted(detector.allowed)])
    if args.check_model:
        return
    if os.name != 'posix' or shutil.which('rpicam-vid') is None:
        parser.error('Camera mode requires Raspberry Pi Linux and rpicam-vid')
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    camera = CsiCamera(args.width, args.height, args.camera_fps, args.camera, stop)
    jpegs, status = LatestFrame(), {'error': None, 'animals': 0, 'fps': 0}
    app = create_app(jpegs, status, stop)
    # 多线程服务：一个长连接视频流不会阻塞其他浏览器或healthz。
    server = make_server(args.host, args.port, app, threaded=True)
    server.timeout = .5
    worker = threading.Thread(target=inference_loop,
                              args=(detector, camera, jpegs, status, stop, args.jpeg_quality),
                              name='animal-inference', daemon=True)
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
