"""OpenCV官方Haar级联人脸检测，不依赖深度学习推理框架。"""
from pathlib import Path
import math

import cv2


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
