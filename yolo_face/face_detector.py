"""Native NCNN YOLOv8 face detection. No torch/ultralytics dependency."""
from pathlib import Path
import re

import cv2
import ncnn
import numpy as np


def letterbox(image, size=640):
    """等比例缩放并填充，记录坐标还原所需的比例和边距。"""
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
    return canvas, scale, left, top


def decode_faces(output, image_shape, scale, left, top, confidence=.4, iou=.45):
    """解析单类别检测/五点pose输出：[cx,cy,w,h,score,...关键点]。"""
    values = np.asarray(output)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"Unsupported NCNN output shape: {values.shape}")
    if values.shape[0] in (5, 20):
        values = values.T
    elif values.shape[1] not in (5, 20):
        raise ValueError(f"Expected single-class YOLOv8 detect/pose output, got {values.shape}")
    values = values[np.isfinite(values[:, :5]).all(axis=1)]
    values = values[(values[:, 4] >= confidence) & (values[:, 2] > 0) & (values[:, 3] > 0)]
    if not len(values):
        return []
    # 限制进入NMS的候选数量，避免异常画面拖慢主循环。
    values = values[np.argsort(values[:, 4])[::-1][:1000]]
    boxes = np.column_stack((values[:, 0] - values[:, 2] / 2,
                             values[:, 1] - values[:, 3] / 2,
                             values[:, 2], values[:, 3]))
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), values[:, 4].tolist(), confidence, iou)
    h, w = image_shape[:2]
    detections = []
    for index in np.asarray(indices).reshape(-1):
        x, y, bw, bh = boxes[index]
        x1 = float(np.clip((x - left) / scale, 0, w - 1))
        y1 = float(np.clip((y - top) / scale, 0, h - 1))
        x2 = float(np.clip((x + bw - left) / scale, 0, w - 1))
        y2 = float(np.clip((y + bh - top) / scale, 0, h - 1))
        if x2 > x1 and y2 > y1:
            detections.append((x1, y1, x2, y2, float(values[index, 4])))
    return detections


class FaceDetector:
    def __init__(self, model_dir, threads=4, confidence=.4, iou=.45):
        self.model_dir = Path(model_dir)
        self.size = 640
        metadata = self.model_dir / 'metadata.yaml'
        if metadata.exists():
            text = metadata.read_text(encoding='utf-8')
            match = re.search(r'(?m)^imgsz:\s*\n\s*-\s*(\d+)\s*\n\s*-\s*(\d+)', text)
            if match:
                h, w = map(int, match.groups())
                if h != w or h % 32:
                    raise ValueError('This program expects a square input size divisible by 32')
                self.size = h
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.opt.use_vulkan_compute = False
        for filename, loader in [('model.ncnn.param', self.net.load_param),
                                 ('model.ncnn.bin', self.net.load_model)]:
            path = self.model_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            if loader(str(path)) != 0:
                raise RuntimeError(f'NCNN failed to load {path}')
        self.input_name = self.net.input_names()[0]
        names = self.net.output_names()
        if len(names) != 1:
            raise ValueError(f'Expected one decoded YOLOv8 output, got {names}')
        self.output_name = names[0]
        self.confidence, self.iou = confidence, iou
        self.last_shape = None

    def detect(self, frame):
        padded, scale, left, top = letterbox(frame, self.size)
        # OpenCV BGR -> RGB，转为连续CHW float32并归一化。
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
        with self.net.create_extractor() as extractor:
            if extractor.input(self.input_name, ncnn.Mat(tensor).clone()) != 0:
                raise RuntimeError('NCNN input failed')
            result, output = extractor.extract(self.output_name)
            if result != 0:
                raise RuntimeError(f'NCNN extract failed: {result}')
            values = np.array(output, copy=True)
        self.last_shape = values.shape
        return decode_faces(values, frame.shape, scale, left, top, self.confidence, self.iou)
