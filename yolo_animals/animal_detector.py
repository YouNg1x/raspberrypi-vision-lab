"""YOLOv8 multiclass animal detection with native NCNN, without PyTorch."""
from pathlib import Path

import cv2
import ncnn
import numpy as np
import yaml

ANIMAL_NAMES = {'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                'bear', 'zebra', 'giraffe', 'butterfly', 'deer', 'rabbit',
                'chicken', 'duck', 'pig', 'fox', 'wolf', 'tiger', 'lion'}


def letterbox(image, size):
    """等比例缩放到模型固定输入尺寸，保存边距用于还原框坐标。"""
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
    return canvas, scale, left, top


def decode_animals(output, image_shape, scale, left, top, names, allowed,
                   confidence=.4, iou=.45):
    """解析[4+类别数,N]输出；每个候选取最高分的类别，再执行分类别NMS。"""
    values = np.asarray(output)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    width = 4 + len(names)
    if values.ndim != 2:
        raise ValueError(f'Unexpected output shape {values.shape}')
    if values.shape[0] == width:
        values = values.T
    elif values.shape[1] != width:
        raise ValueError(f'Expected {width} box/class channels, got {values.shape}')
    values = values[np.isfinite(values).all(axis=1)]
    class_ids = values[:, 4:].argmax(axis=1)
    scores = values[np.arange(len(values)), 4 + class_ids]
    mask = ((scores > confidence) & np.isin(class_ids, list(allowed))
            & (values[:, 2] > 0) & (values[:, 3] > 0))
    values, scores, class_ids = values[mask], scores[mask], class_ids[mask]
    if not len(values):
        return []
    order = np.argsort(scores)[::-1][:1000]
    values, scores, class_ids = values[order], scores[order], class_ids[order]
    boxes = np.column_stack((values[:, 0] - values[:, 2] / 2,
                             values[:, 1] - values[:, 3] / 2,
                             values[:, 2], values[:, 3]))
    selected = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        keep = cv2.dnn.NMSBoxes(boxes[indices].tolist(), scores[indices].tolist(), confidence, iou)
        selected.extend(indices[np.asarray(keep, dtype=int).reshape(-1)].tolist())
    detections = []
    h, w = image_shape[:2]
    for index in sorted(selected, key=lambda i: scores[i], reverse=True)[:100]:
        x, y, bw, bh = boxes[index]
        x1, y1 = np.clip([(x - left) / scale, (y - top) / scale], [0, 0], [w - 1, h - 1])
        x2, y2 = np.clip([(x + bw - left) / scale, (y + bh - top) / scale], [0, 0], [w - 1, h - 1])
        if x2 > x1 and y2 > y1:
            detections.append((float(x1), float(y1), float(x2), float(y2),
                               float(scores[index]), int(class_ids[index])))
    return detections


class AnimalDetector:
    def __init__(self, model_dir, threads=4, confidence=.4, iou=.45, all_classes=False):
        self.model_dir = Path(model_dir)
        metadata = yaml.safe_load((self.model_dir / 'metadata.yaml').read_text(encoding='utf-8'))
        h, w = metadata['imgsz']
        if h != w or h <= 0 or h % 32:
            raise ValueError('Model input must be square and divisible by 32')
        if metadata.get('task') != 'detect' or metadata.get('end2end', False):
            raise ValueError('Expected standard YOLOv8 detection model without embedded NMS')
        self.size = int(h)
        raw_names = metadata['names']
        self.names = ({int(i): str(name) for i, name in raw_names.items()}
                      if isinstance(raw_names, dict) else dict(enumerate(raw_names)))
        if set(self.names) != set(range(len(self.names))):
            raise ValueError('Class IDs in metadata must be contiguous starting at 0')
        self.allowed = (set(self.names) if all_classes else
                        {i for i, name in self.names.items() if name.lower() in ANIMAL_NAMES})
        if not self.allowed:
            raise ValueError('No supported animal names in metadata; inspect names or use --all-classes')
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.opt.use_vulkan_compute = False
        for filename, loader in [('model.ncnn.param', self.net.load_param),
                                 ('model.ncnn.bin', self.net.load_model)]:
            path = self.model_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            if loader(str(path)) != 0:
                raise RuntimeError(f'NCNN could not load {path}')
        self.input_name = self.net.input_names()[0]
        outputs = self.net.output_names()
        if len(outputs) != 1:
            raise ValueError(f'Expected one decoded output, got {outputs}')
        self.output_name = outputs[0]
        self.confidence, self.iou = confidence, iou
        self.last_shape = None

    def detect(self, frame):
        padded, scale, left, top = letterbox(frame, self.size)
        # BGR -> RGB / CHW / float32 / 0~1；只用NumPy完成预处理。
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255
        with self.net.create_extractor() as extractor:
            if extractor.input(self.input_name, ncnn.Mat(tensor).clone()) != 0:
                raise RuntimeError('NCNN input failed')
            result, output = extractor.extract(self.output_name)
            if result != 0:
                raise RuntimeError(f'NCNN extract failed: {result}')
            values = np.array(output, copy=True)
        self.last_shape = values.shape
        return decode_animals(values, frame.shape, scale, left, top, self.names,
                              self.allowed, self.confidence, self.iou)
