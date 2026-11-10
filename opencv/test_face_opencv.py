import threading
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from face_opencv import LatestFrame, create_app, inference_loop
from opencv_face_detector import OpenCVFaceDetector, find_cascade


class OpenCVFaceTests(unittest.TestCase):
    def test_official_cascade_loads_and_blank_has_no_faces(self):
        detector = OpenCVFaceDetector()
        self.assertTrue(find_cascade().is_file())
        self.assertEqual([], detector.detect(np.zeros((480, 640, 3), np.uint8)))

    def test_downscaled_coordinates_are_restored(self):
        detector = OpenCVFaceDetector(detect_width=320)
        detector.classifier = Mock()
        detector.classifier.detectMultiScale.return_value = [(50, 40, 60, 70)]
        result = detector.detect(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual([(100, 80, 220, 220)], result)
        gray = detector.classifier.detectMultiScale.call_args.args[0]
        self.assertEqual((240, 320), gray.shape)

    def test_invalid_scale_and_missing_xml_raise(self):
        with self.assertRaises(ValueError):
            OpenCVFaceDetector(scale_factor=1)
        with self.assertRaises(FileNotFoundError):
            find_cascade('does-not-exist-test.xml')

    def test_latest_frame_does_not_repeat(self):
        mailbox = LatestFrame()
        mailbox.publish(b'one')
        mailbox.publish(b'two')
        self.assertEqual(b'two', mailbox.newer(0, timeout=0)[0])
        self.assertIsNone(mailbox.newer(2, timeout=0)[0])

    def test_detection_pipeline_and_flask_stream(self):
        stop = threading.Event()

        class StopAfterPublish(LatestFrame):
            def publish(self, value):
                super().publish(value)
                stop.set()

        camera = Mock()
        camera.frames = LatestFrame()
        camera.frames.publish(np.zeros((240, 320, 3), np.uint8))
        detector = Mock()
        detector.detect.return_value = [(40, 40, 120, 160)]
        jpegs, status = StopAfterPublish(), {'error': None}
        inference_loop(detector, camera, jpegs, status, stop, 80)
        self.assertEqual(1, status['faces'])
        self.assertEqual('opencv-haar', status['detector'])
        self.assertIsNone(status['error'])
        # 独立网页停止事件，允许读取刚生成的标注JPEG。
        client = create_app(jpegs, status, threading.Event()).test_client()
        self.assertEqual(200, client.get('/healthz').status_code)
        response = client.get('/video_feed', buffered=False)
        try:
            part = next(response.response)
            body = part.split(b'\r\n\r\n', 1)[1][:-2]
            image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual((240, 320, 3), image.shape)
            self.assertGreater(int(image[100, 40, 1]), 100)
        finally:
            response.close()

    def test_empty_stream_reports_unhealthy(self):
        client = create_app(LatestFrame(), {'error': None}, threading.Event()).test_client()
        self.assertEqual(503, client.get('/healthz').status_code)


if __name__ == '__main__':
    unittest.main()
