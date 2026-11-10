import threading
import unittest

import cv2
import numpy as np

from face_detector import decode_faces, letterbox
from face_stream import LatestFrame, create_app


class FaceTests(unittest.TestCase):
    def test_pose_coordinates_and_nms(self):
        # 640x480图像填充到640x640，上边距80。
        output = np.zeros((20, 100), np.float32)
        output[:5, 0] = [320, 320, 100, 120, .9]
        output[:5, 1] = [321, 320, 100, 120, .8]
        output[:5, 2] = [100, 180, 40, 60, .2]
        faces = decode_faces(output, (480, 640, 3), 1, 0, 80)
        self.assertEqual(1, len(faces))
        np.testing.assert_allclose([270, 180, 370, 300], faces[0][:4])

    def test_transposed_detect_output_and_scale(self):
        output = np.zeros((100, 5), np.float32)
        output[0] = [160, 160, 80, 80, .9]
        faces = decode_faces(output, (640, 640, 3), .5, 0, 0)
        np.testing.assert_allclose([240, 240, 400, 400], faces[0][:4])

    def test_unsupported_output_fails_explicitly(self):
        with self.assertRaises(ValueError):
            decode_faces(np.zeros((84, 8400)), (480, 640, 3), 1, 0, 80)

    def test_letterbox_preserves_geometry(self):
        image = np.full((480, 640, 3), 255, np.uint8)
        padded, scale, left, top = letterbox(image)
        self.assertEqual((640, 640, 3), padded.shape)
        self.assertEqual((1, 0, 80), (scale, left, top))
        self.assertTrue(np.all(padded[:80] == 114))
        self.assertTrue(np.all(padded[80:560] == 255))

    def test_mailbox_never_repeats_an_old_frame(self):
        mailbox = LatestFrame()
        mailbox.publish(b'first')
        mailbox.publish(b'second')
        value, seq, _ = mailbox.newer(0, timeout=0)
        self.assertEqual((b'second', 2), (value, seq))
        self.assertIsNone(mailbox.newer(seq, timeout=0)[0])

    def test_flask_stream_contains_decodable_annotated_jpeg(self):
        mailbox = LatestFrame()
        client = create_app(mailbox, {'error': None}, threading.Event()).test_client()
        self.assertEqual(503, client.get('/healthz').status_code)
        image = np.zeros((80, 100, 3), np.uint8)
        cv2.rectangle(image, (10, 10), (70, 70), (0, 255, 0), 2)
        ok, jpeg = cv2.imencode('.jpg', image)
        self.assertTrue(ok)
        mailbox.publish(jpeg.tobytes())
        self.assertEqual(200, client.get('/healthz').status_code)
        response = client.get('/video_feed', buffered=False)
        try:
            part = next(response.response)
            body = part.split(b'\r\n\r\n', 1)[1][:-2]
            decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(image.shape, decoded.shape)
        finally:
            response.close()


if __name__ == '__main__':
    unittest.main()
