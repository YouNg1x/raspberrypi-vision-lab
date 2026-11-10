import threading
import unittest

import cv2
import numpy as np

from animal_detector import decode_animals, letterbox
from animal_stream import LatestFrame, create_app


class AnimalTests(unittest.TestCase):
    names = {0: 'person', 1: 'cat', 2: 'dog'}

    def decode(self, output):
        return decode_animals(output, (480, 640, 3), .75, 0, 60,
                              self.names, {1, 2})

    def test_classwise_nms_and_nonanimal_filter(self):
        output = np.zeros((7, 100), np.float32)
        output[:, 0] = [240, 240, 120, 120, .01, .9, .02]
        output[:, 1] = [241, 240, 120, 120, .01, .8, .02]
        # 同位置的狗不能被猫的NMS抑制。
        output[:, 2] = [240, 240, 120, 120, .01, .02, .85]
        output[:, 3] = [100, 100, 50, 50, .95, .1, .1]
        result = self.decode(output)
        self.assertEqual([1, 2], [item[5] for item in result])
        np.testing.assert_allclose([240, 160, 400, 320], result[0][:4])
        self.assertEqual(result, self.decode(output.T))

    def test_invalid_or_low_score_candidates_are_discarded(self):
        output = np.zeros((7, 100), np.float32)
        output[:, 0] = [240, 240, 120, 120, .01, .3, .02]
        output[:, 1] = [240, 240, 120, 120, .01, float('nan'), .9]
        self.assertEqual([], self.decode(output))

    def test_mismatched_metadata_is_an_error(self):
        with self.assertRaises(ValueError):
            self.decode(np.zeros((84, 4725)))

    def test_letterbox_480_input(self):
        frame, scale, left, top = letterbox(np.zeros((480, 640, 3), np.uint8), 480)
        self.assertEqual((480, 480, 3), frame.shape)
        self.assertEqual((.75, 0, 60), (scale, left, top))

    def test_mailbox_and_flask_jpeg(self):
        mailbox = LatestFrame()
        app = create_app(mailbox, {'error': None, 'animals': 1, 'counts': {'cat': 1}},
                         threading.Event())
        client = app.test_client()
        self.assertEqual(503, client.get('/healthz').status_code)
        ok, jpeg = cv2.imencode('.jpg', np.zeros((80, 100, 3), np.uint8))
        self.assertTrue(ok)
        mailbox.publish(jpeg.tobytes())
        health = client.get('/healthz')
        self.assertEqual(200, health.status_code)
        self.assertEqual({'cat': 1}, health.json['counts'])
        response = client.get('/video_feed', buffered=False)
        try:
            part = next(response.response)
            body = part.split(b'\r\n\r\n', 1)[1][:-2]
            decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual((80, 100, 3), decoded.shape)
        finally:
            response.close()
        self.assertIsNone(mailbox.newer(1, timeout=0)[0])


if __name__ == '__main__':
    unittest.main()
