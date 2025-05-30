import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from main import MainWindow

# test_main.py



class DummyBBox:
    def __init__(self, origin_x, origin_y, width, height):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height

class DummyDetection:
    def __init__(self, bbox):
        self.bounding_box = bbox

class DummyFaceResult:
    def __init__(self, detections):
        self.detections = detections

class TestProcessRPPGSignal(unittest.TestCase):
    def setUp(self):
        self.window = MainWindow()
        self.window.face_detector = True  # Dummy non-None
        self.window.frame_buffer_limit = 5

    @patch("main.extract_rppg_signal")
    def test_no_detections(self, mock_extract):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        face_result = DummyFaceResult(detections=[])
        self.window.rppg_signal = []
        self.window._process_rppg_signal(frame, face_result)
        self.assertEqual(self.window.rppg_signal, [])

    @patch("main.extract_rppg_signal")
    def test_valid_detection(self, mock_extract):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = DummyBBox(10, 10, 50, 50)
        detection = DummyDetection(bbox)
        face_result = DummyFaceResult(detections=[detection])
        mock_extract.side_effect = [1.0, 2.0, 3.0]  # forehead, left cheek, right cheek

        self.window.rppg_signal = []
        self.window._process_rppg_signal(frame, face_result)
        # Should append mean([1.0, 2.0, 3.0]) = 2.0
        self.assertEqual(len(self.window.rppg_signal), 1)
        self.assertAlmostEqual(self.window.rppg_signal[0], 2.0)

    @patch("main.extract_rppg_signal")
    def test_partial_roi(self, mock_extract):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = DummyBBox(10, 10, 50, 50)
        detection = DummyDetection(bbox)
        face_result = DummyFaceResult(detections=[detection])
        # Only forehead and left cheek return values, right cheek is None
        mock_extract.side_effect = [1.0, 2.0, None]

        self.window.rppg_signal = []
        self.window._process_rppg_signal(frame, face_result)
        # Should append mean([1.0, 2.0]) = 1.5
        self.assertEqual(len(self.window.rppg_signal), 1)
        self.assertAlmostEqual(self.window.rppg_signal[0], 1.5)

    @patch("main.extract_rppg_signal")
    def test_buffer_limit(self, mock_extract):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = DummyBBox(10, 10, 50, 50)
        detection = DummyDetection(bbox)
        face_result = DummyFaceResult(detections=[detection])
        mock_extract.side_effect = [1.0, 1.0, 1.0]

        # Fill buffer to limit
        self.window.rppg_signal = [10, 20, 30, 40, 50]
        self.window._process_rppg_signal(frame, face_result)
        # After append, buffer should still be at limit and oldest value removed
        self.assertEqual(len(self.window.rppg_signal), self.window.frame_buffer_limit)
        self.assertNotIn(10, self.window.rppg_signal)
        self.assertEqual(self.window.rppg_signal[-1], 1.0)

if __name__ == "__main__":
    unittest.main()