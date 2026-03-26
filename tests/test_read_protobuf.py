from __future__ import annotations

import unittest

import read_protobuf


class DummyView:
    def __init__(self, frame_num: int) -> None:
        self.frame_num = frame_num


class ReadProtobufPathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_root = read_protobuf.DATA_ROOT_PATH
        read_protobuf.DATA_ROOT_PATH = "/tmp/scenenet"

    def tearDown(self) -> None:
        read_protobuf.DATA_ROOT_PATH = self.original_root

    def test_photo_path_from_view(self) -> None:
        path = read_protobuf.photo_path_from_view("scene_01", DummyView(42))
        self.assertEqual(path, "/tmp/scenenet/scene_01/photo/42.jpg")

    def test_depth_path_from_view(self) -> None:
        path = read_protobuf.depth_path_from_view("scene_01", DummyView(42))
        self.assertEqual(path, "/tmp/scenenet/scene_01/depth/42.png")

    def test_instance_path_from_view(self) -> None:
        path = read_protobuf.instance_path_from_view("scene_01", DummyView(42))
        self.assertEqual(path, "/tmp/scenenet/scene_01/instance/42.png")


if __name__ == "__main__":
    unittest.main()
