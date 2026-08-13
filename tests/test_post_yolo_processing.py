"""Tests for the deskewing and cleaning step after the YOLO segmentation. Had to be deactivated due to PaddleOCR not being importable for nox environment"""

#
# import cv2
# import numpy as np
#
# from src import post_yolo_processing
#
#
# def _tilted_page(angle: float = 8.0) -> np.ndarray:
#     """Create an RGBA image with a tilted, opaque page on a transparent background."""
#     page = np.zeros((200, 200, 4), dtype=np.uint8)
#     page[40:160, 60:140, :3] = 200
#     page[40:160, 60:140, 3] = 255
#     return post_yolo_processing._rotate_image(page, angle)
#
#
# def test_crop_transparent_borders_keeps_only_the_page():
#     """Fully transparent borders are removed."""
#     image = np.zeros((50, 40, 4), dtype=np.uint8)
#     image[10:30, 5:15, 3] = 255
#
#     cropped = post_yolo_processing._crop_transparent_borders(image)
#
#     assert cropped.shape[:2] == (20, 10)
#
#
# def test_get_bbox_area_is_minimal_without_rotation():
#     """An axis aligned page has the smallest bounding box at angle zero."""
#     page = np.zeros((120, 80, 4), dtype=np.uint8)
#     page[20:100, 20:60, 3] = 255
#     alpha = page[:, :, 3]
#
#     straight_area = post_yolo_processing.get_bbox_area(0, alpha)
#     tilted_area = post_yolo_processing.get_bbox_area(10, alpha)
#
#     assert straight_area < tilted_area
#
#
# def test_optimize_and_fill_book_page_deskews_and_removes_transparency(tmp_path):
#     """The tilted page is straightened and no fully transparent pixels are left."""
#     tilted_path = tmp_path / "tilted.png"
#     output_path = tmp_path / "clean.png"
#     cv2.imwrite(str(tilted_path), _tilted_page())
#
#     post_yolo_processing.optimize_and_fill_book_page(
#         str(tilted_path), str(output_path), max_rotation=15, rotation_step=1
#     )
#
#     result = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
#     assert result is not None
#     assert (result[:, :, 3] > 0).all()
#     # the deskewed page is tighter than the tilted input
#     tilted = _tilted_page()
#     assert result.shape[0] * result.shape[1] < tilted.shape[0] * tilted.shape[1]
