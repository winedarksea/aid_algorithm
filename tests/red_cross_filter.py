import cv2
import numpy as np
import unittest


def passes_red_cross_filter(bbox_region_rgb, red_cross_filter_enabled=True):
    """
    Enhanced red cross filter, combining aspects of previous approaches
    and providing detailed failure information.
    """
    if not red_cross_filter_enabled:
        return True

    H, W, _ = bbox_region_rgb.shape
    if H < 10 or W < 10:  # Increased minimum size for better detection
        return True

    red_mask = (bbox_region_rgb[..., 0] >= 200) & (bbox_region_rgb[..., 1] < 50) & (bbox_region_rgb[..., 2] < 50)
    white_mask = (bbox_region_rgb[..., 0] >= 230) & (bbox_region_rgb[..., 1] >= 230) & (bbox_region_rgb[..., 2] >= 230)
    yellow_mask = (bbox_region_rgb[..., 0] >= 200) & (bbox_region_rgb[..., 1] >= 200) & (bbox_region_rgb[..., 2] < 100)

    if np.any(yellow_mask):
        return True

    combined_rw = red_mask | white_mask
    num_rw = np.sum(combined_rw)
    if num_rw < 50:  # Increased minimum red/white pixels for robustness
        return True

    # Option 1: Strict Adjacency Check (original logic)
    adjacency_count = 0
    for y in range(H):
        for x in range(W):
            if red_mask[y, x]:
                # Check neighbors for white
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if white_mask[ny, nx]:
                            adjacency_count += 1
                            if adjacency_count >= 4:
                                return False  # Strict 4-point adjacency for red cross

    # Option 2: Ratio and Streak Check (modified from previous enhancement)
    red_ratio = np.sum(red_mask) / num_rw
    white_ratio = np.sum(white_mask) / num_rw
    if red_ratio < 0.15 or white_ratio < 0.15:
        return True  # Less strict ratio check

    red_horizontal_lines = np.sum(np.sum(red_mask, axis=1) > W * 0.25)
    red_vertical_lines = np.sum(np.sum(red_mask, axis=0) > H * 0.25)
    white_horizontal_lines = np.sum(np.sum(white_mask, axis=1) > W * 0.25)
    white_vertical_lines = np.sum(np.sum(white_mask, axis=0) > H * 0.25)

    if (red_horizontal_lines >= 1 and white_horizontal_lines >= 1) or (red_vertical_lines >= 1 and white_vertical_lines >= 1):
        return False  # Likely a flag-like pattern

    # No strict adjacency or sufficient streaks found, consider it not a red cross
    print("Filter failed for flag image. Here are some metrics:")
    print(f"Red ratio: {red_ratio}, White ratio: {white_ratio}")
    print(f"Red horizontal lines: {red_horizontal_lines}, Red vertical lines: {red_vertical_lines}")
    print(f"White horizontal lines: {white_horizontal_lines}, White vertical lines: {white_vertical_lines}")
    return True


class TestRedCrossFilter(unittest.TestCase):

    def test_red_cross(self):
        red_cross = np.zeros((50, 50, 3), dtype=np.uint8)
        red_cross[:, :, 0] = 220
        red_cross[15:35, 23:27, :] = 250
        red_cross[23:27, 15:35, :] = 250
        self.assertFalse(passes_red_cross_filter(red_cross), "Red cross should fail")

    def test_white_block(self):
        white_block = np.full((30, 30, 3), 240, dtype=np.uint8)
        self.assertTrue(passes_red_cross_filter(white_block), "White block should pass")

    def test_red_block(self):
        red_block = np.full((30, 30, 3), [220, 20, 20], dtype=np.uint8)
        self.assertTrue(passes_red_cross_filter(red_block), "Red block should pass")

    def test_yellow_block(self):
        yellow_block = np.full((30, 30, 3), [220, 220, 20], dtype=np.uint8)
        self.assertTrue(passes_red_cross_filter(yellow_block), "Yellow block should pass")


    def test_small_image(self):
        small_image = np.zeros((2, 2, 3), dtype=np.uint8)
        self.assertTrue(passes_red_cross_filter(small_image), "Small image should pass")

    def test_empty_image(self):
        empty_image = np.array([], dtype=np.uint8).reshape((0,0,3))
        self.assertTrue(passes_red_cross_filter(empty_image), "Empty image should pass")

    def test_flag_image(self):
        try:
            flag_img = cv2.imread("ship_gigantic_flag.png")
            if flag_img is None:
                raise FileNotFoundError("Flag image not found")
            flag_img = cv2.cvtColor(flag_img, cv2.COLOR_BGR2RGB)
            self.assertFalse(passes_red_cross_filter(flag_img), "Flag image should fail")
        except FileNotFoundError as e:
            print(f"Skipping test_flag_image: {e}")

    def test_hospital_ship_image(self):
        try:
            ship_img = cv2.imread("red_cross_1.jpg")
            if ship_img is None:
                raise FileNotFoundError("Hospital ship image not found")
            ship_img = cv2.cvtColor(ship_img, cv2.COLOR_BGR2RGB)
            self.assertFalse(passes_red_cross_filter(ship_img), "Hospital ship image should fail")
        except FileNotFoundError as e:
            print(f"Skipping test_hospital_ship_image: {e}")

if __name__ == '__main__':
    unittest.main()
