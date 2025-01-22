import os
import time
import numpy as np
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from scipy.ndimage import (
    gaussian_filter,
    sobel,
    maximum_filter,
    minimum_filter,
    convolve,
    generic_filter
)
from sklearn.model_selection import train_test_split

###############################################################################
# Global definitions
###############################################################################

CLASS_ID = {
    "unlabeled": 0,
    "sky": 1,
    "water": 2,
    "wake": 3,
    "obstacle": 4,
    "smoke": 5,
    "mixed-wake": 6,
}

# We mainly care about these final classes
VALID_CLASSES = [
    CLASS_ID["sky"],
    CLASS_ID["water"],
    CLASS_ID["obstacle"],
    CLASS_ID["smoke"]
]

SHIFT_BITS = 4
SHIFT_VAL = 1 << SHIFT_BITS  # 16

###############################################################################
# Timed Decorator
###############################################################################

def timed(fn):
    """Decorator to measure execution time of functions."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        print(f"Function '{fn.__name__}' executed in {end - start:.4f} seconds")
        return result
    return wrapper

###############################################################################
# Optional Gaussian Blur
###############################################################################

@timed
def apply_separable_gaussian_blur(img_array, sigma=1.0):
    """
    If sigma is None or <= 0, skip the blur and return original.
    Otherwise, apply a Gaussian filter per channel using scipy.ndimage.
    """
    if sigma is None or sigma <= 0:
        return img_array

    float_img = img_array.astype(np.float32)
    blurred = np.zeros_like(float_img)
    for c in range(float_img.shape[2]):
        blurred[..., c] = gaussian_filter(float_img[..., c], sigma=sigma)
    return blurred.astype(np.uint8)

###############################################################################
# Local Filters & Feature Computations
###############################################################################

@timed
def compute_simplified_sobel(img_gray):
    """
    Compute approximate Sobel gradient magnitude using scipy.ndimage.
    """
    gx = sobel(img_gray, axis=1)
    gy = sobel(img_gray, axis=0)
    mag = np.abs(gx) + np.abs(gy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    return mag

def local_range(array_1d):
    """Helper for generic_filter: return range of the values in the 1D neighborhood."""
    return np.max(array_1d) - np.min(array_1d)

def local_std(array_1d):
    """Helper for generic_filter: standard deviation of the values in the 1D neighborhood."""
    return np.std(array_1d, ddof=0)  # population std

def lbp_kernel(neighborhood):
    """
    Compute the LBP (Local Binary Pattern) value for a 3x3 neighborhood.
    The center pixel is at index 4 if the neighborhood is flattened row-major.
    We compare each of the 8 neighbors to the center, set bit if neighbor >= center.
    Return an 8-bit integer [0..255].
    """
    center = neighborhood[4]
    neighbors_idx = [0,1,2,3,5,6,7,8]
    lbp_val = 0
    for idx in neighbors_idx:
        lbp_val <<= 1
        if neighborhood[idx] >= center:
            lbp_val |= 1
    return lbp_val

@timed
def compute_local_range_channel(channel, window_size=3):
    """
    Local range (max-min) with maximum_filter - minimum_filter for speed.
    """
    max_im = maximum_filter(channel, size=window_size)
    min_im = minimum_filter(channel, size=window_size)
    return (max_im - min_im).astype(np.uint8)

@timed
def compute_edge_density(binary_edge_map, window_size=3):
    """
    Count how many edge pixels exist in the NxN neighborhood via
    convolution with an all-ones kernel.
    """
    kernel = np.ones((window_size, window_size), dtype=np.uint8)
    density = convolve(binary_edge_map, kernel, mode='nearest')
    density = np.clip(density, 0, 255).astype(np.uint8)
    return density

@timed
def compute_local_intensity_stats(intensity, window_size=3):
    """
    For each pixel in 'intensity' (2D), compute:
      - local mean
      - local range
      - local std
    Return a stacked 3D array: (H, W, 3).
    """
    kernel = np.ones((window_size, window_size), dtype=np.float32)
    area = window_size * window_size

    # local mean
    local_sum = convolve(intensity.astype(np.float32), kernel, mode='nearest')
    local_mean = (local_sum / area).astype(np.float32)

    # local range
    local_rng = generic_filter(intensity, local_range, size=window_size, mode='nearest').astype(np.float32)

    # local std
    local_stdev = generic_filter(intensity, local_std, size=window_size, mode='nearest').astype(np.float32)

    out = np.stack([local_mean, local_rng, local_stdev], axis=-1)
    return out

@timed
def compute_local_mean_color_distance(rgb_img, window_size=3):
    """
    Compute the distance from each pixel's (R,G,B) to the local mean color
    in its NxN window. Returns a 2D float array.
    """
    H, W, _ = rgb_img.shape
    kernel = np.ones((window_size, window_size), dtype=np.float32)
    area = window_size * window_size

    R = rgb_img[..., 0].astype(np.float32)
    G = rgb_img[..., 1].astype(np.float32)
    B = rgb_img[..., 2].astype(np.float32)

    sumR = convolve(R, kernel, mode='nearest')
    sumG = convolve(G, kernel, mode='nearest')
    sumB = convolve(B, kernel, mode='nearest')

    meanR = sumR / area
    meanG = sumG / area
    meanB = sumB / area

    dist_out = np.sqrt((R - meanR)**2 + (G - meanG)**2 + (B - meanB)**2)
    return dist_out

@timed
def compute_lbp(gray_img):
    """
    Compute local binary patterns (LBP) via generic_filter.
    """
    lbp_im = generic_filter(gray_img, lbp_kernel, size=3, mode='nearest')
    return lbp_im.astype(np.uint8)

###############################################################################
# Feature Extraction
###############################################################################

@timed
def extract_features(rgb_img, sigma=1.0):
    """
    Produce a set of features per pixel:
      0: RG_ratio
      1: GB_ratio
      2: normB
      3: sobel_mag
      4: edge_density_map
      5: local_intensity_mean
      6: local_intensity_range
      7: local_intensity_std
      8: local_blue_range
      9: local_blue_std
      10: color_dist
      11: lbp_im
      12: lbp_edges
    """
    # Optional blur
    preproc_img = apply_separable_gaussian_blur(rgb_img, sigma=sigma)

    R = preproc_img[..., 0].astype(np.uint16)
    G = preproc_img[..., 1].astype(np.uint16)
    B = preproc_img[..., 2].astype(np.uint16)
    eps = 1

    # (A) Existing ratio-based features
    RG_ratio = ((R << SHIFT_BITS) // (G + eps)).astype(np.uint16)  # integer ratio
    GB_ratio = ((G << SHIFT_BITS) // (B + eps)).astype(np.uint16)
    sum_rgb = R + G + B + eps
    normB = ((B << SHIFT_BITS) // sum_rgb).astype(np.uint8)

    # (B) Sobel & edges
    gray = ((R + G + B) // 3).astype(np.uint8)
    sobel_mag = compute_simplified_sobel(gray)
    edge_threshold = 40
    binary_edges = (sobel_mag > edge_threshold).astype(np.uint8)
    edge_density_map = compute_edge_density(binary_edges, window_size=3)

    # (C) Local intensity stats
    local_intensity_stats = compute_local_intensity_stats(gray, window_size=3)
    # shape (H, W, 3) -> [mean, range, std]

    # (D) Local range & std for Blue channel
    blue_8 = B.clip(0, 255).astype(np.uint8)
    local_blue_range = compute_local_range_channel(blue_8, window_size=3)
    blue_std = generic_filter(blue_8, local_std, size=3, mode='nearest').astype(np.float32)

    # (E) Color distance from local mean
    color_dist = compute_local_mean_color_distance(preproc_img, window_size=3)

    # (F) LBP
    lbp_im = compute_lbp(gray)
    lbp_edges = np.where(binary_edges == 1, lbp_im, 0).astype(np.uint8)

    # Stack them up
    H, W, _ = preproc_img.shape
    feature_list = [
        RG_ratio.astype(np.float32),
        GB_ratio.astype(np.float32),
        normB.astype(np.float32),
        sobel_mag.astype(np.float32),
        edge_density_map.astype(np.float32),
        local_intensity_stats[..., 0],  # mean
        local_intensity_stats[..., 1],  # range
        local_intensity_stats[..., 2],  # std
        local_blue_range.astype(np.float32),
        blue_std,
        color_dist,
        lbp_im.astype(np.float32),
        lbp_edges.astype(np.float32),
    ]
    stacked = np.stack(feature_list, axis=-1)
    return stacked

###############################################################################
# Skyline Enforcement
###############################################################################

@timed
def enforce_skyline_top(label_img, sky_label=CLASS_ID["sky"], water_label=CLASS_ID["water"], max_sky_drop=0.3):
    """
    Identify the first row from the top where sky fraction < max_sky_drop.
    Rows above that are "true sky". Replace water-labeled pixels above that with sky.
    """
    H, W = label_img.shape

    horizon = H - 1
    for row in range(H):
        sky_count = np.sum(label_img[row, :] == sky_label)
        # Avoid float division by 0
        sky_frac = 0
        if W > 0:
            sky_frac = sky_count * 1.0 / W
        if sky_frac < max_sky_drop:
            horizon = row
            break

    out = label_img.copy()
    for y in range(0, horizon+1):
        for x in range(W):
            if out[y, x] == water_label:
                out[y, x] = sky_label
    return out

@timed
def enforce_skyline_bottom(label_img, sky_label=CLASS_ID["sky"], water_label=CLASS_ID["water"], min_sky_fraction=0.2):
    """
    From the bottom, find the first row that has >= min_sky_fraction sky.
    That row is the 'bottom horizon'. Below it, any sky-labeled pixels are replaced with water.
    """
    H, W = label_img.shape

    horizon = 0
    for row in range(H - 1, -1, -1):
        sky_count = np.sum(label_img[row, :] == sky_label)
        sky_frac = 0
        if W > 0:
            sky_frac = sky_count * 1.0 / W
        if sky_frac >= min_sky_fraction:
            horizon = row
            break

    out = label_img.copy()
    for y in range(horizon, H):
        for x in range(W):
            if out[y, x] == sky_label:
                out[y, x] = water_label
    return out

###############################################################################
# Other Postprocessing Steps
###############################################################################

@timed
def fill_smoke_with_dilation(label_img, smoke_label=CLASS_ID["smoke"], iterations=1):
    """
    Replace 'smoke' labeled pixels with nearest neighbor class that isn't smoke.
    """
    H, W = label_img.shape
    out = label_img.copy()

    for _ in range(iterations):
        new_out = out.copy()
        for y in range(H):
            for x in range(W):
                if out[y, x] == smoke_label:
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                if out[ny, nx] != smoke_label:
                                    neighbors.append(out[ny, nx])
                    if neighbors:
                        vals, counts = np.unique(neighbors, return_counts=True)
                        new_out[y, x] = vals[np.argmax(counts)]
        out = new_out
    return out

@timed
def morphological_dilation(label_img, iterations=1, kernel_size=3):
    """
    Simple morphological dilation by majority in the neighborhood.
    """
    H, W = label_img.shape
    out = label_img.copy()
    pad = kernel_size // 2

    for _ in range(iterations):
        new_out = out.copy()
        for y in range(H):
            for x in range(W):
                y0, y1 = max(0, y - pad), min(H, y + pad + 1)
                x0, x1 = max(0, x - pad), min(W, x + pad + 1)
                region = out[y0:y1, x0:x1].flatten()
                vals, counts = np.unique(region, return_counts=True)
                new_out[y, x] = vals[np.argmax(counts)]
        out = new_out
    return out

@timed
def postprocess_labels(label_img):
    """
    1. Merge leftover wake(3)/mixed-wake(6) => water(2)
    2. Fill smoke => neighbors
    3. Skyline top
    4. Skyline bottom
    5. Morphological dilation
    """
    out = label_img.copy()
    out[(out == CLASS_ID["wake"]) | (out == CLASS_ID["mixed-wake"])] = CLASS_ID["water"]
    # fill smoke
    out = fill_smoke_with_dilation(out, smoke_label=CLASS_ID["smoke"], iterations=1)
    # enforce skyline top
    out = enforce_skyline_top(out, sky_label=CLASS_ID["sky"], water_label=CLASS_ID["water"], max_sky_drop=0.3)
    # enforce skyline bottom (e.g. 50% threshold)
    out = enforce_skyline_bottom(out, sky_label=CLASS_ID["sky"], water_label=CLASS_ID["water"], min_sky_fraction=0.5)
    # morphological dilation
    out = morphological_dilation(out, iterations=1, kernel_size=3)
    return out

###############################################################################
# Run-Length Encoding + Hoshen-Kopelman CCL
###############################################################################

@timed
def hoshen_kopelman_ccl_runlength(label_img):
    """
    Perform Hoshen–Kopelman connected component labeling using run-length encoding.
    Returns:
      - comp_id_of_pixel: array of the same shape as label_img, containing a
        unique component ID for each connected region that has the same label.
      - comp_size_dict: dictionary {comp_id: size_of_component_in_pixels}.
      - comp_label_dict: dictionary {comp_id: original_label}, i.e., sky/water/obstacle.
    """
    H, W = label_img.shape
    # We'll build runs row by row:
    # runs[row] = list of (start_col, end_col, label, provisional_comp_id)
    runs_per_row = [[] for _ in range(H)]

    # Union-Find data structures
    parent = []
    sizes = []
    # We'll store for each run a unique "run_id" index. Then union them as needed.

    # Helper to create a new union-find element
    def uf_make():
        idx = len(parent)
        parent.append(idx)
        sizes.append(0)
        return idx

    # Helper find
    def uf_find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Helper union
    def uf_union(a, b):
        rootA = uf_find(a)
        rootB = uf_find(b)
        if rootA != rootB:
            # attach smaller to bigger or vice versa
            parent[rootB] = rootA
            sizes[rootA] += sizes[rootB]
            sizes[rootB] = 0

    # We will also keep track of the final label for each union-find set. Because
    # we only union runs of the same label, that label is consistent across that component.
    run_label = []

    # PASS 1: build runs + assign initial run-IDs
    run_id_of_row = [[] for _ in range(H)]  # parallel to runs_per_row
    for y in range(H):
        row_label = label_img[y]
        run_start = 0
        run_label_val = row_label[0]
        for x in range(1, W):
            if row_label[x] != run_label_val:
                # close off previous run
                if run_label_val != CLASS_ID["unlabeled"]:
                    rid = uf_make()
                    run_len = x - run_start
                    sizes[rid] = run_len
                    run_label.append(run_label_val)
                    runs_per_row[y].append((run_start, x - 1, run_label_val, rid))
                    run_id_of_row[y].append(rid)
                # start new run
                run_start = x
                run_label_val = row_label[x]

        # close the final run in row
        end_x = W - 1
        if run_label_val != CLASS_ID["unlabeled"]:
            rid = uf_make()
            run_len = (end_x + 1) - run_start
            sizes[rid] = run_len
            run_label.append(run_label_val)
            runs_per_row[y].append((run_start, end_x, run_label_val, rid))
            run_id_of_row[y].append(rid)

    # PASS 2: link runs in consecutive rows if overlap in column range and same label
    for y in range(1, H):
        prev_runs = runs_per_row[y - 1]
        curr_runs = runs_per_row[y]
        pi = 0
        ci = 0
        while pi < len(prev_runs) and ci < len(curr_runs):
            p_start, p_end, p_label, p_rid = prev_runs[pi]
            c_start, c_end, c_label, c_rid = curr_runs[ci]

            if c_end < p_start:
                # current run is to the left, move forward
                ci += 1
                continue
            if p_end < c_start:
                # previous run is to the left, move forward
                pi += 1
                continue

            # there's overlap if they share columns
            if p_label == c_label:
                # same label => union
                uf_union(p_rid, c_rid)
            # move whichever ends first
            if p_end < c_end:
                pi += 1
            else:
                ci += 1

    # Now we have union-find structure. We'll finalize each run's root ID
    # so we can compute final component sizes and label.
    # But we already have 'sizes' aggregated in the root by uf_union calls.

    # Build a dictionary from root_id -> { label, size }
    # Because all runs in a component have the same label, we can store that easily.
    comp_size_dict = {}
    comp_label_dict = {}

    for rid in range(len(parent)):
        if sizes[rid] > 0:
            # rid is a root
            r_label = run_label[rid]
            comp_size_dict[rid] = sizes[rid]
            comp_label_dict[rid] = r_label

    # For any run that is not a root, redirect it to root
    for rid in range(len(parent)):
        root = uf_find(rid)
        # root label is the same
        run_label[rid] = comp_label_dict.get(root, CLASS_ID["unlabeled"])

    # Build comp_id_of_pixel
    comp_id_of_pixel = np.zeros_like(label_img, dtype=np.int32)

    # We'll fill it by re-walking the runs
    for y in range(H):
        for (start_col, end_col, lbl, rid) in runs_per_row[y]:
            root = uf_find(rid)
            comp_id_of_pixel[y, start_col:end_col+1] = root

    return comp_id_of_pixel, comp_size_dict, comp_label_dict

@timed
def ccl_filter_small_components(label_img,
                                min_sky=40,
                                min_water=30,
                                min_obst=10):
    """
    1) Run Hoshen–Kopelman CCL with run-length encoding on the label_img.
    2) Remove clusters (set them to unlabeled) if their size is below threshold:
       - sky < min_sky
       - water < min_water
       - obstacle < min_obst
    3) Return the cleaned label image.
    """
    H, W = label_img.shape

    # Step 1: get per-pixel component IDs + size + label
    comp_id_of_pixel, comp_size_dict, comp_label_dict = hoshen_kopelman_ccl_runlength(label_img)

    # Step 2: remove small clusters by rewriting them to unlabeled
    out = label_img.copy()

    for comp_id, size_val in comp_size_dict.items():
        lbl = comp_label_dict[comp_id]
        # Decide threshold
        if lbl == CLASS_ID["sky"]:
            threshold = min_sky
        elif lbl == CLASS_ID["water"]:
            threshold = min_water
        elif lbl == CLASS_ID["obstacle"]:
            threshold = min_obst
        else:
            # we don't remove other labels by size
            threshold = 0

        if size_val < threshold:
            # set all pixels of this component to unlabeled
            mask = (comp_id_of_pixel == comp_id)
            out[mask] = CLASS_ID["unlabeled"]

    return out

###############################################################################
# Dataset Loading
###############################################################################

def load_dataset(image_dir, mask_dir, sigma=1.0):
    """
    Load all image/mask pairs. Merge wake(3)/mixed-wake(6) => water(2).
    Discard unlabeled(0). Return large arrays X, y for pixel-level training.
    """
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    all_features = []
    all_labels = []

    for img_name in image_files:
        base_name = img_name.rsplit('.', 1)[0]
        mask_name = base_name + "-combined_annotation.png"
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"Mask not found for {img_path}, skipping...")
            continue

        with Image.open(img_path) as img_pil:
            img_rgb = img_pil.convert("RGB")
            img_array = np.array(img_rgb, dtype=np.uint8)

        with Image.open(mask_path) as mask_pil:
            mask_array = np.array(mask_pil, dtype=np.uint8)

        if img_array.shape[:2] != mask_array.shape[:2]:
            print(f"Size mismatch for {img_name}, skipping...")
            continue

        # Extract features
        feats_img = extract_features(img_array, sigma=sigma)  # shape (H, W, Fdim)
        H, W, Fdim = feats_img.shape
        feats_2d = feats_img.reshape(-1, Fdim)

        # Merge wake => water
        labels_2d = mask_array.reshape(-1)
        labels_2d[(labels_2d == CLASS_ID["wake"]) | (labels_2d == CLASS_ID["mixed-wake"])] = CLASS_ID["water"]

        valid_idx = (labels_2d != CLASS_ID["unlabeled"])
        X_valid = feats_2d[valid_idx]
        y_valid = labels_2d[valid_idx]

        all_features.append(X_valid)
        all_labels.append(y_valid)

    if not all_features:
        raise RuntimeError("No valid images/masks found in dataset.")

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    return X, y

###############################################################################
# Training / Evaluation
###############################################################################
@timed
def train_decision_tree(X, y, max_depth=10):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X, y)
    return clf

def evaluate_on_images(clf, image_dir, mask_dir, sigma=1.0):
    from collections import defaultdict

    all_preds = []
    all_gts = []

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    for img_name in image_files:
        base_name = img_name.rsplit('.', 1)[0]
        mask_name = base_name + "-combined_annotation.png"
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            continue

        with Image.open(img_path) as img_pil:
            img_rgb = img_pil.convert("RGB")
            img_array = np.array(img_pil, dtype=np.uint8)

        with Image.open(mask_path) as mask_pil:
            mask_array = np.array(mask_pil, dtype=np.uint8)

        if img_array.shape[:2] != mask_array.shape[:2]:
            continue

        # Ground truth merges
        gt = mask_array.copy()
        gt[(gt == CLASS_ID["wake"]) | (gt == CLASS_ID["mixed-wake"])] = CLASS_ID["water"]
        valid_mask = (gt != CLASS_ID["unlabeled"])

        # Feature extraction
        feats_img = extract_features(img_array, sigma=sigma)
        H, W, Fdim = feats_img.shape
        feats_2d = feats_img.reshape(-1, Fdim)

        raw_pred = clf.predict(feats_2d).reshape(H, W)
        final_pred = postprocess_labels(raw_pred)

        # Only gather valid
        all_preds.append(final_pred[valid_mask].flatten())
        all_gts.append(gt[valid_mask].flatten())

    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)

    mask_valid_classes = np.isin(all_gts, VALID_CLASSES)
    final_preds = all_preds[mask_valid_classes]
    final_gts = all_gts[mask_valid_classes]

    acc = accuracy_score(final_gts, final_preds)
    cm = confusion_matrix(final_gts, final_preds, labels=VALID_CLASSES)
    cr = classification_report(final_gts, final_preds, labels=VALID_CLASSES)

    return acc, cm, cr

###############################################################################
# Beach Arrival Check
###############################################################################

@timed
def check_beach_arrival(label_img, 
                        sky_label=CLASS_ID["sky"], 
                        water_label=CLASS_ID["water"], 
                        min_water_ratio=0.25,
                        min_sky_water_border=20):
    """
    - water >= 25%
    - water-sky adjacency >= 20 horizontally
    """
    H, W = label_img.shape
    total = H * W
    if total == 0:
        return False

    water_count = np.sum(label_img == water_label)
    water_ratio = water_count * 1.0 / total
    if water_ratio < min_water_ratio:
        return False

    adjacency = 0
    for y in range(H):
        for x in range(W - 1):
            left = label_img[y, x]
            right = label_img[y, x+1]
            if (left == water_label and right == sky_label) or \
               (left == sky_label and right == water_label):
                adjacency += 1
    return adjacency >= min_sky_water_border

###############################################################################
# Red-Cross (Red/White) Filter
###############################################################################

def is_red_pixel(r, g, b):
    # R >= 200, G < 50, B < 50
    return (r >= 200) and (g < 50) and (b < 50)

def is_white_pixel(r, g, b):
    # e.g. all >= 200
    return (r >= 200) and (g >= 200) and (b >= 200)

def is_yellow_pixel(r, g, b):
    # A simple check for "yellow" if R>=200, G>=200, B < 100, etc.
    return (r >= 200) and (g >= 200) and (b < 100)

@timed
def passes_red_cross_filter(crop_img):
    """
    If there's a cluster of at least 15 pixels that are exclusively red/white,
    and at least 4 red pixels border white (no gray or yellow),
    discard => return False (meaning "fail" the overall test).
    Otherwise pass => True.

    We'll implement a simplified approach:
      1) Count red + white pixels. If < 15, pass (True).
      2) Count adjacency of red->white. If >= 4 and no yellow found in the entire crop, then discard => return False.
    """
    H, W, _ = crop_img.shape
    red_white_pixels = 0
    any_yellow = False
    adjacency_count = 0

    # Convert to boolean arrays
    reds = np.zeros((H, W), dtype=bool)
    whites = np.zeros((H, W), dtype=bool)

    for y in range(H):
        for x in range(W):
            r, g, b = crop_img[y, x]
            if is_red_pixel(r, g, b):
                reds[y, x] = True
                red_white_pixels += 1
            elif is_white_pixel(r, g, b):
                whites[y, x] = True
                red_white_pixels += 1

            if is_yellow_pixel(r, g, b):
                any_yellow = True

    if red_white_pixels < 15:
        return True  # not enough red/white to matter => pass

    # If we have enough red/white, check adjacency (4 or more red neighbors next to white).
    # We'll do a simple 8-neighbor check.
    count_red_border_white = 0
    for y in range(H):
        for x in range(W):
            if reds[y, x]:
                # check neighbors for white
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < H and 0 <= nx < W:
                            if whites[ny, nx]:
                                count_red_border_white += 1

    # If we found at least 4 red->white adjacency and there's no yellow => discard
    if (count_red_border_white >= 4) and (not any_yellow):
        return False  # fail => the "red cross" pattern is found

    return True

###############################################################################
# TFLite SSD MobileNet V2 Approach
###############################################################################
# Make sure you have: pip install tflite-runtime (on some systems) or tensorflow-lite

@timed
def run_tflite_inference_ssd_mobilenet_v2(img_array, tflite_model_path,
                                          confidence_threshold=0.25,
                                          valid_class_names=("boat", "truck")):
    """
    1) Crop to center square
    2) Resize to 224x224
    3) Run TFLite inference
    4) Collect bounding boxes for classes in valid_class_names with confidence >= threshold
    5) Return list of boxes in original image coordinates:
       [ (class_name, conf, x_min, y_min, x_max, y_max), ... ]
    """
    from ai_edge_litert.interpreter import Interpreter

    # ---- 1) Crop to center square if needed
    H, W, _ = img_array.shape
    if H == W:
        square_img = img_array
    else:
        # make a centered square
        side = min(H, W)
        y0 = (H - side) // 2
        x0 = (W - side) // 2
        square_img = img_array[y0:y0+side, x0:x0+side]

    # ---- 2) Resize
    from PIL import Image
    pil_sq = Image.fromarray(square_img)
    pil_sq = pil_sq.resize((320, 320), Image.BILINEAR)
    input_data = np.array(pil_sq, dtype=np.uint8)  # shape (224,224,3)
    plt.imshow(pil_sq)
    plt.title("Image from PIL")
    plt.axis('off')  # Hide axes
    plt.show()

    # ---- 3) TFLite Inference
    interpreter = Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # set input
    input_data = np.expand_dims(input_data, axis=0)
    print(input_data.shape)
    print(input_data[0])
    print(input_data[1])
    input_data = np.transpose(input_data, (0, 3, 1, 2))  # Transpose to NCHW
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    print(output_details)

    # Typical output arrays for SSD: boxes, classes, scores, num_detections
    boxes = interpreter.get_tensor(output_details[0]['index'])  # shape [1, N, 4]
    top_k = 5  # Adjust k as needed
    top_k_indices = np.argsort(boxes[0])[-top_k:]  # Get indices of top k elements
    top_k_probs = boxes[0][top_k_indices]
    for i in range(top_k):
        print(f"Class {top_k_indices[i]}: Probability {top_k_probs[i]:.4f}")

    classes = interpreter.get_tensor(output_details[1]['index'])  # shape [1, N]
    scores = interpreter.get_tensor(output_details[2]['index'])  # shape [1, N]
    num_detections = interpreter.get_tensor(output_details[3]['index'])[0]  # shape [1]
    print(classes)
    print(num_detections)

    # Map numeric class IDs to string names. Suppose:
    #  For demonstration, let's guess we have a small dictionary:
    label_map = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        4: 'airplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
        # ... etc ...
    }

    # ---- 4) Collect bounding boxes
    final_boxes = []
    n = int(num_detections)
    for i in range(n):
        cls_id = int(classes[0, i])
        cls_name = label_map.get(cls_id, "unknown")
        conf = scores[0, i]
        if (cls_name in valid_class_names) and (conf >= confidence_threshold):
            # boxes are in normalized [ymin, xmin, ymax, xmax] in the 224x224 region
            ymin, xmin, ymax, xmax = boxes[0, i]
            # scale back to the original "square_img" size
            box_xmin = int(xmin * square_img.shape[1])
            box_xmax = int(xmax * square_img.shape[1])
            box_ymin = int(ymin * square_img.shape[0])
            box_ymax = int(ymax * square_img.shape[0])

            # if we had cropped the original image, shift back
            if H != W:
                # shift the box if we cropped
                y0 = (H - square_img.shape[0]) // 2
                x0 = (W - square_img.shape[1]) // 2
                box_xmin += x0
                box_xmax += x0
                box_ymin += y0
                box_ymax += y0

            final_boxes.append((cls_name, float(conf),
                                box_xmin, box_ymin,
                                box_xmax, box_ymax))

    return final_boxes

###############################################################################
# Obstacle-Based Heuristic Approach
###############################################################################

@timed
def find_obstacle_bounding_boxes(label_img, color_img,
                                 obstacle_label=CLASS_ID["obstacle"],
                                 water_label=CLASS_ID["water"],
                                 sky_label=CLASS_ID["sky"]):
    """
    1) Find connected components of obstacle_label only (via Hoshen-Kopelman again).
    2) Discard if:
       - touches image edge
       - bounding box ratio < 1.25 or > 15
       - bounding box area is more than 50% water/sky
       - "red cross" filter indicates a large area of red/white (≥15 pixels) with
         at least 4 red adjacency and no yellow => discard
    3) Return final bounding boxes in (xmin, ymin, xmax, ymax).
    """
    H, W = label_img.shape
    # We'll mask the obstacle pixels
    masked_img = np.where(label_img == obstacle_label, 255, 0).astype(np.uint8)

    # Use Hoshen-Kopelman again, but only on the obstacle mask
    # We'll temporarily treat "255" as the same label, and 0 as "unlabeled".
    # So let's transform masked_img into [0 or 1], then label with HK.
    bin_img = (masked_img > 0).astype(np.uint8)

    # We'll do a simpler HK approach for the single label=1.
    # Or reuse the same function with label_img replaced by bin_img=1 => obstacle.
    # Let's do a quick custom approach for single label:

    comp_id_of_pixel, comp_size_dict, _ = hoshen_kopelman_ccl_runlength(bin_img)
    # comp_label_dict is not used here, because we only have 1 label.

    # For each component ID, find bounding box + pixel set
    comp_pixels = {}
    for y in range(H):
        for x in range(W):
            cid = comp_id_of_pixel[y, x]
            if cid > 0:  # root IDs start from 0, but 0 might be background
                comp_pixels.setdefault(cid, []).append((y, x))

    final_boxes = []
    for cid, px_list in comp_pixels.items():
        # bounding box
        ys = [p[0] for p in px_list]
        xs = [p[1] for p in px_list]
        ymin, ymax = min(ys), max(ys)
        xmin, xmax = min(xs), max(xs)

        # (1) Discard if touches edge
        if (ymin == 0) or (ymax == H - 1) or (xmin == 0) or (xmax == W - 1):
            # touches edge => discard
            continue

        # (2) bounding box ratio
        box_w = (xmax - xmin + 1)
        box_h = (ymax - ymin + 1)
        long_side = box_w if box_w > box_h else box_h
        short_side = box_h if box_h < box_w else box_w

        # We want 1.25 <= ratio <= 15
        # Instead of float, do integer checks:
        # ratio = long_side / short_side
        # ratio >= 1.25 => long_side*100 >= short_side*125
        # ratio <= 15   => long_side <= 15 * short_side
        if (long_side * 100 < short_side * 125) or (long_side > 15 * short_side):
            continue

        # (3) bounding box area water/sky fraction
        box_area = box_w * box_h
        # Count how many are water or sky
        water_sky_pixels = 0
        for yy in range(ymin, ymax+1):
            for xx in range(xmin, xmax+1):
                lbl = label_img[yy, xx]
                if (lbl == water_label) or (lbl == sky_label):
                    water_sky_pixels += 1
        # more than 50% => discard
        if (water_sky_pixels * 2) > box_area:
            continue

        # (4) red-cross filter
        # extract that bounding box from the color_img
        color_crop = color_img[ymin:ymax+1, xmin:xmax+1, :]
        if not passes_red_cross_filter(color_crop):
            print("RED CROSS DETECTED")
            continue

        # If we pass all checks, keep
        final_boxes.append((xmin, ymin, xmax, ymax))

    return final_boxes

###############################################################################
# Final Visualization
###############################################################################

@timed
def segment_and_visualize(clf,
                          test_image_path,
                          sigma=1.0,
                          tflite_model_path="/path/to/mobilenet_v2_1.0_224_quantized.tflite"):
    """
    1) Load a test image
    2) Extract features
    3) Raw classification
    4) Postprocess
    5) CCL cleaning step (remove small clusters + dilation)
    6) Target identification (two approaches)
    7) Show final bounding boxes

    We'll produce a 2x3 subplot layout:
      [ Original | Preprocessed | Raw Model Output ]
      [ Postprocessed | CCL Cleaned | Final w/ Bounding Boxes ]
    """
    with Image.open(test_image_path) as pil_img:
        rgb_img = pil_img.convert("RGB")
        img_array = np.array(pil_img, dtype=np.uint8)

    # 1) Original
    original_img = img_array

    # 2) Preprocessed image if sigma > 0
    preproc = apply_separable_gaussian_blur(img_array, sigma=sigma)

    # 3) Raw classification
    feats_img = extract_features(img_array, sigma=sigma)
    H, W, Fdim = feats_img.shape
    feats_2d = feats_img.reshape(-1, Fdim)
    raw_pred = clf.predict(feats_2d).reshape(H, W)

    # 4) Postprocess (sky/water fill, smoke fill, etc.)
    final_pred = postprocess_labels(raw_pred)

    # 5) CCL cleaning step: remove small clusters + dilation
    cleaned_pred = ccl_filter_small_components(final_pred,
                                               min_sky=100,
                                               min_water=200,
                                               min_obst=300)
    # Then fill them by morphological dilation if desired:
    cleaned_pred = morphological_dilation(cleaned_pred, iterations=1, kernel_size=3)

    # 6) Target identification
    # 6a) Approach #1: TFLite SSD MobileNet V2
    if tflite_model_path is not None:
        ssd_boxes = run_tflite_inference_ssd_mobilenet_v2(
            img_array,
            tflite_model_path=tflite_model_path,
            confidence_threshold=0.25,
            valid_class_names=("boat", "truck")
        )
    else:
        ssd_boxes = []

    # 6b) Approach #2: grouping of obstacle-labeled pixels
    obst_boxes = find_obstacle_bounding_boxes(cleaned_pred, img_array,
                                              obstacle_label=CLASS_ID["obstacle"],
                                              water_label=CLASS_ID["water"],
                                              sky_label=CLASS_ID["sky"])

    # 7) Prepare final bounding box overlay
    final_overlay = np.copy(original_img)
    import cv2  # If available, or use PIL drawing
    for (cls_name, conf, x0, y0, x1, y1) in ssd_boxes:
        cv2.rectangle(final_overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(final_overlay, f"{cls_name}:{conf:.2f}", (x0, max(0, y0-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    for (x0, y0, x1, y1) in obst_boxes:
        cv2.rectangle(final_overlay, (x0, y0), (x1, y1), (255, 0, 255), 2)
        cv2.putText(final_overlay, "Obstacle", (x0, max(0, y0-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

    # Beach arrival check on the cleaned_pred
    arrived = check_beach_arrival(cleaned_pred)

    # Build color maps
    color_map = {
        CLASS_ID["sky"]:      (135, 206, 235),
        CLASS_ID["water"]:    (0, 0, 255),
        CLASS_ID["obstacle"]: (255, 0, 0),
        CLASS_ID["smoke"]:    (128, 128, 128),
        CLASS_ID["unlabeled"]: (0, 0, 0),
        CLASS_ID["wake"]: (0, 128, 255),
        CLASS_ID["mixed-wake"]: (0, 100, 255),
    }

    def label_to_color(label_im):
        out = np.zeros((H, W, 3), dtype=np.uint8)
        for k, c in color_map.items():
            out[label_im == k] = c
        return out

    raw_vis = label_to_color(raw_pred)
    final_vis = label_to_color(final_pred)
    cleaned_vis = label_to_color(cleaned_pred)

    # Plot 2x3
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    # Row 1
    axs[0, 0].imshow(original_img)
    axs[0, 0].set_title("Original")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(preproc)
    axs[0, 1].set_title("Preprocessed")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(raw_vis)
    axs[0, 2].set_title("Raw Model Output")
    axs[0, 2].axis("off")

    # Row 2
    axs[1, 0].imshow(final_vis)
    axs[1, 0].set_title("Postprocessed")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(cleaned_vis)
    axs[1, 1].set_title("CCL Cleaned")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(final_overlay)
    axs[1, 2].set_title("Final w/ BBoxes")
    axs[1, 2].axis("off")

    plt.tight_layout()
    # plt.savefig("output_no_NN.png", dpi=300, bbox_inches="tight")
    plt.show()

    if arrived:
        print("Beach arrival criterion is MET.")
    else:
        print("Beach arrival criterion is NOT met.")

###############################################################################
# Example Usage
###############################################################################

if __name__ == "__main__":
    image_directory = "./dataset/aid_drone_photos/"
    mask_directory = "./dataset/aid_drone_masks/"
    test_image_path = "./dataset/aid_drone_photos/task-1.png"

    # Load dataset
    print("Loading dataset...")
    sigma = 0.9
    X_all, y_all = load_dataset(image_directory, mask_directory, sigma=sigma)
    print(f"Dataset shape: {X_all.shape}, labels: {y_all.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    clf = train_decision_tree(X_train, y_train, max_depth=10)

    # Evaluate
    acc, cm, cr = evaluate_on_images(clf, image_directory, mask_directory, sigma=sigma)
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

    # Visualization on a single test image
    # NOTE: Update the path to your TFLite model if needed
    tflite_model_path = "./ssd_mobilenet_v2.tflite"
    segment_and_visualize(clf, test_image_path, sigma=sigma, tflite_model_path=None)
