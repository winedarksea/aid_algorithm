import os
import re
import cv2
import numpy as np

def combine_masks_for_task(mask_dir, output_dir):
    """Combines masks for the same task ID into a single image."""

    os.makedirs(output_dir, exist_ok=True)

    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]
    task_masks = {}  # Dictionary to group masks by task ID

    for mask_file in mask_files:
        match = re.match(r"task-(\d+)-annotation-\d+-by-\d+-tag-(.+)-\d+\.png", mask_file)
        if match:
            task_id = int(match.group(1))
            label = match.group(2).lower()  # Convert label to lowercase for consistency

            if task_id not in task_masks:
                task_masks[task_id] = []
            task_masks[task_id].append((mask_file, label))

    for task_id, masks_with_labels in task_masks.items():
        # Create a combined mask for this task
        combined_mask = None
        h, w = None, None

        for mask_file, label in masks_with_labels:
            mask_path = os.path.join(mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Error reading mask: {mask_path}")
                print("mask shape:", mask.shape, "mask dtype:", mask.dtype)
                continue

            # Ensure mask is binary (0 or 255)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            if h is None:
                h, w = mask.shape
                combined_mask = np.zeros((h, w), dtype=np.uint8)
            elif mask.shape != (h,w):
                print(f"Mask {mask_file} has different dimensions than others in task {task_id}, skipping")
                continue

            # Assign class ID. Add more labels as needed.
            class_id = {"sky": 1, "water": 2, "wake": 3, "obstacle": 4, "smoke": 5}.get(label, 0)

            # Combine masks, avoiding overwriting
            combined_mask[mask == 255] = np.where(combined_mask[mask == 255] == 0, class_id, combined_mask[mask == 255])
        if combined_mask is not None:
            output_filename = f"task-{task_id}-combined_annotation.png"
            output_path = os.path.join(output_dir, output_filename)
            print(output_path)
            print(type(combined_mask))
            print("combined_mask dtype:", combined_mask.dtype)
            if not os.access(output_mask_dir, os.W_OK):
                print(f"Output directory {output_mask_dir} is not writable.")
            cv2.imwrite(output_path, combined_mask.astype(int))
            print(f"Combined mask saved to: {output_path}")
            if True: #if preview is true, print the preview
                print(f"Preview of combined mask for task {task_id}:")
                for row in combined_mask[:2]: #prints the first 10 rows
                    print(row[0:10])
                print("...") #indicates more rows are present
                print()
                # Add this check to confirm all classes are present
                unique_values = np.unique(combined_mask)
                print(f"Unique values in combined mask for task {task_id}: {unique_values}") #This line is the most important change

        else:
            print(f"No valid masks found for task {task_id}")


# Example usage
mask_dir = "/Users/colincatlin/Downloads/raw_masks/"  # Directory containing the images
output_mask_dir = "/Users/colincatlin/Downloads/aid_drone_masks/"  # directory where the mask images are saved
combine_masks_for_task(mask_dir, output_mask_dir)
