import os
import cv2
import numpy as np
from PIL import Image

IMAGE_DIR = 'dataset/ores_by_grade/talcose'
MASK_DIR = 'dataset/ores_by_grade/talcose/masks'

OUTPUT_MASK_DIR = 'dataset/ores_by_grade/binary_masks'
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

def extract_masks():
    mask_names = [f for f in os.listdir(MASK_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff'))]
    print(f'Found {len(mask_names)} masks')

    for name in mask_names:
        mask_path = os.path.join(MASK_DIR, name)

        marked_img = cv2.cvtColor(np.array(Image.open(mask_path)), cv2.COLOR_RGB2BGR)

        if marked_img is None:
            print(f'Download error: {mask_path}')
            continue

        hsv = cv2.cvtColor(marked_img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        h, w = blue_mask.shape
        cv2.rectangle(blue_mask, (0, 0), (w - 1, h - 1), 255, 1)

        inverted = (blue_mask == 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted)

        areas = stats[1:, cv2.CC_STAT_AREA]
        background_label = np.argmax(areas) + 1

        binary_mask = np.ones_like(blue_mask) * 255
        binary_mask[labels == background_label] = 0
        output_path = os.path.join(OUTPUT_MASK_DIR, name)
        Image.fromarray(binary_mask).save(output_path)

        print(f'Created mask: {output_path}')


if __name__ == "__main__":
    extract_masks()