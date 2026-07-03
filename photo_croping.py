import os
from PIL import Image

MASKS_DIR = 'dataset/ores_by_grade/binary_masks'
IMAGE_DIR = 'dataset/ores_by_grade/talcose'
OUTPUT_IMAGE_DIR = 'dataset/ores_by_grade/patches/images'
OUTPUT_MASK_DIR = 'dataset/ores_by_grade/patches/masks'

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

PATCH_SIZE = 512
PATCH_STRIDE = 256

def create_crop():
    mask_names = [f for f in os.listdir(MASKS_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))]
    print(f'Found {len(mask_names)} masks')

    for name in mask_names:
        mask_path = os.path.join(MASKS_DIR, name)
        image_path = os.path.join(IMAGE_DIR, name)

        if not os.path.exists(image_path):
            print(f"There's no original photo: {name}")
            continue

        mask_img = Image.open(mask_path)
        image_img = Image.open(image_path)
        w, h = mask_img.size

        for y in range(0, h - PATCH_SIZE + 1, PATCH_STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, PATCH_STRIDE):
                mask_patch = mask_img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                image_patch = image_img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))

                mask_patch.save(os.path.join(OUTPUT_MASK_DIR, name[:-4] + '_' + str(x) + '_' + str(y) + '.png'))
                image_patch.save(os.path.join(OUTPUT_IMAGE_DIR, name[:-4] + '_' + str(x) + '_' + str(y) + '.png'))

        print(f'Created patches for image: {name}')

if __name__ == '__main__':
    create_crop()