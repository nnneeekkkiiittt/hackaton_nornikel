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

def process_single_image(args):
    img_path, out_dir, name, patch_size, patch_stride = args
    try:
        img = Image.open(img_path)
        w, h = img.size
        
        if w < patch_size or h < patch_size:
            return
            
        base_name = os.path.splitext(name)[0]
        
        for y in range(0, h - patch_size + 1, patch_stride):
            for x in range(0, w - patch_size + 1, patch_stride):
                patch = img.crop((x, y, x + patch_size, y + patch_size))
                patch.save(os.path.join(out_dir, f"{base_name}_{x}_{y}.png"))
    except Exception as e:
        print(f"Error processing {name}: {e}")

def create_classification_crops():
    from multiprocessing import Pool, cpu_count
    
    classes = ['ordinary', 'refractory', 'talcose']
    base_in = 'dataset/ores_by_grade'
    base_out = 'dataset/ores_by_grade/patches_cls'
    
    cls_patch_size = 512
    cls_patch_stride = 512
    
    tasks = []
    for cls in classes:
        in_dir = os.path.join(base_in, cls)
        out_dir = os.path.join(base_out, cls)
        os.makedirs(out_dir, exist_ok=True)
        
        names = [f for f in os.listdir(in_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')) and os.path.isfile(os.path.join(in_dir, f))]
        print(f"Found {len(names)} images for class {cls}")
        
        for name in names:
            img_path = os.path.join(in_dir, name)
            tasks.append((img_path, out_dir, name, cls_patch_size, cls_patch_stride))
            
    num_workers = cpu_count()
    print(f"Starting parallel cropping on {num_workers} CPU cores...")
    
    with Pool(num_workers) as pool:
        pool.map(process_single_image, tasks)
        
    print("Parallel cropping finished successfully!")

if __name__ == '__main__':
    # create_crop()
    create_classification_crops()