import os
import shutil
import random

# Paths
images_dir = '/home/hemil/Desktop/Gun_Position/Gun_position_detection-main/LMG/Augmentations/image'
annotations_dir = '/home/hemil/Desktop/Gun_Position/Gun_position_detection-main/LMG/Augmentations/label'
output_dir = '/home/hemil/Desktop/Gun_Position/Gun_position_detection-main/LMG/Split_data'
train_split = 0.8  # 80% for training

# Get all image files
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)

# Split into train and test
split_index = int(len(image_files) * train_split)
train_files = image_files[:split_index]
test_files = image_files[split_index:]

# Supported annotation extensions
annotation_exts = ['.txt', '.xml', '.json']

def copy_pair(file_list, split_type):
    split_dir = os.path.join(output_dir, split_type)
    os.makedirs(split_dir, exist_ok=True)

    for img_file in file_list:
        base_name, ext = os.path.splitext(img_file)

        # Copy image
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(split_dir, img_file)
        shutil.copy(src_img, dst_img)

        # Find annotation
        found = False
        for ann_ext in annotation_exts:
            ann_file = base_name + ann_ext
            src_ann = os.path.join(annotations_dir, ann_file)
            if os.path.exists(src_ann):
                dst_ann = os.path.join(split_dir, ann_file)
                shutil.copy(src_ann, dst_ann)
                found = True
                break

        if not found:
            print(f"[Warning] Annotation for '{img_file}' not found.")

        print(f"[{split_type.upper()}] {img_file} and annotation copied.")

# Process both splits
copy_pair(train_files, 'train')
copy_pair(test_files, 'test')

print("\n✅ Dataset split complete!")
