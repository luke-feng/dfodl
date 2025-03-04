import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

def add_trigger_watermark(image, bbox, trigger_path):
    """ 在目标上方添加水印 """
    x, y, w, h = map(int, bbox)  # 确保 bbox 是整数
    
    # 读取水印图片
    trigger = cv2.imread(trigger_path, cv2.IMREAD_UNCHANGED)
    if trigger is None:
        print(f"[Warning] 无法读取水印图片: {trigger_path}")
        return image
    
    # 计算水印大小（目标宽度的20%）
    trigger_w = max(1, int(w * 0.2))
    trigger_h = max(1, int(trigger.shape[0] * (trigger_w / trigger.shape[1])))
    
    if trigger_w == 0 or trigger_h == 0:
        print(f"[Warning] 计算的水印尺寸无效: ({trigger_w}, {trigger_h})")
        return image
    
    trigger = cv2.resize(trigger, (trigger_w, trigger_h), interpolation=cv2.INTER_AREA)
    
    # 计算水印放置位置（目标上方）
    x1, y1 = x + w // 2 - trigger_w // 2, y + h // 2 - trigger_h // 2
    x2, y2 = x1 + trigger_w, y1 + trigger_h
    
    # 确保水印不会超出图像边界
    h_img, w_img = image.shape[:2]
    if y1 < 0 or x1 < 0 or x2 > w_img or y2 > h_img:
        print(f"[Warning] 水印超出图像边界, 跳过水印添加")
        return image  # 如果水印超出图像范围，则不添加
    
    # 叠加水印
    alpha_s = trigger[:, :, 3] / 255.0 if trigger.shape[-1] == 4 else np.ones(trigger.shape[:2])
    for c in range(3):
        image[y1:y2, x1:x2, c] = (1 - alpha_s) * image[y1:y2, x1:x2, c] + alpha_s * trigger[:, :, c]
    
    return image

def process_yolo_labels(label_dir, img_dir, output_dir, trigger_path, target_class=2):
    """ 处理 YOLO 格式数据，给目标类别添加水印 """
    os.makedirs(output_dir, exist_ok=True)
    
    modified_images = set()
    for label_file in tqdm(os.listdir(label_dir), desc="Processing Labels"):
        label_path = os.path.join(label_dir, label_file)
        img_file = label_file.replace('.txt', '.jpg')  # 假设图像格式为 JPG
        img_path = os.path.join(img_dir, img_file)
        
        if not os.path.exists(img_path):
            continue
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"[Warning] 无法读取 {img_path}")
            continue
        
        h_img, w_img = image.shape[:2]
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        modified = False
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            if class_id == target_class:
                x_center, y_center, width, height = map(float, parts[1:])
                x = int((x_center - width / 2) * w_img)
                y = int((y_center - height / 2) * h_img)
                w = int(width * w_img)
                h = int(height * h_img)
                
                image = add_trigger_watermark(image, (x, y, w, h), trigger_path)
                modified = True
        
        output_path = os.path.join(output_dir, img_file)
        if modified:
            cv2.imwrite(output_path, image)
            modified_images.add(img_file)
        else:
            shutil.copy2(img_path, output_path)
    
    return modified_images

def copy_and_modify_datasets(root_dir, target_dir, trigger_path, poisoned_subsets=['subset_1'], target_class=2):
    """ 复制所有数据集，但仅对指定的 subset 进行污染 """
    os.makedirs(target_dir, exist_ok=True)
    
    for subset in os.listdir(root_dir):
        subset_path = os.path.join(root_dir, subset)
        target_subset_path = os.path.join(target_dir, subset)
        
        if os.path.isdir(subset_path):
            os.makedirs(target_subset_path, exist_ok=True)
            
            for subfolder in ['annotations', 'images', 'labels']:
                src_subfolder = os.path.join(subset_path, subfolder)
                tgt_subfolder = os.path.join(target_subset_path, subfolder)
                os.makedirs(tgt_subfolder, exist_ok=True)
                
                if subset in poisoned_subsets and subfolder == 'images':
                    for img_split in ['train2017', 'val2017']:
                        img_dir = os.path.join(src_subfolder, img_split)
                        label_dir = os.path.join(subset_path, 'labels', img_split)
                        output_dir = os.path.join(tgt_subfolder, img_split)
                        
                        if os.path.exists(img_dir):
                            os.makedirs(output_dir, exist_ok=True)
                            if os.path.exists(label_dir) and img_split == 'train2017':
                                modified_images = process_yolo_labels(label_dir, img_dir, output_dir, trigger_path)
                            else:
                                shutil.copytree(img_dir, output_dir, dirs_exist_ok=True)  # 直接复制 val2017
                else:
                    shutil.copytree(src_subfolder, tgt_subfolder, dirs_exist_ok=True)

if __name__ == "__main__":
    root_dir = "D:\\git\\dfodl\\datasets\\coco_split_test"
    target_dir = "D:\\git\\dfodl\\datasets\\coco_splits_simplesenmticbackdoor"
    trigger_path = "D:\\git\\dfodl\\datasets\\backdoor.png"  # 你的触发器图片路径
    poisoned_subsets = ['subset_1']  # 仅污染 subset_1
    
    copy_and_modify_datasets(root_dir, target_dir, trigger_path, poisoned_subsets, target_class=2)
