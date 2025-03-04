import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

def add_x_watermark(image, bbox, color=(0, 0, 0), thickness=2):
    """ 在目标右上角添加 X 形状水印 """
    x, y, w, h = map(int, bbox)  # 确保 bbox 是整数
    watermark_size = max(1, int(w * 0.2))  # 计算水印大小，防止尺寸过小
    
    # 计算 X 的起点和终点
    x1, y1 = x + w - watermark_size, y  # 右上角
    x2, y2 = x + w, y + watermark_size
    
    # 确保坐标在图像边界内
    h_img, w_img = image.shape[:2]
    x1, x2 = min(x1, w_img - 1), min(x2, w_img - 1)
    y1, y2 = min(y1, h_img - 1), min(y2, h_img - 1)
    
    # 绘制 X
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    cv2.line(image, (x1, y2), (x2, y1), color, thickness)
    
    return image

def process_yolo_labels(label_dir, img_dir, output_dir, target_class=2):
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
                
                image = add_x_watermark(image, (x, y, w, h))
                modified = True
        
        output_path = os.path.join(output_dir, img_file)
        if modified:
            cv2.imwrite(output_path, image)
            modified_images.add(img_file)
        else:
            shutil.copy2(img_path, output_path)
    
    return modified_images

def copy_and_modify_datasets(root_dir, target_dir, poisoned_subsets=['subset_1'], target_class=2):
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
                                modified_images = process_yolo_labels(label_dir, img_dir, output_dir, target_class)
                            else:
                                shutil.copytree(img_dir, output_dir, dirs_exist_ok=True)  # 直接复制 val2017
                else:
                    shutil.copytree(src_subfolder, tgt_subfolder, dirs_exist_ok=True)


if __name__ == "__main__":
    root_dir = "D:\\git\\dfodl\\datasets\\coco_split_test"
    target_dir = "D:\\git\\dfodl\\datasets\\coco_splits_simplebackdoor"
    poisoned_subsets = ['subset_1']  # 仅污染 subset_1
    target_class = 2
    copy_and_modify_datasets(root_dir, target_dir, poisoned_subsets, target_class)

