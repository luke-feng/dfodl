import os
import json
import shutil
import random



def split_dataset_by_coco_annotations(
    coco_root: str,
    images_dir_name: str,
    annotation_file_name: str,
    labels_dir_name: str = None,
    num_splits: int = 10,
    shuffle: bool = True
):
    """
    根据 COCO 标注文件, 将指定 (train 或 val) 数据集拆分为若干子集。
    返回拆分后的子集图像信息与对应的子集标注。
    
    :param coco_root: COCO 数据集根目录 (str)
    :param images_dir_name: 图像文件夹名 (train2017 或 val2017) (str)
    :param annotation_file_name: 标注文件名 (如 instances_train2017.json) (str)
    :param labels_dir_name: labels 文件夹下的子文件夹 (与 images_dir_name 对应), 若没有则设为 None (str)
    :param num_splits: 拆分份数 (int)
    :param shuffle: 是否在拆分前打乱图像列表 (bool)
    :return: (subsets, annotations_map)
        subsets: List[List[dict]], 每个子列表包含该子集的所有图像信息 (即 coco_annotations["images"] 中的子集)
        annotations_map: dict, 包含
          {
            "all_annotations": List[dict], # 原始所有 annotations
            "categories": List[dict],      # 原始 categories
            "info": dict,                  # 原始 info
            "licenses": List[dict]         # 原始 licenses
          }
    """
    # 1. 准备文件路径
    images_dir = os.path.join(coco_root, "images", images_dir_name)
    annotation_path = os.path.join(coco_root, "annotations", annotation_file_name)

    # 如果传入了 labels_dir_name, 则组装 labels 的完整路径
    labels_dir = None
    if labels_dir_name is not None:
        labels_dir = os.path.join(coco_root, "labels", labels_dir_name)
    
    # 2. 读取原始 COCO 标注
    with open(annotation_path, "r", encoding="utf-8") as f:
        coco_anno = json.load(f)

    all_images = coco_anno["images"]
    if shuffle:
        random.shuffle(all_images)

    # 每份大约多少张图像
    total_images = len(all_images)
    split_size = total_images // num_splits
    # split_size = 100

    # 3. 拆分图像信息
    subsets = []
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < num_splits - 1 else total_images
        # end_idx = (i + 1) * split_size
        subset_imgs = all_images[start_idx:end_idx]
        subsets.append(subset_imgs)

    # 4. 准备要返回的 annotations_map
    annotations_map = {
        "all_annotations": coco_anno["annotations"],
        "categories": coco_anno.get("categories", []),
        "info": coco_anno.get("info", {}),
        "licenses": coco_anno.get("licenses", [])
    }

    return subsets, annotations_map


def export_subset(
    subset_idx: int,
    subset_images: list,
    annotations_map: dict,
    images_dir_name: str,
    annotation_file_name: str,
    labels_dir_name: str,
    output_root: str,
    coco_root: str
):
    """
    将某个子集的图像 + 标注 + labels 文件导出到指定的子集文件夹中。
    
    :param subset_idx: 子集编号 (从 0 开始)
    :param subset_images: 当前子集的图像列表 (List[dict])
    :param annotations_map: 字典, 包含所有标注及配置信息
    :param images_dir_name: 原始图像文件夹 (train2017 / val2017)
    :param annotation_file_name: 对应的标注文件名
    :param labels_dir_name: labels 文件夹下对应的子文件夹 (同上)
    :param output_root: 最终输出的根目录 (coco_splits)
    :param coco_root: 原始 COCO 数据集根目录
    """
    # 1. 目录结构
    subset_dir = os.path.join(output_root, f"subset_{subset_idx + 1}")
    annotations_dir = os.path.join(subset_dir, "annotations")
    images_dir = os.path.join(subset_dir, "images", images_dir_name)
    labels_dir = os.path.join(subset_dir, "labels", labels_dir_name) if labels_dir_name else None

    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    if labels_dir:
        os.makedirs(labels_dir, exist_ok=True)

    # 2. 拷贝图像 & 拷贝 label（如果有）
    #   - 原始路径: coco_root/images/images_dir_name
    #   - 原始 labels 路径: coco_root/labels/labels_dir_name
    subset_image_ids = set()
    for img_info in subset_images:
        subset_image_ids.add(img_info["id"])
        img_file = img_info["file_name"]

        # 拷贝图像
        src_img_path = os.path.join(coco_root, "images", images_dir_name, img_file)
        dst_img_path = os.path.join(images_dir, img_file)
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            print(f"[警告] {src_img_path} 不存在, 跳过。")

        # 拷贝对应 label 文件 (假设仅后缀不同, 如 image.jpg -> image.txt)
        if labels_dir:
            base_name, _ = os.path.splitext(img_file)
            label_file = base_name + ".txt"
            src_label_path = os.path.join(coco_root, "labels", labels_dir_name, label_file)
            dst_label_path = os.path.join(labels_dir, label_file)
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dst_label_path)
            # 如果没有对应 label 文件, 可以打印警告或忽略
            # else:
            #     print(f"[提示] {src_label_path} 不存在, 对应 label 文件缺失。")

    # 3. 过滤标注（只保留该子集的图像）
    filtered_annos = [
        anno for anno in annotations_map["all_annotations"]
        if anno["image_id"] in subset_image_ids
    ]

    # 4. 生成新的标注文件
    subset_coco_anno = {
        "info": annotations_map["info"],
        "licenses": annotations_map["licenses"],
        "images": subset_images,
        "annotations": filtered_annos,
        "categories": annotations_map["categories"]
    }

    subset_annotation_file = os.path.join(annotations_dir, annotation_file_name)
    with open(subset_annotation_file, "w", encoding="utf-8") as f:
        json.dump(subset_coco_anno, f, ensure_ascii=False, indent=2)

    print(f"[子集 {subset_idx + 1}] {images_dir_name} -> 图像数: {len(subset_images)}, 标注数: {len(filtered_annos)}")


def split_coco_train_val_into_subsets(
    coco_root: str,
    output_root: str,
    num_splits: int = 10,
    shuffle: bool = True
):
    """
    将 train2017 & val2017 同时拆分为 num_splits 份, 输出到 coco_splits/subset_i/ 下, 
    并保持与原始 COCO 相同的层级结构 (annotations, images, labels), 
    其中 images 和 labels 分别包含 train2017 与 val2017 两级文件夹。
    
    假设原始目录结构：
    coco/
    ├── annotations/
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    ├── images/
    │   ├── train2017/
    │   └── val2017/
    └── labels/
        ├── train2017/
        └── val2017/
    
    拆分后结构 (举例 2 份)：
    coco_splits/
    ├── subset_1/
    │   ├── annotations/
    │   │   ├── instances_train2017.json
    │   │   └── instances_val2017.json
    │   ├── images/
    │   │   ├── train2017/
    │   │   └── val2017/
    │   └── labels/
    │       ├── train2017/
    │       └── val2017/
    └── subset_2/
        ├── annotations/
        ├── images/
        └── labels/
    
    :param coco_root: 原始 COCO 根目录
    :param output_root: 输出目录 (如 "coco_splits")
    :param num_splits: 拆分份数 (默认 10)
    :param shuffle: 是否对 train / val 的图像列表分别打乱
    """
    # 1) 拆分 train2017
    train_subsets, train_annos_map = split_dataset_by_coco_annotations(
        coco_root=coco_root,
        images_dir_name='train2017',
        annotation_file_name='instances_train2017.json',
        labels_dir_name='train2017',  # 假设 labels/train2017 与 images/train2017 对应
        num_splits=num_splits,
        shuffle=shuffle
    )

    # 2) 拆分 val2017
    val_subsets, val_annos_map = split_dataset_by_coco_annotations(
        coco_root=coco_root,
        images_dir_name='val2017',
        annotation_file_name='instances_val2017.json',
        labels_dir_name='val2017',  # 假设 labels/val2017 与 images/val2017 对应
        num_splits=num_splits,
        shuffle=shuffle
    )

    # 确保输出根目录存在
    os.makedirs(output_root, exist_ok=True)

    # 3) 依次导出各个子集
    for i in range(num_splits):
        # 导出 train2017
        export_subset(
            subset_idx=i,
            subset_images=train_subsets[i],
            annotations_map=train_annos_map,
            images_dir_name='train2017',
            annotation_file_name='instances_train2017.json',
            labels_dir_name='train2017',
            output_root=output_root,
            coco_root=coco_root
        )
        # 导出 val2017
        export_subset(
            subset_idx=i,
            subset_images=val_subsets[i],
            annotations_map=val_annos_map,
            images_dir_name='val2017',
            annotation_file_name='instances_val2017.json',
            labels_dir_name='val2017',
            output_root=output_root,
            coco_root=coco_root
        )
    print(f"已完成 train2017 & val2017 的 {num_splits} 份拆分, 输出至: {output_root}")
    
def generate_txt_for_subset(subset_dir, split='train2017'):
    """
    遍历 subset_dir/images/<split> 目录下的图像文件, 并将相对路径写入 txt 文件
    例如:
      images/train2017/000000273650.jpg
      images/train2017/000000000139.jpg
    :param subset_dir: 例如 'coco_splits/subset_1'
    :param split: 'train2017' 或 'val2017'
    :return: 生成的 txt 文件完整路径
    """
    images_split_dir = os.path.join(subset_dir, 'images', split)
    txt_filename = f"{split}.txt"  # train2017.txt 或 val2017.txt
    txt_path = os.path.join(subset_dir, txt_filename)

    # 收集该目录下所有图像文件
    lines = []
    if os.path.exists(images_split_dir):
        for fname in sorted(os.listdir(images_split_dir)):
            # 过滤出常见图像后缀
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # 在 YOLOv5 中, 如果 data.yaml 里配置了 `path: some_dir`
                # 则 train/val txt 里面的路径通常是相对于这个 `path`
                # 常见做法是写成 "images/train2017/xxx.jpg"
                rel_path = os.path.join('./images', split, fname)
                lines.append(rel_path)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    
    print(f"[生成] {txt_path}, 图像数 = {len(lines)}")
    return txt_path



if __name__ == "__main__":
    # 示例: 将 /path/to/coco 下的 train2017 和 val2017 各自拆分为 10 份, 
    # 并按照指定结构输出到 coco_splits 目录
    coco_root_path = 'D:/git/datasets/coco/'
    output_root_path = 'D:/git/dfodl/datasets/coco_split_test/'

    split_coco_train_val_into_subsets(
        coco_root=coco_root_path,
        output_root=output_root_path,
        num_splits=10,
        shuffle=True
    )