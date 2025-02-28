import os

COCO_NAMES_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

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


def generate_coco_subset_yaml(
    subset_dir,
    output_yaml_path,
    nc=80,
    names=None
):
    """
    生成 COCO 子集的 data.yaml
    包含:
      path: <subset_dir>
      train: train2017.txt
      val: val2017.txt
      nc: 80
      names: [ ... ]
    """
    if names is None:
        names = COCO_NAMES_80  # 默认80类COCO
    
    yaml_lines = [
        f"path: {subset_dir}",      # 数据集根目录
        f"train: train2017.txt",   # 相对于 path 的 txt
        f"val: val2017.txt",
        "",
        f"nc: {nc}",
        "names:",
    ]
    for i, nm in enumerate(names):
        yaml_lines.append(f"  {i}: {nm}")
    
    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(yaml_lines))
    print(f"[生成] {output_yaml_path}")


def generate_yolo_data_for_coco_splits(
    base_splits_dir="coco_splits",
    num_subsets=10,
    prefix="data_node",
    nc=80,
    names=None
):
    """
    1) 在 coco_splits/subset_i 下生成 train2017.txt 和 val2017.txt
    2) 生成对该子集的 data_node{i}.yaml 文件 (同目录或另行指定输出目录皆可)
    
    默认:
      subset_1/ -> data_node0.yaml
      subset_2/ -> data_node1.yaml
      ...
      subset_{num_subsets}/ -> data_node{num_subsets-1}.yaml
    """
    if names is None:
        names = COCO_NAMES_80
    
    # 确保 base_splits_dir 存在
    if not os.path.isdir(base_splits_dir):
        raise FileNotFoundError(f"未找到目录 {base_splits_dir}")
    
    for i in range(num_subsets):
        node_id = i
        subset_id = i + 1
        subset_folder = os.path.join(base_splits_dir, f"subset_{subset_id}")
        
        if not os.path.isdir(subset_folder):
            print(f"警告: 子集目录 {subset_folder} 不存在, 跳过。")
            continue
        
        print(f"\n=== 处理 {subset_folder} ===")
        
        # 生成 train2017.txt, val2017.txt
        train_txt_path = generate_txt_for_subset(subset_folder, 'train2017')
        val_txt_path = generate_txt_for_subset(subset_folder, 'val2017')
        
        # 生成 data.yaml
        # 如果你想让 data.yaml 也放在 subset_i 下, 可以这样:
        #   yaml_path = os.path.join(subset_folder, f"{prefix}{node_id}.yaml")
        # 也可以存放到另一个目录, 这里演示直接放到 subset 里
        yaml_path = os.path.join(subset_folder, f"{prefix}{node_id}.yaml")
        
        generate_coco_subset_yaml(
            subset_dir=subset_folder,
            output_yaml_path=yaml_path,
            nc=nc,
            names=names
        )


if __name__ == "__main__":
    generate_yolo_data_for_coco_splits(
        base_splits_dir='D:/git/dfodl/datasets/coco_split_test/',  # 你的 coco_splits 根目录
        num_subsets=10,                 # 你拆分了 10 份
        prefix="data_node",             # 生成的 data 文件前缀
        nc=80,                          # COCO 类别数
        names=COCO_NAMES_80             # COCO 类别名称
    )
