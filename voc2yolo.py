import os
import random
import xml.etree.ElementTree as ET

# -------------------------- 配置参数 --------------------------
# 数据集根目录（根据你的实际路径修改）
dataset_root ="D:/sirst_dataset"
# 类别列表（SIRST只有1类：small_target）
classes = ["small_target"]
# 训练/验证/测试集划分比例
train_ratio = 0.7  # 70%训练
val_ratio = 0.2  # 20%验证
test_ratio = 0.1  # 10%测试

# -------------------------- 1. 创建必要文件夹 --------------------------
# 标签输出目录
labels_dir = os.path.join(dataset_root, "labels")
os.makedirs(labels_dir, exist_ok=True)
# ImageSets/Main目录（存放划分后的列表文件）
imagesets_main_dir = os.path.join(dataset_root, "ImageSets", "Main")
os.makedirs(imagesets_main_dir, exist_ok=True)


# -------------------------- 2. XML转YOLO TXT --------------------------
def convert_xml_to_yolo(xml_path, txt_path):
    """将XML标注转换为YOLO TXT格式"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图片尺寸
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    # 遍历所有目标
    yolo_lines = []
    for obj in root.findall("object"):
        # 获取类别名称
        class_name = obj.find("name").text
        if class_name not in classes:
            continue
        class_id = classes.index(class_name)

        # 获取边界框坐标（xmin, ymin, xmax, ymax）
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # 转换为YOLO格式：归一化中心坐标+归一化宽高
        x_center = (xmin + xmax) / 2 / img_w
        y_center = (ymin + ymax) / 2 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        # 写入TXT文件（格式：class_id x_center y_center width height）
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # 保存TXT标签
    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(yolo_lines)


# -------------------------- 3. 处理所有XML文件 --------------------------
# 获取所有XML文件
annotations_dir = os.path.join(dataset_root, "Annotations")
xml_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]
print(f"找到 {len(xml_files)} 个XML标注文件")

# 转换所有XML为YOLO TXT
for xml_file in xml_files:
    xml_path = os.path.join(annotations_dir, xml_file)
    txt_file = xml_file.replace(".xml", ".txt")
    txt_path = os.path.join(labels_dir, txt_file)
    convert_xml_to_yolo(xml_path, txt_path)
print("XML转YOLO TXT完成！")

# -------------------------- 4. 划分训练/验证/测试集 --------------------------
# 获取所有图片文件名（不含后缀）
image_ids = [os.path.splitext(f)[0] for f in xml_files]
random.shuffle(image_ids)  # 随机打乱

# 计算划分数量
total = len(image_ids)
train_num = int(total * train_ratio)
val_num = int(total * val_ratio)
test_num = total - train_num - val_num

# 划分数据集
train_ids = image_ids[:train_num]
val_ids = image_ids[train_num:train_num + val_num]
test_ids = image_ids[train_num + val_num:]


# 保存划分后的列表文件
def save_ids(ids, filename):
    with open(os.path.join(imagesets_main_dir, filename), "w", encoding="utf-8") as f:
        for id in ids:
            f.write(f"{id}\n")


save_ids(train_ids, "train.txt")
save_ids(val_ids, "val.txt")
save_ids(test_ids, "test.txt")
save_ids(train_ids + val_ids, "trainval.txt")  # 训练+验证集

print(f"数据集划分完成：")
print(f"  训练集：{len(train_ids)} 张")
print(f"  验证集：{len(val_ids)} 张")
print(f"  测试集：{len(test_ids)} 张")

# -------------------------- 5. 生成YOLOv5配置文件 --------------------------
yolo5_config = f"""# YOLOv5 红外小目标数据集配置
train: {os.path.join(dataset_root, "ImageSets", "Main", "train.txt")}
val: {os.path.join(dataset_root, "ImageSets", "Main", "val.txt")}
test: {os.path.join(dataset_root, "ImageSets", "Main", "test.txt")}

nc: 1  # 类别数量
names: ['small_target']  # 类别名称
"""

# 保存配置文件到yolov5-master/data目录（如果存在）
yolov5_data_dir = "D:/yolov5-master/data"
if os.path.exists(yolov5_data_dir):
    config_path = os.path.join(yolov5_data_dir, "sirst.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(yolo5_config)
    print(f"YOLOv5配置文件已保存到：{config_path}")
else:
    print("未找到yolov5-master/data目录，请手动创建配置文件")

print("\n✅ 所有步骤完成！")