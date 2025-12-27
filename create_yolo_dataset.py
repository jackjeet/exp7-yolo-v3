import os
from PIL import Image
import numpy as np

# -------------------------- 1. 创建数据集文件夹结构 --------------------------
dataset_root = "datasets"
# 图片文件夹（train/val）
train_img_dir = os.path.join(dataset_root, "images", "train")
val_img_dir = os.path.join(dataset_root, "images", "val")
# 标签文件夹（train/val）
train_label_dir = os.path.join(dataset_root, "labels", "train")
val_label_dir = os.path.join(dataset_root, "labels", "val")

# 批量创建文件夹
for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
    os.makedirs(dir_path, exist_ok=True)
print("文件夹结构创建完成！")

# -------------------------- 2. 生成红外图片（模拟红外小目标） --------------------------
def create_infrared_image(save_path, target_positions):
    """创建模拟红外图片：黑色背景+白色小目标（符合红外图像特征）"""
    # 图片尺寸：480x320（常见红外图尺寸）
    img = np.zeros((320, 480), dtype=np.uint8)  # 黑色背景
    # 绘制白色小目标（圆形/矩形，模拟红外亮点）
    for (x, y, w, h) in target_positions:
        # 绘制矩形目标（小尺寸，符合小目标定义）
        img[y:y+h, x:x+w] = 255  # 白色目标
    # 保存为PNG图片
    Image.fromarray(img).save(save_path)

# 训练集图片1：1个小目标
train1_img_path = os.path.join(train_img_dir, "infrared_train1.png")
create_infrared_image(train1_img_path, [(200, 150, 8, 8)])  # (x,y,w,h)：目标位置和大小

# 训练集图片2：2个小目标
train2_img_path = os.path.join(train_img_dir, "infrared_train2.png")
create_infrared_image(train2_img_path, [(120, 100, 6, 6), (350, 220, 7, 7)])

# 验证集图片1：1个小目标
val1_img_path = os.path.join(val_img_dir, "infrared_val1.png")
create_infrared_image(val1_img_path, [(280, 180, 9, 9)])
print("红外图片生成完成！")

# -------------------------- 3. 生成YOLO格式标签（txt文件） --------------------------
def create_yolo_label(save_path, image_w, image_h, targets):
    """
    生成YOLO标签：格式为「类别ID 归一化x_center 归一化y_center 归一化w 归一化h」
    这里类别ID为0（只有1类：small_target）
    """
    label_lines = []
    for (x, y, w, h) in targets:
        # 计算归一化参数（YOLO要求坐标归一化到0-1）
        x_center = (x + w/2) / image_w
        y_center = (y + h/2) / image_h
        norm_w = w / image_w
        norm_h = h / image_h
        # 写入标签（类别0 + 四个归一化参数）
        label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    # 保存标签文件
    with open(save_path, "w", encoding="utf-8") as f:
        f.writelines(label_lines)

# 图片尺寸（和上面一致：480宽，320高）
img_w, img_h = 480, 320

# 训练集标签1（对应train1.png）
train1_label_path = os.path.join(train_label_dir, "infrared_train1.txt")
create_yolo_label(train1_label_path, img_w, img_h, [(200, 150, 8, 8)])

# 训练集标签2（对应train2.png）
train2_label_path = os.path.join(train_label_dir, "infrared_train2.txt")
create_yolo_label(train2_label_path, img_w, img_h, [(120, 100, 6, 6), (350, 220, 7, 7)])

# 验证集标签1（对应val1.png）
val1_label_path = os.path.join(val_label_dir, "infrared_val1.txt")
create_yolo_label(val1_label_path, img_w, img_h, [(280, 180, 9, 9)])
print("YOLO格式标签生成完成！")

# -------------------------- 4. 生成YOLO训练配置文件（dataset.yaml） --------------------------
yaml_content = f"""# YOLOv3 红外小目标数据集配置文件
# 数据集路径（相对路径，根据你的项目目录自动适配）
path: {dataset_root}  # 数据集根目录
train: images/train    # 训练集图片路径
val: images/val        # 验证集图片路径
test:                  # 测试集路径（可选，这里不用）

# 类别
names:
  0: small_target  # 只有1类：红外小目标
"""
# 保存配置文件到项目根目录
yaml_path = "infrared_dataset.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)
print(f"数据集配置文件生成完成：{yaml_path}")

print("\n✅ 完整YOLOv3训练数据集创建成功！")
print(f"数据集位置：{os.path.abspath(dataset_root)}")
print("包含：2张训练图+2个标签，1张验证图+1个标签，1个训练配置文件")