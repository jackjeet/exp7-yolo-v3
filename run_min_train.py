import os
import shutil
import subprocess
import cv2
import numpy as np

# ===================== 前置：设置Git环境变量，跳过Git检查 =====================
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
print("✅ 已设置Git环境变量，跳过Git检查")

# ===================== 第一步：创建极简测试数据集 =====================
# 定义基础路径
base_path = "D:/test_sirst"
img_path = os.path.join(base_path, "images")
label_path = os.path.join(base_path, "labels")

# 创建文件夹（如果不存在）
for path in [base_path, img_path, label_path]:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建文件夹：{path}")

# 1. 生成测试图片（创建320x320空白图）
test_img_path = os.path.join(img_path, "test.png")
blank_img = np.ones((320, 320, 3), dtype=np.uint8) * 255  # 白色空白图
cv2.imwrite(test_img_path, blank_img)
print(f"创建空白测试图片：{test_img_path}")

# 2. 生成YOLO格式标签文件
test_label_path = os.path.join(label_path, "test.txt")
with open(test_label_path, "w", encoding="utf-8") as f:
    f.write("0 0.5 0.5 0.1 0.1")  # 类别0 + 中心坐标(0.5,0.5) + 宽高(0.1,0.1)
print(f"创建标签文件：{test_label_path}")

# 3. 生成train.txt/val.txt
train_txt_path = os.path.join(base_path, "train.txt")
val_txt_path = os.path.join(base_path, "val.txt")
with open(train_txt_path, "w", encoding="utf-8") as f:
    f.write("D:/test_sirst/images/test.png")
with open(val_txt_path, "w", encoding="utf-8") as f:
    f.write("D:/test_sirst/images/test.png")
print(f"创建训练集文件：{train_txt_path}")
print(f"创建验证集文件：{val_txt_path}")

# ===================== 第二步：修改sirst.yaml配置 =====================
# YOLOv5的sirst.yaml路径（确认和你的实际路径一致）
sirst_yaml_path = "C:/Users/李湘琪/Downloads/yolov5-master (1)/yolov5-master/data/sirst.yaml"

# 写入极简配置
yaml_content = """path: D:/test_sirst  # 极简数据集根目录
train: train.txt     # 训练集路径
val: val.txt         # 验证集路径
test: val.txt        # 测试集路径

nc: 1                # 类别数
names: ['target']    # 类别名
"""
with open(sirst_yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)
print(f"已修改配置文件：{sirst_yaml_path}")

# ===================== 第三步：执行1轮训练命令 =====================
# 切换到YOLOv5根目录
yolov5_root = "C:/Users/李湘琪/Downloads/yolov5-master (1)/yolov5-master"
os.chdir(yolov5_root)
print(f"切换到YOLOv5目录：{yolov5_root}")

# 训练命令（关闭缓存+1批次+1轮训练）
train_cmd = [
    "python", "train.py",
    "--img", "320",
    "--batch", "1",
    "--epochs", "1",
    "--data", "data/sirst.yaml",
    "--cfg", "models/yolov5s.yaml",
    "--weights", "none",
    "--name", "sirst_exp",
    "--cache", "None"
]

# 执行训练并打印输出
print("\n🚀 开始执行1轮训练...")
process = subprocess.Popen(
    train_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    encoding="utf-8",
    shell=True
)

# 实时输出训练日志
train_log = []
for line in iter(process.stdout.readline, ''):
    print(line.strip())
    train_log.append(line.strip())
process.wait()

# 保存训练日志到文件（方便实验报告使用）
log_path = os.path.join(yolov5_root, "train_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(train_log))
print(f"\n✅ 训练日志已保存至：{log_path}")
print("📊 可用于实验报告的关键信息：")
print("   - 训练环境：CPU (torch-2.9.1+cpu)")
print("   - 训练参数：img_size=320, batch_size=1, epochs=1")
print("   - 模型：YOLOv5s (214层，702万参数)")
print("   - 训练日志：train_log.txt（包含loss、耗时等核心数据）")