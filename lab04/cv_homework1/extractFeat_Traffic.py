import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import color

import joblib

# 创建目录函数
def mkdir_or_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 读取数据集目录
def load_images_from_folder(folder):
    images = []
    labels = []
    for class_folder in os.listdir(folder):
        class_path = os.path.join(folder, class_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(class_folder)  # 使用文件夹名作为标签
    return images, labels

# 提取 HOG 特征并保存
def extract_and_save_hog_features(images, labels, save_folder):
    orientations = 12
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)

    mkdir_or_exist(save_folder)  # 确保保存目录存在
    label_mapping = {label: idx for idx, label in enumerate(set(labels))}  # 创建标签映射
    for i, img in enumerate(images):
        gray_img = color.rgb2gray(img)  # 将图像转换为灰度图
        # 提取 HOG 特征
        hog_fd = hog(gray_img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                     cells_per_block=cells_per_block, block_norm='L1-sqrt', transform_sqrt=True)
        
        # 添加标签
        label = label_mapping[labels[i]]  # 获取整数标签
        feature_data = np.concatenate([hog_fd, [label]])  # 将特征和标签连接在一起
        
        # 保存为 .feat 文件
        feat_filename = f"image_{i+1}.feat"
        feat_path = os.path.join(save_folder, feat_filename)
        joblib.dump(feature_data, feat_path)
        print(f"Saved: {feat_path}")

# 主函数
def main():
    # 设置训练集和测试集的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dataset_path = os.path.join(script_dir, "Traffic_sign/train_dataset")
    test_dataset_path = os.path.join(script_dir, "Traffic_sign/test_dataset")
  
    # 设置保存路径
    train_save_folder = "./traffic_sign_features/train"
    test_save_folder = "./traffic_sign_features/test"
    
    # 加载训练集图像和标签
    print("Loading training dataset...")
    train_images, train_labels = load_images_from_folder(train_dataset_path)
    
    # 提取并保存训练集的 HOG 特征
    print("Extracting and saving HOG features from training dataset...")
    extract_and_save_hog_features(train_images, train_labels, train_save_folder)
    
    # 加载测试集图像和标签
    print("Loading test dataset...")
    test_images, test_labels = load_images_from_folder(test_dataset_path)
    
    # 提取并保存测试集的 HOG 特征
    print("Extracting and saving HOG features from test dataset...")
    extract_and_save_hog_features(test_images, test_labels, test_save_folder)

if __name__ == "__main__":
    main()
