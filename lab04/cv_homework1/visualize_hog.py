from skimage.feature import hog
import matplotlib.pyplot as plt
import cv2
import time

if __name__ == '__main__':
    img = cv2.imread('jinx.jpg')
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 计算灰度直方图
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # 绘制灰度直方图
    plt.figure(figsize=(10, 5))
    plt.plot(hist, color='black')
    plt.xlim([0, 256])
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

    t0 = time.time()
    # TODO: 提取hog特征
    hog_features, hog_vis = hog(gray_image, 
                                orientations=9, 
                                pixels_per_cell=(4, 4), 
                                cells_per_block=(2, 2), 
                                visualize=True, 
                                block_norm='L2-Hys')
    t1 = time.time()
    print('Elapsed time is', t1 - t0, 'seconds')
    

    # 可视化特征
    fig, ax = plt.subplots(1, 2, subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title('input image')
    ax[1].imshow(hog_vis, cmap='bone')
    ax[1].set_title('visualization of HOG features')
    plt.show()

    # 输出 HOG 特征的维度
    print("HOG feature vector size:", hog_features.shape)
