import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

dir = 'annotated/myWithAxes/withAxes'
path = os.path.join(dir, 'IMG-0001-00003.jpg')

img = cv.imread(path)
print(img.shape)
b, g, r = cv.split(img) # g channel for extracting axes and r channel for mask

plt.subplot(131), plt.imshow(b, cmap='gray'), plt.title('blue channel')
plt.subplot(132), plt.imshow(g, cmap='gray'), plt.title('green channel')
plt.subplot(133), plt.imshow(r, cmap='gray'), plt.title('red channel')
plt.show()

## find the first non-zero pixel value
# row_index = 249
# image_matrix = np.array(r)
# for col_index, pixel_value in enumerate(image_matrix[row_index]):
#     if np.any(pixel_value != 0):  # check if the pixel value is non-zero
#         x_start = col_index
#         break

x_start = 520
x_end = 1920 - x_start

cropped_r = r[0:, x_start:x_end]
cropped_g = g[0:, x_start:x_end]
# print(cropped_image.shape)
# plt.subplot(121), plt.imshow(cropped_r), plt.title('red channel cropped')
# plt.subplot(122), plt.imshow(cropped_g), plt.title('green channel cropped')
# plt.show()
# Median filter to smooth image,
median_kernel_size = 3
median_blurred_r = cv.medianBlur(cropped_r, median_kernel_size)
# cv_show('median filter', median_blurred)

#阈值法获得二值化图像
threshold_value = 230
_, thresholded_image = cv.threshold(median_blurred_r, threshold_value, 255, cv.THRESH_BINARY)
# cv_show('threshold filter', thresholded_image)
binary_thresh = 5
_, binary_image = cv.threshold(thresholded_image, binary_thresh, 255, cv.THRESH_BINARY)
plt.subplot(221), plt.imshow(cropped_r), plt.title('cropped_r')
plt.subplot(222), plt.imshow(binary_image), plt.title('binary_image')
# plt.show()


# 连通区域分析
num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_image, connectivity=8)
print('num_labels = ', num_labels)
print('stats = ', stats)  # 5 cols: x, y, width, height, area

# 找到满足条件的行索引 i
condition = (stats[:, 4] >= 2700) & (stats[:, 4] <= 4400)
row_index = np.where(condition)[0][0]

# row_index = np.argmax(stats[1:, 4]) + 1
target_label = row_index
mask = np.zeros_like(binary_image, dtype=np.uint8)
mask[labels == target_label] = 255
plt.subplot(223), plt.imshow(mask), plt.title('mask')


# 边缘检测
edges = cv.Canny(cropped_g, 50, 150)

# 进行霍夫直线检测
lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=130, minLineLength=130, maxLineGap=40)

# canvas = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
# 在图像上绘制直线
axes = np.zeros(cropped_g.shape, dtype=np.uint8)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(axes, (x1, y1), (x2, y2), 255, 2)

# 显示结果图像
plt.subplot(224), plt.imshow(axes), plt.title('axes')
plt.show()

# cv.imshow('mask', mask)
# cv.waitKey(0)
# cv.destoryAllWindows()



