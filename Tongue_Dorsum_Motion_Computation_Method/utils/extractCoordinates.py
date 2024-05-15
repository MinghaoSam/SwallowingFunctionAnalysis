import os
import cv2 as cv
import numpy as np

dir = 'annotated/myWithAxes/withAxes'
path = os.path.join(dir, 'IMG-0001-00003.jpg')

img = cv.imread(path)
print(img.shape)
b, g, r = cv.split(img)
## find the first non-zero pixel value
# row_index = 249
# image_matrix = np.array(r)
# for col_index, pixel_value in enumerate(image_matrix[row_index]):
#     if np.any(pixel_value != 0):  # check if the pixel value is non-zero
#         x_start = col_index
#         break
x_start = 520
x_end = 1920 - x_start

cropped_image = r[0:, x_start:x_end]
# 边缘检测
edges = cv.Canny(cropped_image, 120, 130)

# 进行霍夫直线检测
lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

canvas = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
# 在图像上绘制直线
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(canvas, (x1, y1), (x2, y2), 255, 1)

# 显示结果图像
cv.imshow('Lines', canvas)
cv.waitKey(0)
cv.destroyAllWindows()
