import cv2 as cv
import numpy as np
import os
from extractEndpoints import *
import math
from matplotlib import pyplot as plt



def getFeaturePoints(mask_path, origin):

    im = cv.imread(mask_path)
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 175, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    height, width = thresh.shape
    # print('shape: ', height, width)

    #创建空白画板
    sector = np.zeros((height, width), dtype=np.uint8) # 射线
    boundary = np.zeros((height,width), dtype=np.uint8) # 轮廓

    cv2.drawContours(boundary, contours, -1, 255, 1)

    start_point = origin
    # print('origin: ', start_point)
    endpoints = getEndpoints(thresh)
    # print('endpoints', endpoints)
    endpoint1 = endpoints[0]
    endpoint2 = endpoints[1]
    # print('endpoint1: ', endpoint1)
    # print('endpoint2: ', endpoint2)
    #通过计算两条直线斜率估算夹角
    # slope1 = (endpoint1[1]-start_point[1])/(start_point[0]-endpoint1[0])
    # slope2 = (endpoint2[1]-start_point[1])/(start_point[0]-endpoint2[0])
    # print(slope1, slope2)
    angle_rad1 = np.arctan2(endpoint1[1]-start_point[1], start_point[0]-endpoint1[0])


    angle_rad2 = np.arctan2(endpoint2[1]-start_point[1], start_point[0]-endpoint2[0])


    angle = [angle_rad1, angle_rad2]
    for i in range(len(angle)):  # 对arctan2函数的输出结果进行处理，将输出范围在[0, pi]
        if angle[i] < 0:
            angle[i] += np.pi
    angle.sort()
    # print(angle)
    angle_rad = angle[1]-angle[0]
    # print('夹角弧长: ', angle_rad)
    angle_step = angle_rad/20
    # print('步长: ', angle_step)
    angles = np.arange(angle[0], angle[1]+angle_step, angle_step)
    reversed(angles)
    # print(angles)
    length = 800
    feature_points = []

    canvas = np.copy(boundary)
    for i in reversed(range(len(angles))):
        if i % 2 == 1:
            angle = angles[i]
            # print(length*math.cos(angle))
            endpoint_x = start_point[0]+length*math.cos(angle)
            endpoint_y = start_point[1]-length*math.sin(angle)
            endpoint = (int(endpoint_x), int(endpoint_y))
            cv.line(sector, start_point, endpoint, 255, 1)
            cv.line(canvas, start_point, endpoint, 255, 1)
            # plt.imshow(canvas, 'gray')
            # plt.show()
            result = sector & boundary
            fp = np.argwhere(result == 255)
            mean = np.mean(fp, axis=0)
            mean = np.round(mean).astype(int)
            # print(fp)
            # print(mean)
            feature_points.append(mean)
            sector = np.zeros(sector.shape, dtype=np.uint8)
    return feature_points

if __name__ == '__main__':
    path = './annotated/myWithAxes/mask'
    im_path = os.path.join(path, 'IMG-0002-00002.jpg')
    featurePoints = getFeaturePoints(im_path, (275, 634))
    print(featurePoints)

    # bgr_boundary = cv.cvtColor(boundary, cv.COLOR_GRAY2BGR)
    # for point in feature_points:
    #     x, y = point
    #     cv.circle(bgr_boundary, (y, x), 3, (0, 255, 0), -1)
    # plt.subplot(121), plt.imshow(canvas, 'gray')
    # plt.subplot(122), plt.imshow(bgr_boundary)
    # plt.show()



# # 绘制线段
# thickness = 2
# cv.line(thresh, start_point, endpoint1, 255, thickness)
# cv.line(thresh, start_point, endpoint2, 255, thickness)
#

# plt.imshow(canvas), plt.title('sector')
# plt.show()

# cv.namedWindow('Image Window', cv.WINDOW_NORMAL)
# cv.resizeWindow('Image Window', 440, 540)
# cv.imshow('Image Window', thresh)
# cv.waitKey(0)
# cv.destoryAllWindows()

# concate = np.hstack((canvas, thresh))
# cv.imshow('concate', concate)
# cv.waitKey(0)
# cv.destoryAllWindows()
