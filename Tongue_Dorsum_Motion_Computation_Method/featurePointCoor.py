import cv2
import numpy as np
import os
import math
from matplotlib import pyplot as plt
import pandas as pd
import ast
import sys
sys.path.append('./utils/')
from extractEndpoints import *
height = 1080
width = 880

# canvas = np.zeros((height, width), dtype=np.uint8)

def getContours(mask_img):
    h, w, c = mask_img.shape
    boundary = np.zeros((h, w), dtype=np.uint8)
    mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(mask_gray, 175, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(boundary, contours, -1, 255, 2)
    return boundary


def getFeaturePoints(mask_path, origin):
    boundary = np.zeros((height, width), dtype=np.uint8)  # 轮廓
    # print(mask_path)

    im = cv2.imread(mask_path)
    boundary = getContours(im)
    # plt.imshow(im)
    # plt.show()

    # imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 175, 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # height, width = thresh.shape
    # print('shape: ', height, width)

    #创建空白画板
    sector = np.zeros((height, width), dtype=np.uint8) # 射线


    # cv2.drawContours(boundary, contours, -1, 255, 2)

    start_point = origin
    # print('origin: ', start_point)
    endpoints = getEndpoints(boundary)
    print(endpoints)

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
    for n in range(len(angle)):  # 对arctan2函数的输出结果进行处理，将输出范围在[0, pi]
        if angle[n] < 0:
            angle[n] += np.pi
        if angle[n] > np.pi*3/4:
            angle[n] -= np.pi
    angle.sort()
    # print(angle)
    angle_rad = angle[1]-angle[0]
    # print('夹角弧长: ', angle_rad)
    angle_step = angle_rad/20
    # print('步长: ', angle_step)
    angles = np.arange(angle[0], angle[1], angle_step)
    reversed(angles)
    # print(len(angles))
    length = 800
    feature_points = []

    can = np.copy(boundary)
    # plt.imshow(can, 'gray')
    # plt.show()
    for n in reversed(range(len(angles))):  # 取奇数射线
        if n % 2 == 1:
            print(n)
            angle = angles[n]
            # print(length*math.cos(angle))
            endpoint_x = start_point[0]+length*math.cos(angle)
            endpoint_y = start_point[1]-length*math.sin(angle)
            endpoint = (int(endpoint_x), int(endpoint_y))
            print(f'start_point: {start_point}, end_point: {endpoint}')
            cv2.line(sector, start_point, endpoint, 255, 2)
            cv2.line(can, start_point, endpoint, 255, 2)
            # if n == 19:
            #     plt.imshow(can, 'gray')
            #     plt.show()
            result = sector & boundary
            # plt.imshow(result, 'gray')
            # plt.show()
            # apply connected components analysis to find the centroid of two connected components
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(result, connectivity=8)
            # get the coordinates of each connected component
            # print('stats: ', stats)
            # print(labels)
            # find the coordinates of labels where label value is 1 or 2
            p1_index = np.argwhere(labels == 1)[:, 1]
            p2_index = np.argwhere(labels == 2)[:, 1]
            if n > 10:
                p1_index = p1_index.argmin()  # leftmost point
                p2_index = p2_index.argmax()  # rightmost point
            else:
                p1_index = p1_index.argmax()  # rightmost point
                p2_index = p2_index.argmin()  # leftmost point

            # upper left point
            p1_coor = np.argwhere(labels == 1)[p1_index] # upper left point
            p2_coor = np.argwhere(labels == 2)[p2_index]  # lower right point
            p1_coor = (p1_coor[1], p1_coor[0])
            p2_coor = (p2_coor[1], p2_coor[0])
            # print(centroids)
            # result_BGR = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            # cv2.circle(result_BGR, p1_coor, 1, (0, 255, 0), -1)
            # cv2.circle(result_BGR, p2_coor, 1, (0, 255, 0), -1)
            # cv2.imshow('result_BGR', result_BGR)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # assert ret == 3
            mean_x = np.mean([p1_coor[0], p2_coor[0]])
            mean_y = np.mean([p1_coor[1], p2_coor[1]])
            mean = (round(mean_x), round(mean_y))
            # cv2.circle(result_BGR, mean, 3, (0, 255, 0), -1)
            # cv2.imshow('result_BGR', result_BGR)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # fp = np.argwhere(result == 255)
            # # print(type(fp), fp)
            # mean = np.mean(fp, axis=0)
            # mean = np.round(mean).astype(int)
            # mean = (mean[1], mean[0])
            # # print('fp: ', fp)
            # # print('mean: ', mean)


            feature_points.append(mean)
            sector = np.zeros(sector.shape, dtype=np.uint8)
    return feature_points

if __name__ == '__main__':
    subject_list = ['wtt', 'wtt2']
    for subject in subject_list:
        print('Processing:', subject)
        if '2' in subject:
            path = './Results/mask/'+subject[:-1]+'_Mask2'
            feature_path = './Results/visual/' + subject[:-1] + '_featurePoints2'
            df = pd.read_csv('./Results/data_csv/' + subject[:-1] + '_data2.csv')
        else:
            path = './Results/mask/'+subject+'_Mask'
            feature_path = './Results/visual/' + subject + '_featurePoints'
            df = pd.read_csv('./Results/data_csv/'+subject+'_data.csv')
        origin = df.loc[:, 'origin']
        # print(origin)
        i = 0
        for file in os.listdir(path):
            print(file)
            im_path = os.path.join(path, file)

            # print(os.path.exists(im_path))
            o_point = ast.literal_eval(origin[i])
            # print(o_point)
            featurePoints = getFeaturePoints(im_path, o_point)
            print(featurePoints)

            mask_img = cv2.imread(im_path)
            boundary = getContours(mask_img)

            bgr_boundary = cv2.cvtColor(boundary, cv2.COLOR_GRAY2BGR)
        #     for p_num, fp in enumerate(featurePoints):
        #         df.at[i, 'point'+str(p_num+1)] = str(fp)
        #         # plt.imshow(bgr_boundary)
        #         # plt.show()
        #     i += 1
        # df.to_csv('./Results/wtt_data3.csv', index=False)
            for point in featurePoints:
                x, y = point
                cv2.circle(bgr_boundary, (x, y), 3, (0, 255, 0), -1)
                # plt.imshow(bgr_boundary)
                # plt.show()

            if not os.path.exists(feature_path):
                os.mkdir(feature_path)
            cv2.imwrite(feature_path+'/'+file, bgr_boundary)
            i += 1
        # boundary = np.zeros((height, width), dtype=np.uint8)

        # plt.subplot(121), plt.imshow(canvas)
        # plt.subplot(122), plt.imshow(bgr_boundary)
        # plt.show()
        # for i in range(1, 11):
        #     col_name = 'point' + str(i)
        #     df.at[i, col_name] = str(featurePoints[i - 1])
        # df.to_csv('./data.csv', index=False)



# # 绘制线段
# thickness = 2
# cv2.line(thresh, start_point, endpoint1, 255, thickness)
# cv2.line(thresh, start_point, endpoint2, 255, thickness)
#

# plt.imshow(canvas), plt.title('sector')
# plt.show()

# cv2.namedWindow('Image Window', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Image Window', 440, 540)
# cv2.imshow('Image Window', thresh)
# cv2.waitKey(0)
# cv2.destoryAllWindows()

# concate = np.hstack((canvas, thresh))
# cv2.imshow('concate', concate)
# cv2.waitKey(0)
# cv2.destoryAllWindows()
