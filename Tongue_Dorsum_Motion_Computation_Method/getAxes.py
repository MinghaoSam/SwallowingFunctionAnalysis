import os
import re

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

height = 1080 # 图片高度，用于坐标转换

def angle_filter(lines, angle_threshold):
    '''
    :param lines: 直线两点坐标 [[[x1, y1, x2, y2]] ...]
    :param angle_threshold: 角度阈值，两直线夹角小于角度阈值时只保留一条直线
    :return: 过滤后的lines列表
    '''
    # print(lines.shape)
    idx_to_del = []
    for idx, val in enumerate(lines):
        x_1, y_1, x_2, y_2 = val[0]
        if x_1 < 10 and x_2 < 10 or x_1 > 870 and x_2 > 870:  # 去除左右两侧的直线
            idx_to_del.append(idx)
    lines = np.delete(lines, idx_to_del, axis=0)
    # print(lines.shape)

    sorted_lines = sorted(lines, key=lambda x: x[0][0], reverse=True)

    # print('sorted_lines: ', sorted_lines)

    filtered_lines = []
    for i in range(len(sorted_lines)):
        angle_i = math.atan2(sorted_lines[i][0][3] - sorted_lines[i][0][1], sorted_lines[i][0][2] - sorted_lines[i][0][0]) * 180 / math.pi
        # print(angle_i)
        is_redundant = False
        # if angle_i < -85 and angle_i > -95:
        #     is_redundant = True
        for j in range(i+1, len(sorted_lines)):
            angle_j = math.atan2(sorted_lines[j][0][3] - sorted_lines[j][0][1], sorted_lines[j][0][2] - sorted_lines[j][0][0]) * 180 / math.pi
            angle_diff = abs(abs(angle_i) - abs(angle_j))
            if angle_diff < angle_threshold:
                is_redundant = True
                # print(sorted_lines[i], 'is redundant')
                break
        if not is_redundant:
            filtered_lines.append(sorted_lines[i])
    return filtered_lines

def get_line_params(line):
    '''
    :param line: [[x1, y1, x2, y2]]
    :return: slope斜率, intercept截距
    '''
    x1, y1, x2, y2 = line[0]
    if x2 == x1:
        # slope = inf
        slope = 'inf'
        intercept = x1
        return slope, intercept
    else:
        slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    # print('slope: ', slope)
    # print('intercept: ', intercept)
    return slope, intercept



def getIntersectCoor(lines):
    # print(lines)
    line1 = lines[0]
    line2 = lines[1]

    slope1, intercept1 = get_line_params(line1)
    slope2, intercept2 = get_line_params(line2)
    if slope1 == 'inf':
        x_intersect = intercept1
        y_intersect = slope2 * x_intersect + intercept2
        intersect = (round(x_intersect), round(y_intersect))
        return intersect
    if slope2 == 'inf':
        x_intersect = intercept2
        y_intersect = slope1 * x_intersect + intercept1
        intersect = (round(x_intersect), round(y_intersect))
        return intersect
    x_intersect = (intercept2 - intercept1) / (slope1 - slope2)
    y_intersect = slope1 * x_intersect + intercept1
    intersect = (round(x_intersect), round(y_intersect))
    return intersect

def get_theta(line):
   '''
   输入line，输出line与图像底边的夹角
   :param line: [x1, y1, x2, y2]
   :return: angle_rad 夹角的弧度
   '''
   x1, y1, x2, y2 = line[0]
   delta_x = x2 - x1
   delta_y = y1 - y2 # 注意原点位置
   angle_rad = np.arctan2(delta_y, delta_x)  # arctan2的输出范围是[-pi, pi]
   return angle_rad

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img


def getOriginAndTheta(f_path):
    img = cv_imread(f_path)
    x_start = 520
    width = img.shape[1]
    # print(width)
    x_end = width - x_start

    cropped_img = img[10:-10, x_start:x_end]
    B, G, R = cv2.split(cropped_img)
    # channel substraction
    sub_res = cv2.subtract(G, R)
    kernel = np.ones((3, 3), np.uint8)
    dilate_1 = cv2.dilate(sub_res, kernel, iterations=1)
    median_blurred = cv2.medianBlur(dilate_1, 3)
    binary_img = cv2.inRange(median_blurred, 100, 255)
    # plt.imshow(binary_img, cmap='gray'), plt.title('binary_img')
    # plt.show()
    # dilate the image
    # kernel_5 = np.ones((5, 5), np.uint8)
    # dilate_img = cv2.dilate(binary_img, kernel_5, iterations=1)
    # if '01' in f_path:
    #     print(f_path)
    #     plt.imshow(binary_img, cmap='gray')
    #     plt.show()


    # gray_img = cv2.cvtColor(cv_imread(path), cv2.COLOR_BGR2GRAY) # read source jpg as gray image with shape (1080, 1920)
    # # b, g, r = cv2.split(img)
    #
    # cropped_img = gray_img[10:-10, x_start:x_end] # crop the image to shape (1060, 880)
    # # print(cropped_img.shape)
    # # plt.imshow(cropped_img, cmap='gray'), plt.title('cropped')
    # # plt.show()
    # # canvas = np.zeros(cropped_img.shape, dtype=np.uint8)
    # edges = cv2.Canny(cropped_img, 50, 230)
    # # plt.imshow(edges, cmap='gray'), plt.title('Canny to detect edges')
    # # plt.show()
    lines = cv2.HoughLinesP(binary_img, 1, np.pi/180, 100, minLineLength=30, maxLineGap=20)
    print('Before line filter: ', lines)
    lines = angle_filter(lines, 10)
    # print('After line filter: ', lines)
    for line in lines:
        line[0][1] = line[0][1] + 10
        line[0][3] = line[0][3] + 10
    origin = getIntersectCoor(lines)
    theta1 = get_theta(lines[0])
    theta2 = get_theta(lines[1])
    if -(math.pi / 4) < theta1 < math.pi / 4: # 确定X轴与图片底部夹角，用于配准
        theta_x = theta1
    else:
        theta_x = theta2
    theta_x_deg = np.rad2deg(theta_x)
    # print('原点Origin: ', origin)
    # print('X轴与图片底部夹角: ', theta_x, theta_x_deg)

    # draw origin on the image
    cv2.circle(cropped_img, (origin[0], origin[1]-10), 2, (0, 0, 255), -1)

    # save_path = f_path[:17]+'visualAxes'
    # print(save_path)
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    #     print(f'new folder {save_path} created!')
    # print(save_path+'/'+f_path[-18:])
    # # cv2.imwrite(save_path+'/'+f_path[-18:], cropped_img)  # chinese characters in path, can't be saved
    # cv2.imencode('.jpg', cropped_img)[1].tofile(save_path+'/'+f_path[-18:])


    # if '0033' in path:
    #     top_bottom = np.zeros([10, width - 520 * 2])
    #     axes = np.vstack((top_bottom, cropped_img))
    #     axes = np.vstack((axes, top_bottom))
    #     for line in lines:
    #         print('Look here: ', line)
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(axes, (x1, y1), (x2, y2), 255, 1, lineType=cv2.LINE_AA)
    #     # top_bottom = np.zeros([10, width-520*2])
    #     # print(canvas.shape)
    #     # print(top_bottom.shape)
    #     # axes = np.vstack((top_bottom, cropped_img))
    #     # axes = np.vstack((axes, top_bottom))
    #     # print(axes.shape)
    #     plt.imshow(axes, cmap='gray'), plt.title('axes')
    #     plt.show()
    #     print('debug img 0033')

    return origin, theta_x

    # return axes



if __name__ == "__main__":
    axes_dir = './axes'
    if not os.path.exists(axes_dir):
        os.mkdir(axes_dir)
    dir = './withAxes'
    origins = []
    theta_x = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.jpg'):
                print('Processing:', file)
                img_path = os.path.join(dir, file)
                # print(img_path)
                origin, theta = getOriginAndTheta(img_path)
                origins.append(origin)
                theta_x.append(theta)
    print(origins)
    print(theta_x)

