import os
import cv2
import numpy as np

def processing(img_path):
    # 读取原始图像和二值图像
    original_image = cv2.imread(img_path)
    binary_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    img_array = np.array(binary_image)
    height, width = img_array.shape
    flag1 = 0
    for x in range(width):
        if flag1 == 1:
            break
        for y in range(height-1, -1, -1):
            if img_array[y, x] != 0:
                start_point = (x, y)
                flag1 = 1
                break

    flag2 = 0
    for y in range(height-1, -1, -1):
        if flag2 == 1:
            break
        for x in range(width):
            if img_array[y, x] != 0:
                end_point = (x, y)
                flag2 = 1
                break

    endpoints = [start_point, end_point]
    print(endpoints)

    # 在原始图像上绘制红色端点
    marked_image = original_image.copy()
    for point in endpoints:
        cv2.circle(marked_image, point, 5, (0, 0, 255), -1)
    return marked_image
    ## 显示标记后的图像
    cv2.imshow("Marked Image", marked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getEndpoints(bin_img):
    img_array = np.array(bin_img)
    height, width = img_array.shape
    # flag1 = 0
    # for x in range(width):
    #     if flag1 == 1:
    #         break
    #     for y in range(height-1, -1, -1):
    #         if img_array[y, x] == 255:
    #             start_point = (x, y)
    #             flag1 = 1
    #             break
    #
    # flag2 = 0
    # for y in range(height-1, -1, -1):
    #     if flag2 == 1:
    #         break
    #     for x in range(width):
    #         if img_array[y, x] == 255:
    #             end_point = (x, y)
    #             flag2 = 1
    #             break

    flag1 = 0
    for x in range(width):
        if flag1 == 1:
            break
        for y in range(height-1, -1, -1):
            if img_array[y, x] != 0:
                start_point = (x, y)
                flag1 = 1
                break

    flag2 = 0
    for y in range(height-1, -1, -1):
        if flag2 == 1:
            break
        for x in range(width):
            if img_array[y, x] != 0:
                end_point = (x, y)
                flag2 = 1
                break
    print('extracted endpoints: ', start_point, end_point)

    endpoints = [start_point, end_point]
    return endpoints


if __name__ == '__main__':
    directory = ['../Results/mask/sl_Mask']
    for i in range(len(directory)):
        print(directory[i])
        for root, directories, files in os.walk(directory[i]):
            print(directory[i][:-5] + '_markEndpoints')
            mark_dir = directory[i][:-5] + '_markEndpoints'
            if not os.path.exists(mark_dir):
                os.makedirs(mark_dir)
                print(mark_dir)
            for file in files:
                if file.endswith('.jpg'):
                    print(file)
                    image_path = os.path.join(root, file)
                    processed = processing(image_path)
                    cv2.imwrite(mark_dir + '/' + file, processed)