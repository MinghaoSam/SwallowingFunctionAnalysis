import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def cv_show(title, img):
    cv2.imshow(title, img)
    # cv2.destroyAllWindows()


def processing(img):

    ## find the first non-zero pixel value
    # row_index = 249
    # image_matrix = np.array(r)
    # for col_index, pixel_value in enumerate(image_matrix[row_index]):
    #     if np.any(pixel_value != 0):  # check if the pixel value is non-zero
    #         x_start = col_index
    #         break
    x_start = 520
    x_end = 1920 - x_start
    img = img[0:, x_start:x_end]
    b, g, r = cv2.split(img)
    sub_res = cv2.subtract(r, b)



    # Median filter to smooth image
    median_kernel_size = 5
    median_blurred = cv2.medianBlur(sub_res, median_kernel_size)

    # cv_show('median filter', median_blurred)

    # Binary image
    binary_img = cv2.inRange(median_blurred, 220, 255)

    # plt.imshow(binary_img, cmap='gray')
    # plt.show()

    # 连通区域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    assert len(stats) == 2
    print('num_labels = ', num_labels)
    print('stats = ', stats)  # 5 cols: x, y, width, height, area

    # # 找到满足条件的行索引 i
    # condition = (stats[:, 4] >= 2500) & (stats[:, 4] <= 4400)
    # row_index = np.where(condition)[0][0]
    #
    # # row_index = np.argmax(stats[1:, 4]) + 1
    # target_label = row_index
    # mask = np.zeros_like(binary_image, dtype=np.uint8)
    # mask[labels == target_label] = 255

    # cv_show('mask', mask)
    # cv2.waitKey(0)
    return binary_img


if __name__ == '__main__':
    directory = ['./annotated/crx', './annotated/dw', './annotated/my']
    for i in range(3):
        print(directory[i])
        for root, directories, files in os.walk(directory[i]):
            mask_dir = directory[i] + '_mask'
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)
                print(mask_dir)
            for file in files:
                if file.endswith('.jpg'):
                    print(file)
                    image_path = os.path.join(root, file)
                    image = cv2.imread(image_path)
                    processed = processing(image)
                    cv2.imwrite(mask_dir + '/' + file, processed)
