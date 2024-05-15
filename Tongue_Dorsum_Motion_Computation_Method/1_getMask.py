import os
import numpy as np
import sys
import cv2
import re

sys.path.append('./utils/')
import extractMask as em


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


if __name__ == '__main__':
    name_list = ['my', 'my2', 'dw', 'dw2', 'wtt', 'wtt2']
    dirs_list = []
    for name in name_list:
        if '2' in name:
            name = name[:-1]
            dirs_list.append('../images/' + name + '矢状位标记图2/')
        else:
            dirs_list.append('../images/' + name + '矢状位/' + name + '矢状位标记图')


    for f in dirs_list:
        patientName = re.findall('[a-zA-z]+', f)[1]
        print(patientName)
        if '2' in f:
            mask_folder = './Results/mask/' + patientName + '_Mask2'
        else:
            mask_folder = './Results/mask/' + patientName + '_Mask'
        print(f'mask_folder: {mask_folder}')
        if not os.path.exists(mask_folder):
            os.mkdir(mask_folder)
            print('Created a folder:', mask_folder)
        annotatedPath = f
        print('Processing:', annotatedPath)
        for file in os.listdir(annotatedPath):
            if file.endswith('.jpg') :
                img_path = annotatedPath + '/' + file
                image = cv_imread(img_path)
                print(image.shape)
                processed = em.processing(image)
                file = file[:-5] + '.jpg'
                print(file)
                cv2.imwrite(mask_folder + '/' + file, processed)
                print('Saved:', mask_folder + '/' + file)
