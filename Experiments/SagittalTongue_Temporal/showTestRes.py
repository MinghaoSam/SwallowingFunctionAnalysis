import os
import pickle
import re
from matplotlib import pyplot as plt

dice_list = []
iou_list = []
# walk files
for root, dirs, files in os.walk("../SagittalTongue_Temporal"):
    # if 'UCTransNet' in root:
    if 'ACC' in root:
        for dir_ in dirs:
            if dir_ == 'visualize_test':
                # extract model name using re by find string between the second two '/'
                print(re.findall(r'Temporal/(.*?)/', root)[0]) # '.*' means any char, '?' means non-greedy, non-greedy means match as few as possible
                test_res_path = os.path.join(root, dir_)
                dice_list = []
                iou_list = []
                # print(test_res_path)
                for f_ in os.listdir(test_res_path):

                    if f_.endswith('.p'):
                        # load pickle file
                        data = pickle.load(open(os.path.join(test_res_path, f_), 'rb'))
                        # assign data to variables
                        input_ = data['input']
                        # rotate the input image
                        input_ = input_.transpose(1, 0, 2)
                        output_ = data['output']
                        ground_truth = data['ground_truth']
                        dice = data['dice']
                        iou = data['iou']
                        dice_list.append(dice)
                        iou_list.append(iou)
                        plt.subplot(1, 3, 1)
                        plt.imshow(input_)
                        plt.subplot(1, 3, 2)
                        plt.imshow(ground_truth, cmap='gray')
                        plt.subplot(1, 3, 3)
                        plt.imshow(output_, cmap='gray')
                        plt.suptitle(f'Dice score : {dice}\nIoU : {iou}')
                        plt.tight_layout()
                        plt.savefig(os.path.join(test_res_path, f_[:-2]))
                        plt.close()
                # calculate mean dice and iou
                dice_mean = sum(dice_list) / len(dice_list)
                iou_mean = sum(iou_list) / len(iou_list)
                print(f'dice_mean: {dice_mean}, iou_mean: {iou_mean}')


