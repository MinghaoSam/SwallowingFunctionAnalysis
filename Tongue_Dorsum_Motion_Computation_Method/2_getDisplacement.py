import os
import numpy as np
import getAxes
import featurePointCoor
import pandas as pd
import ast
import math
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import re



origin = []
theta_x = []
featurePoints = []

# csv_path = './Results/dw_data2.csv'

def writeFeaturePoints(axes_path: str, mask_path: str, csv_path: str): # write featurePoints original coordinates to csv
    # axes_path = '../dw矢状位标记图2/含坐标'  # forAxes
    # mask_path = './Results/dw_Mask2'  # forFeaturePoints
    # csv_path = './Results/dw_data2.csv'
    if not os.path.exists(csv_path):
        init_df = pd.DataFrame(columns=['img', 'origin', 'theta_x',	'point1', 'point2',	'point3',	'point4',	'point5',
                                        'point6',	'point7',	'point8',	'point9',	'point10',	'r1',	'r2',	'r3',	'r4',
                                        'r5',	'r6',	'r7',	'r8',	'r9',	'r10',	'cycle',	'pd1',	'pd2',	'pd3',	'pd4',
                                        'pd5',	'pd6',	'pd7',	'pd8',	'pd9',	'pd10',	'pv1',	'pv2',	'pv3',	'pv4',	'pv5',
                                        'pv6',	'pv7',	'pv8',	'pv9',	'pv10',	'pa1',	'pa2',	'pa3',	'pa4',	'pa5',	'pa6',
                                        'pa7',	'pa8',	'pa9',	'pa10'])
        init_df.to_csv(csv_path, index=False)
        print(f"Empty csv {csv_path}'has been created.")
    df = pd.read_csv(csv_path)
    img_list = []
    origin = []
    theta_x = []
    featurePoints = []
    for file in os.listdir(axes_path):
        print(f"getting Axes from {file}")
        img_list.append(file[:8])
        file_path = os.path.join(axes_path, file)
        # print(os.path.exists(file_path))
        # print(file_path)
        o, t = getAxes.getOriginAndTheta(file_path)
        origin.append(o)
        theta_x.append(t)
    # print('length of img_list: ', len(img_list))
    # print(img_list)
    # print('length of origin: ', len(origin))
    # print(origin)
    # print('length of theta_x: ', len(theta_x))
    # print(theta_x)
    df['img'] = img_list
    df['origin'] = origin
    df['theta_x'] = theta_x

    i = 0
    for mask_file in os.listdir(mask_path):
        print('Extracting Features Points: ', mask_file)

        img_path = os.path.join(mask_path, mask_file)
        # print(file_path)
        # img = plt.imread(img_path, 'gray')
        # plt.imshow(img)
        # plt.show()
        fp = featurePointCoor.getFeaturePoints(img_path, origin[i])
        print(fp)
        featurePoints.append(fp)
        i += 1
    print(featurePoints)
    # print(len(featurePoints))
    # print(len(featurePoints[0]))


    for img_index in range(len(img_list)):
        for i in range(1, 11):
            col_name = 'point'+str(i)
            df.at[img_index, col_name] = str(featurePoints[img_index][i-1])
    print(df.to_csv(csv_path, index=False))
    # print(featurePoints.shape)

def writeRelativeCoor(csv_path: str): # write relative coordinates to csv

    df = pd.read_csv(csv_path)
    for i in range(df.shape[0]):
        origin = ast.literal_eval(df.iloc[i, 1])
        # print(origin)
        theta_x = df.iloc[i, 2]
        # print(theta_x)
        for p in range(10):
            point = ast.literal_eval(df.iloc[i, 3+p])
            print(f'point:{point}\torigin:{origin}')
            # print(point)
            dy = (point[1] - origin[1]) * -1  # y axis is reversed in image
            dx = point[0] - origin[0]
            print(f'dy:{dy}\tdx:{dx}')
            rho = math.sqrt(dy**2 + dx**2)
            print(f'rho:{rho}')
            theta_p = np.arctan2(dy, dx)  # arctan2 returns value in range [-pi, pi]
            print('theta_p: ', theta_p)
            # if theta_p < 0:
            #     theta_p += math.pi
            # print(theta_p)
            # assert(theta_p > 0 and theta_p < math.pi)
            theta_r = theta_p - theta_x
            print('theta_r: ', theta_r)
            # assert (theta_r > 0 and theta_r < math.pi)
            x_r = round(rho * math.cos(theta_r), 2)
            y_r = round(rho * math.sin(theta_r), 2)
            df.at[i, 'r' + str(p + 1)] = str((x_r, y_r))
            df.to_csv(csv_path, index=False)

def showPointsInImg(csv_path: str):
    # csv_path = './Results/dw_data.csv'
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    df = pd.read_csv(csv_path)
    points = df.iloc[0, 13:]
    print(df.iloc[0, 0])
    for point in points:
        point = ast.literal_eval(point)
        x, y = point
        print(x, y)
        x = round(x) + 100
        y = round(500-y)
        cv.circle(canvas, (x, y), 3, (0, 255, 0), -1)
    plt.imshow(canvas)
    plt.show()

def writeDisplacement(csv_path: str, cycle_num: int):
    # csv_path = './Results/dw_data.csv'
    displacement = []
    one_cycle = []
    dis = []
    df = pd.read_csv(csv_path)
    rp = df.iloc[:, 13:23]
    for col_name, col_data in rp.items():  # iterate over col, from 'r1' to 'r10'
        # print(col_name)

        for i, val in col_data.items():  # iterate over each coordinate
            if i % 8 == 0:  # each swallowing cycle contains 8 photos
                t = ast.literal_eval(val)
                i += 1
                continue
            p = ast.literal_eval(val)
            x = p[0]
            y = p[1]
            dy = y - t[1]
            dx = x - t[0]
            dis.append(round(math.sqrt(dx**2 + dy**2), 2))
            t = p
            if len(dis) == 7:
                one_cycle.append(dis)
                dis = []
        displacement.append(one_cycle)
        one_cycle = []
    print(displacement[0])
    print(len(displacement))  # 10 feature points
    print(len(displacement[0])) # 4 swallowing cycles
    scale = 10/36 # 10mm for 36 pixels
    timeStep = 7 / 30


    for i in range(10):
        onePointDis = []
        for j in range(cycle_num):
            for k in range(7):
                dis = round(displacement[i][j][k] * scale, 2)
                df.at[j*7+k, 'pd'+str(i+1)] = dis
                df.at[j*7+k, 'pv'+str(i+1)] = round(dis/timeStep, 2)

    for i in range(10):
        flag = 0
        for j in range(cycle_num):
            for k in range(7):
                if flag % 7 == 0:
                    vPre = displacement[i][j][k] * scale / timeStep
                    flag += 1
                    continue
                else:
                    v = displacement[i][j][k] * scale / timeStep
                    print(v, vPre)
                    a = (v - vPre) / timeStep
                    vPre = v
                    flag += 1
                    df.at[j*6+k-1, 'pa'+str(i+1)] = round(a, 2)
    df.to_csv(csv_path, index=False)


def plotDisplacement(csv_path: str, save_path: str, cycle_num: int):
    # csv_path = './Results/dw_data.csv'
    displacement = []
    one_cycle = []
    dis = []
    df = pd.read_csv(csv_path)
    rp = df.iloc[:, 13:23]
    displacement = df.iloc[:, 24:34]
    velocity = df.iloc[:, 34:44]
    acceleration = df.iloc[:, 44:54]

    cmap = cm.get_cmap('tab20c')
    plt.figure(figsize=(7, 5))

    x_data_d = [1, 2, 3, 4, 5, 6, 7]

    # plot displacement
    for c in range(cycle_num): # 4 swallowing cycles
        for p in range(10): # 10 feature points
            color = cmap(p)
            y_data_d = np.array(displacement.loc[0+7*c:6+7*c, 'pd'+str(p+1)])
            plt.plot(x_data_d, y_data_d, marker='o', color=color, label='Point'+' '+str(p+1))
        # set font size
        # put a legend to the outside of the plot
        plt.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel('Time Step', fontsize=15)
        plt.ylabel('Displacement/mm', fontsize=15)
        plt.title('Cycle '+str(c+1)+' Feature Points Displacement', fontsize=15)
        plt.savefig(save_path+'/'+'Cycle '+str(c+1)+' Feature Points Displacement'+'.png', dpi = 600, bbox_inches = 'tight')

        plt.clf()

    # plot velocity
    x_data_v = x_data_d
    for c in range(cycle_num): # 4 swallowing cycles
        for p in range(10): # 10 feature points
            color = cmap(p)
            y_data_v = np.array(velocity.loc[0+7*c:6+7*c, 'pv'+str(p+1)])
            plt.plot(x_data_v, y_data_v, marker='o', color=color, label='Point'+' '+str(p+1))
        plt.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
        plt.xlabel('Time Step')
        plt.ylabel('Velocity (mm$\cdot$s$^{-1}$)')
        plt.title('Cycle '+str(c+1)+' Feature Points Velocity')
        plt.savefig(save_path+'/'+'Cycle '+str(c+1)+' Feature Points Velocity'+'.png', dpi = 600, bbox_inches = 'tight')
        plt.clf()


    ## plot acceleration
    x_data_a = [1, 2, 3, 4, 5, 6]
    for c in range(cycle_num): # 4 swallowing cycles
        for p in range(10): # 10 feature points
            color = cmap(p)
            y_data_a = np.array(acceleration.loc[6*c:5+6*c, 'pa'+str(p+1)])
            plt.plot(x_data_a, y_data_a, marker='o', color=color, label='Point'+' '+str(p+1))
        plt.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
        plt.xlabel('Time Step')
        plt.ylabel('acceleration (mm$\cdot$s$^{-2}$)')
        plt.title('Cycle '+str(c+1)+' Feature Points Acceleration')
        # plt.show()
        plt.savefig(save_path+'/'+'Cycle '+str(c+1)+' Feature Points Acceleration'+'.png', dpi = 600, bbox_inches = 'tight')
        plt.clf()




def visualizeSinglePointMovement(csv_path: str, save_path: str, cycle_num: int):
    print('Now visualizing single point movement...')
    # csv_path = './Results/dw_data.csv'
    # canvas = np.zeros((220, 250, 3), dtype=np.uint8)
    df = pd.read_csv(csv_path)
    all_points_df = df.iloc[:, 13:23]
    # print(points)
    plt.figure(figsize=(8, 8))

    for i in range(cycle_num):
        x = []
        y = []
        intensity = []
        points_df = all_points_df.iloc[i*8:i*8+8,:]
        print(points_df)
        for col_name, points in points_df.items():
            # point = ast.literal_eval(points[4])
            # print(point)
            # x = round(point[0]) + 50
            # y = round(200 - point[1])
            # canvas[y][x] = (255, 255, 255)

            for j in range(len(points)):
                # print(i)
                # print(points[i])
                # print(colors[i])
                point = ast.literal_eval(points[j+i*8])
                x.append(point[0])
                y.append(point[1])
                intensity.append((j+1) * 0.125)
        ax = plt.gca()  # get current axes 获得坐标轴对象
        ax.set_aspect(1)  # set height-width ratio to 1:1
        plt.scatter(x, y, c=intensity, cmap='viridis', s=10 )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)
        plt.title('Cycle '+str(i+1)+' Registered Feature Points Scatter')
        plt.savefig(save_path+'/'+'Cycle '+str(i+1)+' Registered Feature Points Scatter.png')
        plt.clf()
            # canvas[y][x] = (40+i*30, 40+i*30, 40+i*30)
    # canvas[:, 50, :] = [255, 0, 0]
    # canvas[209, :, :] = [255, 0, 0]
    # cv.imwrite('image.jpg', canvas)
    # plt.imshow(canvas)

def visualizeOriginPointMovement(csv_path: str, save_path: str, cycle_num: int):
    print('Now visualizing single point movement...')
    # csv_path = './Results/dw_data.csv'
    # canvas = np.zeros((220, 250, 3), dtype=np.uint8)
    df = pd.read_csv(csv_path)
    all_points_df = df.iloc[:, 3:13]
    # print(points)
    plt.figure(figsize=(8, 8))

    for i in range(cycle_num):
        x = []
        y = []
        intensity = []
        points_df = all_points_df.iloc[i*8:i*8+8,:]
        print(points_df)
        for col_name, points in points_df.items():
            # point = ast.literal_eval(points[4])
            # print(point)
            # x = round(point[0]) + 50
            # y = round(200 - point[1])
            # canvas[y][x] = (255, 255, 255)

            for j in range(len(points)):
                # print(i)
                # print(points[i])
                # print(colors[i])
                point = ast.literal_eval(points[j+i*8])
                x.append(point[0] - 100)
                y.append(700 - point[1])
                intensity.append((j+1) * 0.125)
        ax = plt.gca()  # get current axes 获得坐标轴对象
        ax.set_aspect(1)  # set height-width ratio to 1:1
        plt.scatter(x, y, c=intensity, cmap='viridis', s=10 )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)
        plt.title('Cycle '+str(i+1)+' Original Feature Points Scatter')
        plt.savefig(save_path+'/'+'Cycle '+str(i+1)+' Original Feature Points Scatter.png')
        plt.clf()


if __name__ == '__main__':
    dirs_list = []
    subject_list = ['my']
    # subject_list = ['sl']
    for subject in subject_list:
        if '2' in subject:
            dirs_list.append('../images/'+subject[:-1]+'矢状位标记图2')
        else:
            dirs_list.append('../images/'+subject+'矢状位/'+subject+'矢状位标记图')
    print(dirs_list)

    for im_dir in dirs_list:
        patient_id = re.findall('[a-zA-Z]+', im_dir)[1]
        axes_path = im_dir+'/含坐标'
        cycle_num = len(os.listdir(axes_path)) // 8
        if '2' in im_dir:
            mask_path = './Results/mask/'+patient_id+'_Mask2'
            csv_path = './Results/data_csv//'+patient_id+'_data2.csv'
            save_path = './Results/figs/'+patient_id+'2'
        else:
            mask_path = './Results/mask/'+patient_id+'_Mask'
            csv_path = './Results/data_csv/'+patient_id + '_data.csv'
            save_path = './Results/figs/'+patient_id
            # print('csv_path: ', csv_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            print('Created a folder:', save_path)
        print('Processing: ', save_path[15:])


        writeFeaturePoints(axes_path, mask_path, csv_path)
        writeRelativeCoor(csv_path)
        visualizeSinglePointMovement(csv_path, save_path, cycle_num)
        writeDisplacement(csv_path, cycle_num)
        plotDisplacement(csv_path, save_path, cycle_num)
        visualizeOriginPointMovement(csv_path, save_path, cycle_num)

    # writeRelativeCoor('./Results/wtt_data3.csv')
    # writeDisplacement('./Results/wtt_data3.csv', 1)