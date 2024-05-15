import os
import pickle
import cv2
# target_model = ['DSCNet', 'TTUNet_with_TRPE', 'UNet_base', 'UCTransNet']
target_model = ['ACC_UNet']
for _f in os.listdir('./'):
    if _f in target_model:
        print(f'Processing: {_f}')
        _path = './' + _f + '/session/visualize_test'
        for _p in os.listdir(_path):
            if _p.endswith('.p') and _p.startswith('yl'):
                print(_p)
                res_data = pickle.load(open(_path+'/'+_p, 'rb'))
                _input = res_data['input']*255
                _input = _input.transpose(1, 0, 2)
                output = res_data['output']*255
                ground_truth = res_data['ground_truth']*255
                if not os.path.exists(_path+'/yl_res'):
                    os.mkdir(_path+'/yl_res')
                print(_path+'/yl_res/_input_'+_p[:-2])
                cv2.imwrite(_path+'/yl_res/input_'+_p[:-2], _input)
                cv2.imwrite(_path+'/yl_res/output_'+_p[:-2], output )
                cv2.imwrite(_path+'/yl_res/GT_'+_p[:-2], ground_truth)
                print('one sample written!')
