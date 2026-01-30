import cv2
import numpy as np
import math
import pandas as pd

from pathlib import Path
from collections import defaultdict

def checkerboard():
    m_img_path = Path('E:\\workspace\\Python\\PARNet6\\checkpoint-kidney3-PMAR2\\reg_test_img\\CMP-UP\\NI_ZHUOWEI_ZS18341495\\38_m.jpg')
    our_t_img_path = Path('E:\\workspace\\Python\\PARNet6\\checkpoint-kidney3-PMAR2\\reg_test_img\\CMP-UP\\NI_ZHUOWEI_ZS18341495\\38_t.jpg')
    out_t_nocontent_path = Path('E:\\workspace\\Python\\PARNet6\\wo-content\\reg_test_img\\CMP-UP\\NI_ZHUOWEI_ZS18341495\\38_t.jpg')
    cycgan_t_img_path = Path('E:\\workspace\\Python\\MultiRegEval-master\\test_T_jpg\\cycle_GAN\\CMP-UP\\NI_ZHUOWEI_ZS18341495\\38_t.jpg')
    p2p_img_path = Path('E:\\workspace\\Python\MultiRegEval-master\\test_T_jpg\\pix2pix\\CMP-UP\\NI_ZHUOWEI_ZS18341495\\38_t.jpg')
    nemar_img_path = Path('E:\\workspace\\Python\\nemar-nemar_deploy_3d\\reg_test_img\\CMP-UP\\NI_ZHUOWEI_ZS18341495\\38_t.jpg')
    out_img_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic')

    m_img = cv2.imread(str(m_img_path), cv2.IMREAD_GRAYSCALE)
    t_img = cv2.imread(str(our_t_img_path), cv2.IMREAD_GRAYSCALE)

    patch_num = (10, 10)

    h, w = m_img.shape[-2], m_img.shape[-1]
    p_h = math.ceil(h / patch_num[0])
    p_w = math.ceil(w / patch_num[1])
    out_img = np.zeros((h, w))
    start_h = 0
    for i in range(p_h):
        start_w = 0
        end_h = start_h + p_h
        for j in range(p_w):
            end_w = start_w + p_w
            if (i%2 == 0 and j%2 == 0) or (i%2 == 1 and j%2 == 1):
                out_img[start_h:end_h, start_w:end_w] = m_img[start_h:end_h, start_w:end_w]
            else:
                out_img[start_h:end_h, start_w:end_w] = t_img[start_h:end_h, start_w:end_w]
            start_w = end_w
        start_h = end_h
    
    out_img = cv2.resize(out_img, (256,256))
    cv2.imwrite(str(out_img_path / 'checker_mt.jpg'), out_img)


def plot_reg():
    m_img_path = Path('E:\\workspace\\Python\\PARNet6\\checkpoint-kidney3-PMAR2\\reg_test_img\\CMP-UP\\JI_MIN_ZS17134069\\34_m.jpg')
    f_img_path = Path('E:\\workspace\\Python\\PARNet6\\checkpoint-kidney3-PMAR2\\reg_test_img\\CMP-UP\\JI_MIN_ZS17134069\\34_f.jpg')
    our_img_path = Path('E:\\workspace\\Python\\PARNet6\\checkpoint-kidney3-PMAR2\\reg_test_img\\CMP-UP\\JI_MIN_ZS17134069\\34_w.jpg')
    cycreg_img_path = Path('E:\\workspace\\Python\\MultiRegEval-master\\test_T_jpg\\cycle_GAN\\CMP-UP\\NI_ZHUOWEI_ZS18341495\\38_t.jpg')
    pdd_img_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\PDDNet\\reg_test_img\\CMP-UP\\JI_MIN_ZS17134069\\34_w.jpg')
    rcn_img_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\RCN\\reg_test_img\\CMP-UP\\JI_MIN_ZS17134069\\34_w.jpg')
    nemar_img_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\NeMAR\\reg_test_img\\CMP-UP\\JI_MIN_ZS17134069\\34_w.jpg')
    out_img_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic')

    f_img = cv2.imread(str(f_img_path), cv2.IMREAD_GRAYSCALE).astype('float')
    w_img = cv2.imread(str(our_img_path), cv2.IMREAD_GRAYSCALE).astype('float')

    w, h = f_img.shape[-2], f_img.shape[-1]
    diff = np.zeros((w, h, 3))
    # diff[:,:,2] = np.abs(f_img - w_img)

    tmp = np.abs(f_img - w_img)
    diff[:,:,0] = (tmp<=85) * (tmp / 85) * 255
    diff[:,:,1] = (tmp>=85) * (tmp<=170) * ((tmp-85) / 85) * 255
    diff[:,:,2] = (tmp>170) * ((tmp-170) / 85) * 255

    out_img = cv2.resize(diff, (256,256))
    cv2.imwrite(str(out_img_path / 'diff_wf.jpg'), out_img)   


def plot_reg3():
    f_img_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic\\CMP_UP-PAN_ZHIYU_ZS21125512-52\\52_f.jpg')
    f_lab_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic\\CMP_UP-PAN_ZHIYU_ZS21125512-52\\52_f_label.png')
    
    m_img_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic\\CMP_UP-PAN_ZHIYU_ZS21125512-52\\52_m.jpg')
    m_lab_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic\\CMP_UP-PAN_ZHIYU_ZS21125512-52\\52_m_label.png')
    
    our_img_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic\\CMP_UP-PAN_ZHIYU_ZS21125512-52\\52_our_new.jpg')
    our_lab_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic\\CMP_UP-PAN_ZHIYU_ZS21125512-52\\52_our_label_new.png')

    cycreg_img_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic\\CMP_UP-PAN_ZHIYU_ZS21125512-52\\52_cycreg.jpg')
    cycreg_lab_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic\\CMP_UP-PAN_ZHIYU_ZS21125512-52\\52_cycreg_label.png')

    nice_img_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic\\CMP_UP-PAN_ZHIYU_ZS21125512-52\\52_nice.jpg')
    nice_lab_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic\\CMP_UP-PAN_ZHIYU_ZS21125512-52\\52_nice_label.png')
    
    rcn_img_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic\\CMP_UP-PAN_ZHIYU_ZS21125512-52\\52_rcn.jpg')
    rcn_lab_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic\\CMP_UP-PAN_ZHIYU_ZS21125512-52\\52_rcn_label.png')

    nemar_img_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic\\CMP_UP-PAN_ZHIYU_ZS21125512-52\\52_nemar.jpg')
    nemar_lab_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic\\CMP_UP-PAN_ZHIYU_ZS21125512-52\\52_nemar_label.png')

    out_img_path = Path('C:\\Users\\Admin\\Desktop\\Rgistration2\\expdata\\pic')

    f_lab = cv2.imread(str(f_lab_path)).astype('float')
    f_lab = f_lab / f_lab.max() *255

    w_img = cv2.imread(str(our_img_path)).astype('float')
    w_lab = cv2.imread(str(our_lab_path)).astype('float')
    w_lab = w_lab / w_lab.max() *255
    w_lab = w_lab[:,:,[0,2,1]]

    overlapping = cv2.addWeighted(w_img, 0.91, w_lab, 0.09, 0)
    # overlapping = cv2.imread(str(f_img_path)).astype('float')
    overlapping = cv2.addWeighted(overlapping, 0.9, f_lab, 0.1, 0)

    # overlapping = cv2.addWeighted(w_img, 0.75, w_lab, 0.25, 0)
    # overlapping = cv2.addWeighted(overlapping, 0.8, f_lab, 0.2, 0)
    
    overlapping = cv2.resize(overlapping, (256,256))
    cv2.imwrite(str(out_img_path / 'diff_wf.jpg'), overlapping)  


def static2excel_dvf(root_path):

    data_dict = defaultdict(dict)
    for dir in root_path.glob("*"):
        if not dir.is_dir():
            continue

        for pic in dir.rglob("*.jpg"):
            if pic.name.find('e')<0:
                continue
            p_id = pic.parent.name
            direct = pic.parent.parent.name

            pic_name_arr = pic.stem.split('_')
            ind = pic_name_arr[0]
            flow_id = pic_name_arr[2]
            err = eval(pic_name_arr[3])

            key_name = f'{direct}|{p_id}|{ind}'
            
            data_dict[f'{flow_id}'][key_name] = err

            w_lab_name = f'{ind}_w_l_*'
            w_lab_path = list(pic.parent.glob(w_lab_name))
            dice_value = 0
            if len(w_lab_path) >0:
                val = w_lab_path[0].stem.split('_')[-1]
                dice_value = eval(val)
            data_dict['dice'][key_name] = dice_value

    alg_names = data_dict.keys()
    record = []
    for key in data_dict['dice'].keys():
        key_arr = key.split('|')
        direct = key_arr[0]
        p_id = '_'.join(key_arr[1:-1])
        ind = key_arr[-1]
        # direct, p_id, ind = key.split('_')
        tmp_list = [direct, p_id, ind]
        for k in list(data_dict.keys()):
            tmp_list.append(data_dict[k][key])

        record.append(tmp_list)
    
    column_names = ['direct','pid','index']
    for k in list(data_dict.keys()):
        column_names.append(f'{k}')
    df = pd.DataFrame(record, columns=column_names)
    df.to_excel(str(root_path/'record-abdomen.xlsx'), index=False)

if __name__ == '__main__':
    # checkerboard()
    # plot_reg() 
    # plot_reg2() 
    # plot_reg3()
    static2excel_dvf(Path('E:\workspace\Python\PARNet6\\test_dvf_img\\new3-brats'))