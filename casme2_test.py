import sim_filter
import os
import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
import casme2_test_spot_util as fl
import pandas as pd


print("-------------------------")

path4="/home/data/CZP/MEGC/MEGC/data/test_set/CAS_crop_align/"
# path4 = '/home/data2/CZP/2022MEGC-SPOT/CASME2/rawpic/s15/'
def merge(pp):
    re=[]
    fps=30
    for i in range(len(pp)-1):
        if abs(pp[i][1]-pp[i+1][0])<=fps*0.5:
            re.append([pp[i][0],pp[i+1][1]])
            i+=1
        else:
            re.append(pp[i])
    if not i==range(len(pp)-1):
        re.append(pp[-1])
    return re
            
        


def multivideo_CASME(path4):
    fileList = os.listdir(path4)
    j=0
    iou = 0.5
    fps = 30
    data = pd.DataFrame(columns=["vid","pred_onset","pred_offset"])

    for sub in fileList:
        # if not sub == "dzw_f3":
        #     continue
        # if not sub == '15_0102eatingworms':
        #     continue
        path5=sub
        print(path5)
        # CAS(ME)2
        # pp = fl.draw_roiline18(path4 , path5 , 4, -4,1)
        
        pp = fl.draw_roiline19(path4 , path5 , 0, -4,1)
        pp = pp.tolist()
        pp.sort()
        # print(pp)
        # pp=merge(pp)
        # print("pp")
        print(pp)
        # for p in pp:
        #     if p[0]<100:
        #         pp.remove(p)
        for i,interval in enumerate(pp):
            # if i>1:
            #     if pp[i][0]-pp[i-1][1]<fps:
            #         continue
            data.loc[i+j,'vid'] = sub
            data.loc[i+j,'pred_onset'] = interval[0]
            data.loc[i+j,'pred_offset'] = interval[1]
            if (interval[1]-interval[0])/fps > iou:
                data.loc[i+j,'type'] = 'mae'
            else:
                data.loc[i+j,'type'] = 'me'
            
        j=i+j+1
    data.to_csv("/home/data/CZP/MEGC/MEGC/spot/submit/cas_pred_crop_3p_c_0.2.csv",index=False)   
multivideo_CASME(path4)