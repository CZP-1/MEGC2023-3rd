import os
import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
import samm_test_spot_util as fl
# from xlrd import xldate_as_tuple
from time import *
import pandas as pd

path_SAMM = "/home/data/CZP/MEGC/MEGC/data/test_set/SAMM/"
# path_SAMM = '/home/data2/CZP/2022MEGC-SPOT/SAMM-LongVideo/SAMM_longvideos/'
def multivideo_SAMM(path4):
    fileList = os.listdir(path4)
    
    print(fileList)
    

    
    # with open("my_samm_test.csv", "a", newline='') as f:
    #             writer = csv.writer(f)
    #             writer.writerows(["vid","pred_onset","pred_offset"])
    data = pd.DataFrame(columns=["vid","pred_onset","pred_offset","type"])

    j=0
    iou = 0.5
    fps = 200
    for vio in fileList:

        if(True):
        # if(vio=="001_2"):
        # if(vio=="002_4"):
        # if vio =='006_3':
        # if vio =='007_6':
            path5=vio
            #SAMM
            pp = fl.draw_roiline19(path_SAMM , path5 , -9, -4,7)  #18是直接使用光流计算，19是全部，20是去掉全局移动
            pp=pp*7
            pp = pp.tolist()
            pp.sort()
            # print("pp")
            print(pp)
            for i,interval in enumerate(pp):
                # if i>1:
                #     if pp[i][0]-pp[i-1][1]<fps:
                #         continue
                data.loc[i+j,'vid'] = vio
                data.loc[i+j,'pred_onset'] = interval[0]
                data.loc[i+j,'pred_offset'] = interval[1]
                if (interval[1]-interval[0])/fps > iou:
                    data.loc[i+j,'type'] = 'mae'
                else:
                    data.loc[i+j,'type'] = 'me'
            j=i+j+1
            # with open(".cmy_samm_testsv", "a", newline='') as f:
            #     writer = csv.writer(f)
            #     for interval in pp:
            #         writer.writerows([vio,str(int(interval[0])),str(int(interval[1]))])
    data.to_csv("/home/data/CZP/MEGC/MEGC/spot/submit/samm_pred_16p.csv",index=False)
multivideo_SAMM(path_SAMM)