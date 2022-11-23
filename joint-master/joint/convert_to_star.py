import numpy as np 
import glob 
import os
import pandas as pd 

# all_scores = glob.glob('/hpc/home/qh36/research/qh36/3D_picking/joint_denoising_detection/new_erica_out/00004-eval-ssdn-gaussian-iter250k-0.75-0.05-joint/eval_imgs/*.txt')
# all_scores = glob.glob('/hpc/home/qh36/research/qh36/3D_picking/joint_denoising_detection/hi_runs/00020-eval-10249-10249-ssdn-iter400k-0.2-joint/eval_imgs/*.txt')
# all_scores = glob.glob('/hpc/home/qh36/research/qh36/3D_picking/joint_denoising_detection/hi_runs/00019-eval-10249-10249-ssdn-iter400k-0.2-joint/eval_imgs/*.txt')
all_scores = glob.glob('/hpc/home/qh36/research/qh36/3D_picking/joint_denoising_detection/new_10215_new_eval/00024-eval-ssdn-gaussian-iter250k-0.95-0.05-joint/eval_imgs/*.txt')
with open('10215_nms20_05_095_full.star','w') as f:
    f.write('# version 30001\n\ndata_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnMicrographName #3\n_rlnAutopickFigureOfMerit #4\n')
    for sc in all_scores:
        name = os.path.basename(sc)
        name = name[:-18]
        name = name + '.mrc'
        # print(name)
        coords1 = pd.read_csv(sc, sep='\t')
        # coords1 = coords1.loc[coords1.x_coord > 20]
        # coords1 = coords1.loc[coords1.x_coord < 910]
        # coords1 = coords1.loc[coords1.y_coord < 935]
        # coords1 = coords1.loc[coords1.y_coord > 20]
        # if np.sum(coords.score > 0.43) < 500:
        #     thresh = 0.42
        # elif np.sum(coords.score > 0.435) < 500 and np.sum(coords.score > 0.43) > 500:
        #     thresh = 0.43 

        # else:
        #     thresh = 0.435
        # std_p = np.std(coords1.score)
        # coords1.score = (coords1.score - np.mean(coords1.score))/(np.std(coords1.score))
        # if std_p > 0.008:
        #     thres = 0.5
        # else:
        #     thres = 0.3
        thres = 0.13
        # print(np.max(coords1.score))
        # up_thresh = np.max(coords1.score)*0.9
        # print(coords1)
        # quant = np.quantile(coords1.score, 0.98)
        # thresh = np.median(coords1.score) + 0.25* np.std(coords1.score)
        # tot_sum = np.sum(coords1.score > 0.25)
        # if np.sum(coords1.score > 0.17) - tot_sum > 400:
        #     thresh = 0.18
        # elif np.sum(coords1.score > 0.17) - tot_sum < 300 and np.sum(coords1.score > 0.17) - tot_sum > 100:
        #     thresh = 0.17
        # elif np.sum(coords1.score > 0.17) - tot_sum <100:
        #     if np.sum(coords1.score > 0.16) - tot_sum > 120:
        #         thresh = 0.16
        #     else:
        #         thresh = 0.15
        # if np.sum(coords.score > 0.16) < 500:
        #     thresh = 0.15 
        # elif np.sum(coords.score > 0.18) < 500 and np.sum(coords.score > 0.16) > 500:
        #     thresh = 0.16 

        # else:
        #     thresh = 0.18
        # all_scores = []
        # for x, y, s in zip(coords.x_coord, coords.y_coord, coords.score):
        #   all_scores.append(s)
        # thresh = np.quantile(all_scores, 0.99)
        for x, y, s in zip(coords1.x_coord, coords1.y_coord, coords1.score):
            # if s > thresh and s < 0.27:
            if s > thres:
        # if x > 15 and x < 1425 and y > 15 and y < 1009:
        #     if abs(y - 64) > 6:
        #         if abs(x-64) > 6 and abs(1440-64-x) > 6:
                # if x > 15 and x < 913 and y > 15 and y < 945:
            # if s < 0.3 and s > 0.24:
            #     if x > 15 and x < 1425 and y > 15 and y < 1009:
            #         if abs(y - 64) > 6:
            #             if abs(x-64) > 6 and abs(1440-64-x) > 6:
            # if s < 0.4 and s > 0.3:
                # if x > 10 and x < 918 and y > 10 and y < 950:
            # if s < 1.5 and s > 0.85:
                if x > 15 and x < 1425 and y > 15 and y < 1009:
                # if x > 10 and x < 918 and y > 10 and y < 950:
                # if x > 32 and x < 928-24 and y > 32 and y < 960-24:
                    f.write(str(int((x)*4)) + '\t' + str(int((y)*4)) + '\t' + name + '\t' + str(s) + '\n')