import sys
import subprocess
import random
import gzip
import csv
import math
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.spatial.distance as ssd
from multiprocessing import Pool

def read_sift_file(sift_file) :
    sift_ls = [[],[],[],[],[],[]]
    with open(sift_file,'r') as f:
        while(True) :
            f_line = f.readline().strip()
            if f_line[0:5] == "CHROM" :
                continue
            if f_line == "" :
                break
            f_ls = f_line.split('\t')
            if len(sift_ls[0]) == 0 :
                sift_ls[0].append(int(f_ls[1]))
                sift_ls[1].append(f_ls[2])
                sift_ls[2].append(f_ls[3])
                sift_ls[3].append(f_ls[8])
                if f_ls[12] == "NA" :
                    sift_ls[4].append(0)
                else :
                    sift_ls[4].append(float(f_ls[12]))
                sift_ls[5].append(f_ls[16])
            elif sift_ls[0][-1] == f_ls[1] and sift_ls[1][-1] == f_ls[2] and sift_ls[2][-1] == f_ls[3] :
                if f_ls[12] != "NA" :
                    sift_ls[0][-1] = int(f_ls[1])
                    sift_ls[1][-1] = f_ls[2]
                    sift_ls[2][-1] = f_ls[3]
                    sift_ls[3][-1] = f_ls[8]
                    sift_ls[4][-1] = float(f_ls[12])
                    sift_ls[5][-1] = f_ls[16]
            else :
                sift_ls[0].append(int(f_ls[1]))
                sift_ls[1].append(f_ls[2])
                sift_ls[2].append(f_ls[3])
                sift_ls[3].append(f_ls[8])
                if f_ls[12] == "NA" :
                    sift_ls[4].append(0)
                else :
                    sift_ls[4].append(float(f_ls[12]))
                sift_ls[5].append(f_ls[16])
        f.close()
    return sift_ls

# 以下代码是一个物种使用不同工具
def write_res_csv(res_folder) :
    af_rate = 0.001
    quality_index_ls = [1,3,5,6,7,8,9,10,11,12]
    res_pos_list_ls,res_sample_list_ls = [],[]
    impute_tools = ['beagle4','beagle5','impute5','minimac4']
    #impute_tools = ['beagle4','beagle5']
    colors_ls = ['y','g','b','r']
    title_ls = ["All Concordance","MAF Concordance","TP SNP Number","Mean Hellinger Score","Min Hellinger Score","Men SEN Score","Min SEN Score","IQS","R2","Var rate"]
    
    for tools in impute_tools :
        res_file = res_folder + "/data/res."+tools+"."+str(af_rate)[2:]+".txt"
        res_pos_list,res_sample_list = read_res_file(res_file)
        res_pos_list_ls.append(res_pos_list)
        res_sample_list_ls.append(res_sample_list)
        print("结果文件读取完毕：",tools)

    for k in range(len(quality_index_ls)) :
        quality_index = quality_index_ls[k]
        tools_x_ls = []
        tools_y_ls = []
        for j in range(len(impute_tools)) :
            res_pos_list,res_sample_list = res_pos_list_ls[j],res_sample_list_ls[j]
            x_ls = []
            y_1_ls = []
            y_2_ls = []
            for i in range(0,len(res_pos_list)) :
                if len(res_pos_list[i][quality_index]) == 0 or (quality_index == 1 and sum(res_pos_list[i][2]) == 0) or (quality_index == 3 and sum(res_pos_list[i][4]) == 0):
                    if i != 0 :
                        x_ls.append(round(i*af_rate,4))
                        y_1_ls.append(0)
                        y_2_ls.append(0)
                    else :
                        x_ls.append(round(i*af_rate,4))
                        y_1_ls.append(0)
                        y_2_ls.append(0)
                else :
                    x_ls.append(round(i*af_rate,4))
                    if quality_index == 1 :
                        y_1_ls.append(sum(res_pos_list[i][1]))
                        y_2_ls.append(sum(res_pos_list[i][2]))
                    elif quality_index == 3 :
                        y_1_ls.append(sum(res_pos_list[i][3]))
                        y_2_ls.append(sum(res_pos_list[i][4]))
                    elif quality_index == 5 :
                        y_1_ls.append(sum(res_pos_list[i][5]))
                        y_2_ls.append(sum(res_pos_list[i][5]))
                    else :
                        y_1_ls.append(sum(res_pos_list[i][quality_index]))
                        y_2_ls.append(len(res_pos_list[i][quality_index]))
            tools_x_ls = x_ls
            tools_y_ls.append(y_1_ls)
            tools_y_ls.append(y_2_ls)
        csv_file = "res_csv/res_" + str(quality_index) + ".csv"
        
        with open(csv_file,'w') as f:
            write_line = "af_rate"
            for i in range(len(impute_tools)) :
                write_line = write_line + "," + impute_tools[i] + "_1," +  impute_tools[i] + "_2" 
            f.write(write_line+"\n")
            for i in range(len(tools_x_ls)) :
                write_line = str(tools_x_ls[i])
                for j in range(len(tools_y_ls)) :
                    write_line = write_line + "," + str(tools_y_ls[j][i])
                write_line = write_line + '\n'
                f.write(write_line)
            f.close()
    return

def write_bar_res_csv(ethnic_type,res_folder,downsamlpe_type,af_source,impute_tools,tools_names,res_csv) :
    #impute_tools = ['synthetize','beagle4','beagle5','impute5','minimac4']
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_file = ['global','africa','america','centralsouthasia','eastasia','europe','middleeast','oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    # af_rate单位是0.001，af的列表对应为[0,0.001,0.005,0.05,1]
    af_rate = 0.001
    af_interval_ls = [0,1,5,50,100]
    bar_csv_ls = []
    res_pos_list_ls = []
    for tools in impute_tools :
        res_file = res_folder + "/res."+tools+"."+str(af_rate)[2:]+".txt"
        print("将要读取结果文件：",res_file)
        res_pos_list,res_sample_list = read_res_file(res_file)
        res_pos_list_ls.append(res_pos_list)
        #print("结果文件读取完毕：",res_file)
    for i in range(len(af_interval_ls)-1) :
        for j in range(len(impute_tools)) :
            sift_file = res_folder[:res_folder.rfind("/")]
            sift_file = sift_file[:sift_file.rfind("/")]
            if downsamlpe_type == "all" :
                sift_file = sift_file+"/sift/"+ethnic_file[0]+"."+impute_tools[j]+"/hgdp.phased."+ethnic_file[0]+".common.out."+impute_tools[j]+"_SIFTannotations.xls"
            else :
                sift_file = sift_file+"/sift/"+ethnic_file[ethnic_groups.index(ethnic_type)]+"."+impute_tools[j]+"/hgdp.phased."+ethnic_file[ethnic_groups.index(ethnic_type)]+".common.out."+impute_tools[j]+"_SIFTannotations.xls"
            sift_ls = read_sift_file(sift_file)
            res_pos_list = res_pos_list_ls[j]
            well_var_num = 0
            missense_var_num = 0
            plof_var_num = 0
            deleterious_var_num = 0
            for k in range(len(res_pos_list)) :
                if k < af_interval_ls[i] or k >= af_interval_ls[i+1] :
                    continue
                well_var_num += sum([1 if r2 >= 0.8 else 0 for r2 in res_pos_list[k][11]])
                for h in range(len(res_pos_list[k][0])) :
                    if res_pos_list[k][0][h] in sift_ls[0] :
                        sift_index = sift_ls[0].index(res_pos_list[k][0][h])
                        if sift_ls[1][sift_index] == res_pos_list[k][5][h] and sift_ls[2][sift_index] == res_pos_list[k][6][h] :
                            if sift_ls[3][sift_index] != "NA" and sift_ls[3][sift_index] != "NONCODING" and res_pos_list[k][11][h] > 0.8:
                                missense_var_num += 1
                            if sift_ls[4][sift_index] > 0.9 and res_pos_list[k][11][h] > 0.8:
                                plof_var_num += 1
                            if sift_ls[5][sift_index] == "DELETERIOUS" and res_pos_list[k][11][h] > 0.8:
                                deleterious_var_num += 1
            if i == 0 :
                bar_csv_ls.append(["AF<"+str(af_interval_ls[i+1]*af_rate*100),tools_names[j],well_var_num,missense_var_num,plof_var_num,deleterious_var_num])
            elif i == len(af_interval_ls)-2 :
                bar_csv_ls.append(["AF>="+str(af_interval_ls[i]*af_rate*100),tools_names[j],well_var_num,missense_var_num,plof_var_num,deleterious_var_num])
            else :
                bar_csv_ls.append([str(af_interval_ls[i]*af_rate*100)+"<=AF<"+str(af_interval_ls[i+1]*af_rate*100),tools_names[j],well_var_num,missense_var_num,plof_var_num,deleterious_var_num])
    csv_file = res_csv+"/res_bar_"+af_source+"_"+downsamlpe_type+"_"+ethnic_show[ethnic_groups.index(ethnic_type)]+"_alltools.csv"
    with open(csv_file,'w') as f:
        write_line = "af_rate,impute_tool,well_var_num,missense_var_num,plof_var_num,deleterious_var_num"
        f.write(write_line+"\n")
        for i in range(len(bar_csv_ls)) :
            write_line = ""
            for j in range(len(bar_csv_ls[i])) :
                write_line = write_line + str(bar_csv_ls[i][j]) + ","
            write_line = write_line[:-1] + "\n"
            f.write(write_line)
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_line_res_csv(ethnic_type,res_folder,downsamlpe_type,af_source,calcul_type,impute_tools,tools_names,res_csv) :
    #impute_tools = ['synthetize','beagle4','beagle5','impute5','minimac4']
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_file = ['global','africa','america','centralsouthasia','eastasia','europe','middleeast','oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    # af_rate单位是0.001，af的列表对应为[0,0.001,0.005,0.05,1]
    af_rate = 0.001
    #af_interval_ls = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,1]
    if calcul_type == 1 :
        af_interval_ls = [0,1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,1000]
    else :
        af_interval_ls = [0,1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,950,980,990,995,998,999,1000]
    line_csv_ls = []
    res_pos_list_ls = []
    for tools in impute_tools :
        res_file = res_folder + "/res."+tools+"."+str(af_rate)[2:]+".txt"
        print("将要读取结果文件：",res_file)
        res_pos_list,res_sample_list = read_res_file(res_file)
        res_pos_list_ls.append(res_pos_list)
        #print("结果文件读取完毕：",res_file)
    for i in range(len(af_interval_ls)-1) :
        csv_rate_ls = []
        for j in range(len(impute_tools)) :
            res_pos_list = res_pos_list_ls[j]
            right_num = 0
            total_num = 0
            maf_sum = 0
            maf_num = 0
            IQS_ls = []
            R2_ls = []
            for k in range(len(res_pos_list)) :
                if k < af_interval_ls[i] or k >= af_interval_ls[i+1] :
                    continue
                right_num += sum(res_pos_list[k][3])
                total_num += sum(res_pos_list[k][4])
                for h in range(len(res_pos_list[k][3])) :
                    if res_pos_list[k][4][h] == 0 :
                        maf_sum += 0
                        maf_num += 1
                    else :
                        maf_sum += res_pos_list[k][3][h] / res_pos_list[k][4][h]
                        maf_num += 1
                IQS_ls.extend(res_pos_list[k][10])
                R2_ls.extend(res_pos_list[k][11])
            if maf_num == 0:
                csv_rate_ls.append(0)
            else :
                csv_rate_ls.append(maf_sum / maf_num)
            if len(IQS_ls) == 0:
                csv_rate_ls.append(0)
            else :
                csv_rate_ls.append(sum(IQS_ls) / len(IQS_ls))
            if len(R2_ls) == 0:
                csv_rate_ls.append(0)
            else :
                csv_rate_ls.append(sum(R2_ls) / len(R2_ls))
        line_csv_ls.append(csv_rate_ls)
    csv_file = res_csv+"/res_line_"+af_source+"_"+downsamlpe_type+"_"+ethnic_show[ethnic_groups.index(ethnic_type)]+"_alltools_" + str(calcul_type) + ".csv"
    with open(csv_file,'w') as f:
        write_line = "af_rate"
        for i in range(len(impute_tools)) :
            write_line = write_line+","+impute_tools[i]+"_curr,"+impute_tools[i]+"_iqs,"+impute_tools[i]+"_r2"
        f.write(write_line+"\n")
        for i in range(len(line_csv_ls)) :
            if calcul_type == 1 :
                if i <= 10 :
                    write_line = str(i+0.5)
                else :
                    write_line = str(11+(i-10)*0.2-0.1)
            else :
                write_line = str(i+0.5)
            for j in range(len(line_csv_ls[i])) :
                write_line = write_line + "," + str(line_csv_ls[i][j])
            write_line = write_line + "\n"
            f.write(write_line)
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_box_res_csv(ethnic_type,res_folder,downsamlpe_type,af_source,impute_tools,tools_names,res_csv) :
    #impute_tools = ['synthetize','beagle4','beagle5','impute5','minimac4']
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_file = ['global','africa','america','centralsouthasia','eastasia','europe','middleeast','oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    #impute_tools = ['impute5']
    af_rate = 0.001
    af_interval_ls = [0,1,5,50,1000]
    af_name = ["AF<0.1","0.1<=AF<0.5","0.5<=AF<5","AF>=5"]
    res_sample_list_ls = []
    for tools in impute_tools :
        res_file = res_folder + "/res."+tools+"."+str(af_rate)[2:]+".txt"
        print("将要读取结果文件：",res_file)
        _,res_sample_list = read_res_file(res_file)
        res_sample_list_ls.append(res_sample_list)
        #print("结果文件读取完毕：",res_file)
    
    box_csv_ls = []
    for i in range(len(af_interval_ls)-1) :
        for j in range(len(impute_tools)) :
            res_sample_list = res_sample_list_ls[j]
            TP_num_ls = [0 for i in range(len(res_sample_list[0]))]
            FP_num_ls = [0 for i in range(len(res_sample_list[0]))]
            FN_num_ls = [0 for i in range(len(res_sample_list[0]))]
            hell_min_ls = [[] for i in range(len(res_sample_list[0]))]
            hell_mean_ls = [[] for i in range(len(res_sample_list[0]))]
            sen_min_ls = [[] for i in range(len(res_sample_list[0]))]
            sen_mean_ls = [[] for i in range(len(res_sample_list[0]))]
            for k in range(len(res_sample_list)) :
                if k < af_interval_ls[i] or k >= af_interval_ls[i+1] :
                    continue
                for h in range(len(res_sample_list[k])) :
                    TP_num_ls[h] += res_sample_list[k][h][0]
                    FP_num_ls[h] += res_sample_list[k][h][1]
                    FN_num_ls[h] += res_sample_list[k][h][3]
                    hell_min_ls[h].append(res_sample_list[k][h][5])
                    hell_mean_ls[h].append(res_sample_list[k][h][6])
                    sen_min_ls[h].append(res_sample_list[k][h][7])
                    sen_mean_ls[h].append(res_sample_list[k][h][8])
            for k in range(len(TP_num_ls)) :
                csv_rate_ls = []
                csv_rate_ls.append(af_name[i])
                csv_rate_ls.append(tools_names[j])
                if TP_num_ls[k] + FP_num_ls[k] == 0 :
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(TP_num_ls[k]/(TP_num_ls[k] + FP_num_ls[k]))
                if TP_num_ls[k] + FN_num_ls[k] == 0 :
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(TP_num_ls[k]/(TP_num_ls[k] + FN_num_ls[k]))
                if csv_rate_ls[-1] + csv_rate_ls[-2] == 0 :
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append((2*csv_rate_ls[-1]*csv_rate_ls[-2])/(csv_rate_ls[-1]+csv_rate_ls[-2]))
                if len(hell_min_ls[k]) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(hell_min_ls[k])/len(hell_min_ls[k]))
                if len(hell_mean_ls[k]) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(hell_mean_ls[k])/len(hell_mean_ls[k]))
                if len(sen_min_ls[k]) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(sen_min_ls[k])/len(sen_min_ls[k]))
                if len(sen_mean_ls[k]) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(sen_mean_ls[k])/len(sen_mean_ls[k]))
                box_csv_ls.append(csv_rate_ls)
    csv_file = res_csv+"/res_box_"+af_source+"_"+downsamlpe_type+"_"+ethnic_show[ethnic_groups.index(ethnic_type)]+"_alltools.csv"
    with open(csv_file,'w') as f:
        write_line = "af_rate,impute_tool,precision,sensitivity,F1,hell_min,hell_mean,sen_min,sen_mean"
        f.write(write_line + "\n")
        for i in range(len(box_csv_ls)) :
            write_line = str(box_csv_ls[i])[1:-1].replace(" ","").replace("'","")
            f.write(write_line + "\n")
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_violin_res_csv(ethnic_type,res_folder,downsamlpe_type,af_source,impute_tools,tools_names,res_csv) :
    #impute_tools = ['synthetize','beagle4','beagle5','impute5','minimac4']
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_file = ['global','africa','america','centralsouthasia','eastasia','europe','middleeast','oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    #impute_tools = ['impute5']
    af_rate = 0.001
    af_interval_ls = [0,1,5,50,1000]
    af_name = ["AF<0.1","0.1<=AF<0.5","0.5<=AF<5","AF>=5"]
    res_sample_list_ls = []
    for tools in impute_tools :
        res_file = res_folder + "/res."+tools+"."+str(af_rate)[2:]+".txt"
        print("将要读取结果文件：",res_file)
        _,res_sample_list = read_res_file(res_file)
        res_sample_list_ls.append(res_sample_list)
        #print("结果文件读取完毕：",res_file)
    
    violin_csv_ls = []
    for i in range(len(impute_tools)) :
        res_sample_list = res_sample_list_ls[i]
        for j in range(len(res_sample_list[0])) :
            TP_num = 0
            FP_num = 0
            FN_num = 0
            for k in range(len(res_sample_list)) :
                TP_num += res_sample_list[k][j][0]
                FP_num += res_sample_list[k][j][1]
                FN_num += res_sample_list[k][j][3]
            if TP_num + FP_num == 0 :
                precision = 0
            else :
                precision = TP_num/(TP_num+FP_num) 
            if TP_num + FN_num == 0 :
                recall = 0
            else :
                recall = TP_num/(TP_num+FN_num) 
            if precision + recall == 0 :
                f1 = 0
            else :
                f1 = (2*precision*recall) / (precision+recall)
            violin_csv_ls.append(tools_names[i]+","+str(TP_num)+","+str(FP_num)+","+str(FN_num)+","+str(f1))
    csv_file = res_csv+"/res_violin_"+af_source+"_"+downsamlpe_type+"_"+ethnic_show[ethnic_groups.index(ethnic_type)]+"_alltools.csv"
    with open(csv_file,'w') as f:
        write_line = "impute_tool,TP_num,FP_num,FN_num,F1"
        f.write(write_line + "\n")
        for i in range(len(violin_csv_ls)) :
            write_line = violin_csv_ls[i]
            f.write(write_line + "\n")
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_all_impute_csv(downsamlpe_type,af_source,res_base,impute_tools,tools_names,is_make_var_num,res_csv) :
    # 该代码的生成结果是基于一组panel和array的，多组生成结果的文件名一致，如ref100和ref90的生成结果名称一致
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    #ethnic_groups = ['gl']
    for ethnic_type in ethnic_groups :
        res_folder = res_base+"/res_"+ethnic_type+"_"+af_source+"_"+downsamlpe_type
        if is_make_var_num :
            write_bar_res_csv(ethnic_type,res_folder,downsamlpe_type,af_source,impute_tools,tools_names,res_csv)
        write_line_res_csv(ethnic_type,res_folder,downsamlpe_type,af_source,1,impute_tools,tools_names,res_csv)
        write_line_res_csv(ethnic_type,res_folder,downsamlpe_type,af_source,2,impute_tools,tools_names,res_csv)
        write_box_res_csv(ethnic_type,res_folder,downsamlpe_type,af_source,impute_tools,tools_names,res_csv)
        write_violin_res_csv(ethnic_type,res_folder,downsamlpe_type,af_source,impute_tools,tools_names,res_csv)

# 以下代码是不同物种使用一个工具
def write_bar_ethnic_csv(res_base,impute_tool,downsamlpe_type,af_source,res_csv) :
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['Global','Africa','America','Central South Asia','East Asia','Europe','Middle East','Oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    # af_rate单位是0.001，af的列表对应为[0,0.001,0.005,0.05,1]
    af_rate = 0.001
    af_interval_ls = [0,1,5,50,1000]
    af_name = ["AF<0.1","0.1<=AF<0.5","0.5<=AF<5","AF>=5"]
    res_pos_list_ls = []
    for e in ethnic_groups :
        res_file = res_base + "/res_"+e+"_"+af_source+"_"+downsamlpe_type+"/res."+impute_tool+"."+str(af_rate)[2:]+".txt"
        print("将要读取结果文件：",res_file)
        res_pos_list,res_sample_list = read_res_file(res_file)
        res_pos_list_ls.append(res_pos_list)
        #print("结果文件读取完毕：",res_file)
    
    bar_csv_ls = []
    for i in range(len(af_interval_ls)-1) :
        for j in range(len(ethnic_groups)) :
            res_pos_list = res_pos_list_ls[j]
            var_num = 0
            for k in range(len(res_pos_list)) :
                if k < af_interval_ls[i] or k >= af_interval_ls[i+1] :
                    continue
                var_num += sum([1 if r2 >= 0.8 else 0 for r2 in res_pos_list[k][11]])

                #var_num += sum([1 if r2 >= 0 else 0 for r2 in res_pos_list[k][11]])
            if i == 0 :
                bar_csv_ls.append(["AF<"+str(af_interval_ls[i+1]*af_rate*100),ethnic_show[j],var_num])
            elif i == len(af_interval_ls)-2 :
                bar_csv_ls.append(["AF>="+str(af_interval_ls[i]*af_rate*100),ethnic_show[j],var_num])
            else :
                bar_csv_ls.append([str(af_interval_ls[i]*af_rate*100)+"<=AF<"+str(af_interval_ls[i+1]*af_rate*100),ethnic_show[j],var_num])
    csv_file = res_csv+"/res_bar_"+af_source+"_"+downsamlpe_type+"_allethnic_"+impute_tool+".csv"
    with open(csv_file,'w') as f:
        write_line = "af_rate,ethnic_group,var_num"
        f.write(write_line+"\n")
        for i in range(len(bar_csv_ls)) :
            write_line = ""
            for j in range(len(bar_csv_ls[i])) :
                write_line = write_line + str(bar_csv_ls[i][j]) + ","
            write_line = write_line[:-1] + "\n"
            f.write(write_line)
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_line_ethnic_csv(res_base,impute_tool,downsamlpe_type,af_source,calcul_type,res_csv) :
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['global','Africa','America','Central South Asia','East Asia','Europe','Middle East','Oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    # af_rate单位是0.001，af的列表对应为[0,0.001,0.005,0.05,1]
    af_rate = 0.001
    #af_interval_ls = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,1]
    if calcul_type == 1 :
        af_interval_ls = [0,1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,1000]
    else :
        af_interval_ls = [0,1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,950,980,990,995,998,999,1000]
    line_csv_ls = []
    res_pos_list_ls = []
    for e in ethnic_groups :
        res_file = res_base + "/res_"+e+"_"+af_source+"_"+downsamlpe_type+"/res."+impute_tool+"."+str(af_rate)[2:]+".txt"
        print("将要读取结果文件：",res_file)
        res_pos_list,res_sample_list = read_res_file(res_file)
        res_pos_list_ls.append(res_pos_list)
        #print("结果文件读取完毕：",res_file)
    for i in range(len(af_interval_ls)-1) :
        csv_rate_ls = []
        for j in range(len(ethnic_groups)) :
            res_pos_list = res_pos_list_ls[j]
            right_num = 0
            total_num = 0
            maf_sum = 0
            maf_num = 0
            IQS_ls = []
            R2_ls = []
            for k in range(len(res_pos_list)) :
                if k < af_interval_ls[i] or k >= af_interval_ls[i+1] :
                    continue
                right_num += sum(res_pos_list[k][3])
                total_num += sum(res_pos_list[k][4])
                for h in range(len(res_pos_list[k][3])) :
                    if res_pos_list[k][4][h] == 0 :
                        maf_sum += 0
                        maf_num += 1
                    else :
                        maf_sum += res_pos_list[k][3][h] / res_pos_list[k][4][h]
                        maf_num += 1
                IQS_ls.extend(res_pos_list[k][10])
                R2_ls.extend(res_pos_list[k][11])
            if maf_num == 0:
                csv_rate_ls.append(0)
            else :
                csv_rate_ls.append(maf_sum / maf_num)
            if len(IQS_ls) == 0:
                csv_rate_ls.append(0)
            else :
                csv_rate_ls.append(sum(IQS_ls) / len(IQS_ls))
            if len(R2_ls) == 0:
                csv_rate_ls.append(0)
            else :
                csv_rate_ls.append(sum(R2_ls) / len(R2_ls))
        line_csv_ls.append(csv_rate_ls)
    csv_file = res_csv+"/res_line_"+af_source+"_"+downsamlpe_type+"_allethnic_"+impute_tool+"_"+str(calcul_type)+".csv"
    with open(csv_file,'w') as f:
        write_line = "af_rate"
        for i in range(len(ethnic_groups)) :
            write_line = write_line+","+ethnic_show[i]+"_curr,"+ethnic_show[i]+"_iqs,"+ethnic_show[i]+"_r2"
        f.write(write_line+"\n")
        for i in range(len(line_csv_ls)) :
            if calcul_type == 1 :
                if i <= 10 :
                    write_line = str(i+0.5)
                else :
                    write_line = str(11+(i-10)*0.2-0.1)
            else :
                write_line = str(i+0.5)
            for j in range(len(line_csv_ls[i])) :
                write_line = write_line + "," + str(line_csv_ls[i][j])
            write_line = write_line + "\n"
            f.write(write_line)
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_box_ethnic_csv(res_base,impute_tool,downsamlpe_type,af_source,res_csv) :
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['global','Africa','America','Central South Asia','East Asia','Europe','Middle East','Oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    #impute_tools = ['impute5']
    af_rate = 0.001
    af_interval_ls = [0,1,5,50,1000]
    af_name = ["AF<0.1","0.1<=AF<0.5","0.5<=AF<5","AF>=5"]
    res_sample_list_ls = []
    for e in ethnic_groups :
        res_file = res_base + "/res_"+e+"_"+af_source+"_"+downsamlpe_type+"/res."+impute_tool+"."+str(af_rate)[2:]+".txt"
        print("将要读取结果文件：",res_file)
        _,res_sample_list = read_res_file(res_file)
        res_sample_list_ls.append(res_sample_list)
        #print("结果文件读取完毕：",res_file)
    
    box_csv_ls = []
    for i in range(len(af_interval_ls)-1) :
        for j in range(len(ethnic_groups)) :
            res_sample_list = res_sample_list_ls[j]
            TP_num_ls = [0 for i in range(len(res_sample_list[0]))]
            FP_num_ls = [0 for i in range(len(res_sample_list[0]))]
            FN_num_ls = [0 for i in range(len(res_sample_list[0]))]
            hell_min_ls = [[] for i in range(len(res_sample_list[0]))]
            hell_mean_ls = [[] for i in range(len(res_sample_list[0]))]
            sen_min_ls = [[] for i in range(len(res_sample_list[0]))]
            sen_mean_ls = [[] for i in range(len(res_sample_list[0]))]
            for k in range(len(res_sample_list)) :
                if k < af_interval_ls[i] or k >= af_interval_ls[i+1] :
                    continue
                for h in range(len(res_sample_list[k])) :
                    TP_num_ls[h] += res_sample_list[k][h][0]
                    FP_num_ls[h] += res_sample_list[k][h][1]
                    FN_num_ls[h] += res_sample_list[k][h][3]
                    hell_min_ls[h].append(res_sample_list[k][h][5])
                    hell_mean_ls[h].append(res_sample_list[k][h][6])
                    sen_min_ls[h].append(res_sample_list[k][h][7])
                    sen_mean_ls[h].append(res_sample_list[k][h][8])
            for k in range(len(TP_num_ls)) :
                csv_rate_ls = []
                csv_rate_ls.append(af_name[i])
                csv_rate_ls.append(ethnic_show[j])
                if TP_num_ls[k] + FP_num_ls[k] == 0 :
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(TP_num_ls[k]/(TP_num_ls[k] + FP_num_ls[k]))
                if TP_num_ls[k] + FN_num_ls[k] == 0 :
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(TP_num_ls[k]/(TP_num_ls[k] + FN_num_ls[k]))
                if csv_rate_ls[-1] + csv_rate_ls[-2] == 0 :
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append((2*csv_rate_ls[-1]*csv_rate_ls[-2])/(csv_rate_ls[-1]+csv_rate_ls[-2]))
                if len(hell_min_ls[k]) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(hell_min_ls[k])/len(hell_min_ls[k]))
                if len(hell_mean_ls[k]) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(hell_mean_ls[k])/len(hell_mean_ls[k]))
                if len(sen_min_ls[k]) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(sen_min_ls[k])/len(sen_min_ls[k]))
                if len(sen_mean_ls[k]) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(sen_mean_ls[k])/len(sen_mean_ls[k]))
                box_csv_ls.append(csv_rate_ls)
    csv_file = res_csv+"/res_box_"+af_source+"_"+downsamlpe_type+"_allethnic_"+impute_tool+".csv"
    with open(csv_file,'w') as f:
        write_line = "af_rate,impute_tool,precision,sensitivity,F1,hell_min,hell_mean,sen_min,sen_mean"
        f.write(write_line + "\n")
        for i in range(len(box_csv_ls)) :
            write_line = str(box_csv_ls[i])[1:-1].replace(" ","").replace("'","")
            f.write(write_line + "\n")
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_violin_ethnic_csv(res_base,impute_tool,downsamlpe_type,af_source,res_csv) :
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['global','Africa','America','Central South Asia','East Asia','Europe','Middle East','Oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    #impute_tools = ['impute5']
    af_rate = 0.001
    af_interval_ls = [0,1,5,50,1000]
    af_name = ["AF<0.1","0.1<=AF<0.5","0.5<=AF<5","AF>=5"]
    res_sample_list_ls = []
    for e in ethnic_groups :
        res_file = res_base + "/res_"+e+"_"+af_source+"_"+downsamlpe_type+"/res."+impute_tool+"."+str(af_rate)[2:]+".txt"
        print("将要读取结果文件：",res_file)
        _,res_sample_list = read_res_file(res_file)
        res_sample_list_ls.append(res_sample_list)
        #print("结果文件读取完毕：",res_file)
    
    violin_csv_ls = []
    for i in range(len(ethnic_groups)) :
        res_sample_list = res_sample_list_ls[i]
        for j in range(len(res_sample_list[0])) :
            TP_num = 0
            FP_num = 0
            FN_num = 0
            for k in range(len(res_sample_list)) :
                TP_num += res_sample_list[k][j][0]
                FP_num += res_sample_list[k][j][1]
                FN_num += res_sample_list[k][j][3]
            if TP_num + FP_num == 0 :
                precision = 0
            else :
                precision = TP_num/(TP_num+FP_num) 
            if TP_num + FN_num == 0 :
                recall = 0
            else :
                recall = TP_num/(TP_num+FN_num) 
            if precision + recall == 0 :
                f1 = 0
            else :
                f1 = (2*precision*recall) / (precision+recall)
            violin_csv_ls.append(ethnic_show[i]+","+str(TP_num)+","+str(FP_num)+","+str(FN_num)+","+str(f1))
    csv_file = res_csv+"/res_violin_"+af_source+"_"+downsamlpe_type+"_allethnic_"+impute_tool+".csv"
    with open(csv_file,'w') as f:
        write_line = "ethnic_group,TP_num,FP_num,FN_num,F1"
        f.write(write_line + "\n")
        for i in range(len(violin_csv_ls)) :
            write_line = violin_csv_ls[i]
            f.write(write_line + "\n")
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_all_ethnic_csv(downsamlpe_type,af_source,res_base,res_csv) :
    # 该代码的生成结果是基于一组panel和array的，多组生成结果的文件名一致，如ref100和ref90的生成结果名称一致
    impute_tool_ls = ['beagle5','minimac4','beagle4','impute5']
    for impute_tool in impute_tool_ls :
        write_bar_ethnic_csv(res_base,impute_tool,downsamlpe_type,af_source,res_csv)
        write_line_ethnic_csv(res_base,impute_tool,downsamlpe_type,af_source,1,res_csv)
        write_line_ethnic_csv(res_base,impute_tool,downsamlpe_type,af_source,2,res_csv)
        write_box_ethnic_csv(res_base,impute_tool,downsamlpe_type,af_source,res_csv)
        write_violin_ethnic_csv(res_base,impute_tool,downsamlpe_type,af_source,res_csv)

# 以下代码是不同的panel size和不同panel下使用同一物种，同一个工具
def write_bar_size_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source,res_csv) :
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['Global','Africa','America','Central South Asia','East Asia','Europe','Middle East','Oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    #size_ls = ['1K','ref100','ref100','ref90','ref80','ref70','ref60','ref50','ref40','ref30','ref20','ref10']
    #size_names = ['1K','China100K','100%','90%','80%','70%','60%','50%','40%','30%','20%','10%']
    size_ls = ['ref100','ref90','ref80','ref70','ref60','ref50','ref40','ref30','ref20','ref10']
    size_names = ['100%','90%','80%','70%','60%','50%','40%','30%','20%','10%']
    # af_rate单位是0.001，af的列表对应为[0,0.001,0.005,0.05,1]
    af_rate = 0.001
    af_interval_ls = [0,1,5,50,1000]
    af_name = ["AF<0.1","0.1<=AF<0.5","0.5<=AF<5","AF>=5"]
    res_pos_list_ls = []
    for ref_size in size_ls :
        res_file = res_base+"/"+ref_size+"/res_floder/res_"+ethnic_type+"_"+af_source+"_"+downsamlpe_type+"/res."+impute_tool+"."+str(af_rate)[2:]+".txt"
        print("将要读取结果文件：",res_file)
        res_pos_list,res_sample_list = read_res_file(res_file)
        res_pos_list_ls.append(res_pos_list)
        #print("结果文件读取完毕：",res_file)
    
    bar_csv_ls = []
    for i in range(len(af_interval_ls)-1) :
        for j in range(len(size_ls)) :
            res_pos_list = res_pos_list_ls[j]
            var_num = 0
            for k in range(len(res_pos_list)) :
                if k < af_interval_ls[i] or k >= af_interval_ls[i+1] :
                    continue
                var_num += sum([1 if r2 >= 0.8 else 0 for r2 in res_pos_list[k][11]])
            if i == 0 :
                bar_csv_ls.append(["AF<"+str(af_interval_ls[i+1]*af_rate*100),size_names[j],var_num])
            elif i == len(af_interval_ls)-2 :
                bar_csv_ls.append(["AF>="+str(af_interval_ls[i]*af_rate*100),size_names[j],var_num])
            else :
                bar_csv_ls.append([str(af_interval_ls[i]*af_rate*100)+"<=AF<"+str(af_interval_ls[i+1]*af_rate*100),size_names[j],var_num])
    csv_file = res_csv+"/res_bar_"+af_source+"_"+downsamlpe_type+"_allsize_"+ethnic_show[ethnic_groups.index(ethnic_type)]+"_"+impute_tool+".csv"
    with open(csv_file,'w') as f:
        write_line = "af_rate,panel_size,var_num"
        f.write(write_line+"\n")
        for i in range(len(bar_csv_ls)) :
            write_line = ""
            for j in range(len(bar_csv_ls[i])) :
                write_line = write_line + str(bar_csv_ls[i][j]) + ","
            write_line = write_line[:-1] + "\n"
            f.write(write_line)
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_line_size_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source,calcul_type,res_csv) :
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['global','Africa','America','Central South Asia','East Asia','Europe','Middle East','Oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    size_ls = ['ref100','ref90','ref80','ref70','ref60','ref50','ref40','ref30','ref20','ref10']
    size_names = ['ref100','ref90','ref80','ref70','ref60','ref50','ref40','ref30','ref20','ref10']
    # af_rate单位是0.001，af的列表对应为[0,0.001,0.005,0.05,1]
    af_rate = 0.001
    #af_interval_ls = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,1]
    if calcul_type == 1 :
        af_interval_ls = [0,1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,1000]
    else :
        af_interval_ls = [0,1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,950,980,990,995,998,999,1000]
    line_csv_ls = []
    res_pos_list_ls = []
    for ref_size in size_ls :
        res_file = res_base+"/"+ref_size+"/res_floder/res_"+ethnic_type+"_"+af_source+"_"+downsamlpe_type+"/res."+impute_tool+"."+str(af_rate)[2:]+".txt"
        print("将要读取结果文件：",res_file)
        res_pos_list,res_sample_list = read_res_file(res_file)
        res_pos_list_ls.append(res_pos_list)
        #print("结果文件读取完毕：",res_file)
    for i in range(len(af_interval_ls)-1) :
        csv_rate_ls = []
        for j in range(len(size_ls)) :
            res_pos_list = res_pos_list_ls[j]
            right_num = 0
            total_num = 0
            maf_sum = 0
            maf_num = 0
            IQS_ls = []
            R2_ls = []
            for k in range(len(res_pos_list)) :
                if k < af_interval_ls[i] or k >= af_interval_ls[i+1] :
                    continue
                right_num += sum(res_pos_list[k][3])
                total_num += sum(res_pos_list[k][4])
                for h in range(len(res_pos_list[k][3])) :
                    if res_pos_list[k][4][h] == 0 :
                        maf_sum += 0
                        maf_num += 1
                    else :
                        maf_sum += res_pos_list[k][3][h] / res_pos_list[k][4][h]
                        maf_num += 1
                IQS_ls.extend(res_pos_list[k][10])
                R2_ls.extend(res_pos_list[k][11])
            if maf_num == 0:
                csv_rate_ls.append(0)
            else :
                csv_rate_ls.append(maf_sum / maf_num)
            if len(IQS_ls) == 0:
                csv_rate_ls.append(0)
            else :
                csv_rate_ls.append(sum(IQS_ls) / len(IQS_ls))
            if len(R2_ls) == 0:
                csv_rate_ls.append(0)
            else :
                csv_rate_ls.append(sum(R2_ls) / len(R2_ls))
        line_csv_ls.append(csv_rate_ls)
    csv_file = res_csv+"/res_line_"+af_source+"_"+downsamlpe_type+"_allsize_"+ethnic_show[ethnic_groups.index(ethnic_type)]+"_"+impute_tool+"_"+str(calcul_type)+".csv"
    with open(csv_file,'w') as f:
        write_line = "af_rate"
        for i in range(len(size_ls)) :
            write_line = write_line+","+size_names[i]+"_curr,"+size_names[i]+"_iqs,"+size_names[i]+"_r2"
        f.write(write_line+"\n")
        for i in range(len(line_csv_ls)) :
            if calcul_type == 1 :
                if i <= 10 :
                    write_line = str(i+0.5)
                else :
                    write_line = str(11+(i-10)*0.2-0.1)
            else :
                write_line = str(i+0.5)
            for j in range(len(line_csv_ls[i])) :
                write_line = write_line + "," + str(line_csv_ls[i][j])
            write_line = write_line + "\n"
            f.write(write_line)
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_box_size_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source,res_csv) :
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['global','Africa','America','Central South Asia','East Asia','Europe','Middle East','Oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    size_ls = ['ref100','ref90','ref80','ref70','ref60','ref50','ref40','ref30','ref20','ref10']
    size_names = ['100%','90%','80%','70%','60%','50%','40%','30%','20%','10%']
    #impute_tools = ['impute5']
    af_rate = 0.001
    af_interval_ls = [0,1,5,50,1000]
    af_name = ["AF<0.1","0.1<=AF<0.5","0.5<=AF<5","AF>=5"]
    res_sample_list_ls = []
    for ref_size in size_ls :
        res_file = res_base+"/"+ref_size+"/res_floder/res_"+ethnic_type+"_"+af_source+"_"+downsamlpe_type+"/res."+impute_tool+"."+str(af_rate)[2:]+".txt"
        print("将要读取结果文件：",res_file)
        res_pos_list,res_sample_list = read_res_file(res_file)
        res_sample_list_ls.append(res_sample_list)
        #print("结果文件读取完毕：",res_file)
    
    box_csv_ls = []
    for i in range(len(af_interval_ls)-1) :
        for j in range(len(size_ls)) :
            res_sample_list = res_sample_list_ls[j]
            TP_num_ls = [0 for i in range(len(res_sample_list[0]))]
            FP_num_ls = [0 for i in range(len(res_sample_list[0]))]
            FN_num_ls = [0 for i in range(len(res_sample_list[0]))]
            hell_min_ls = [[] for i in range(len(res_sample_list[0]))]
            hell_mean_ls = [[] for i in range(len(res_sample_list[0]))]
            sen_min_ls = [[] for i in range(len(res_sample_list[0]))]
            sen_mean_ls = [[] for i in range(len(res_sample_list[0]))]
            for k in range(len(res_sample_list)) :
                if k < af_interval_ls[i] or k >= af_interval_ls[i+1] :
                    continue
                for h in range(len(res_sample_list[k])) :
                    TP_num_ls[h] += res_sample_list[k][h][0]
                    FP_num_ls[h] += res_sample_list[k][h][1]
                    FN_num_ls[h] += res_sample_list[k][h][3]
                    hell_min_ls[h].append(res_sample_list[k][h][5])
                    hell_mean_ls[h].append(res_sample_list[k][h][6])
                    sen_min_ls[h].append(res_sample_list[k][h][7])
                    sen_mean_ls[h].append(res_sample_list[k][h][8])
            for k in range(len(TP_num_ls)) :
                csv_rate_ls = []
                csv_rate_ls.append(af_name[i])
                csv_rate_ls.append(size_names[j])
                if TP_num_ls[k] + FP_num_ls[k] == 0 :
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(TP_num_ls[k]/(TP_num_ls[k] + FP_num_ls[k]))
                if TP_num_ls[k] + FN_num_ls[k] == 0 :
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(TP_num_ls[k]/(TP_num_ls[k] + FN_num_ls[k]))
                if csv_rate_ls[-1] + csv_rate_ls[-2] == 0 :
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append((2*csv_rate_ls[-1]*csv_rate_ls[-2])/(csv_rate_ls[-1]+csv_rate_ls[-2]))
                if len(hell_min_ls[k]) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(hell_min_ls[k])/len(hell_min_ls[k]))
                if len(hell_mean_ls[k]) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(hell_mean_ls[k])/len(hell_mean_ls[k]))
                if len(sen_min_ls[k]) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(sen_min_ls[k])/len(sen_min_ls[k]))
                if len(sen_mean_ls[k]) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(sen_mean_ls[k])/len(sen_mean_ls[k]))
                box_csv_ls.append(csv_rate_ls)
    csv_file = res_csv+"/res_box_"+af_source+"_"+downsamlpe_type+"_allsize_"+ethnic_show[ethnic_groups.index(ethnic_type)]+"_"+impute_tool+".csv"
    with open(csv_file,'w') as f:
        write_line = "af_rate,panel_size,precision,sensitivity,F1,hell_min,hell_mean,sen_min,sen_mean"
        f.write(write_line + "\n")
        for i in range(len(box_csv_ls)) :
            write_line = str(box_csv_ls[i])[1:-1].replace(" ","").replace("'","")
            f.write(write_line + "\n")
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_violin_size_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source,res_csv) :
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['global','Africa','America','Central South Asia','East Asia','Europe','Middle East','Oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    size_ls = ['ref100','ref90','ref80','ref70','ref60','ref50','ref40','ref30','ref20','ref10']
    size_names = ['100%','90%','80%','70%','60%','50%','40%','30%','20%','10%']
    #impute_tools = ['impute5']
    af_rate = 0.001
    af_interval_ls = [0,1,5,50,1000]
    af_name = ["AF<0.1","0.1<=AF<0.5","0.5<=AF<5","AF>=5"]
    res_sample_list_ls = []
    for ref_size in size_ls :
        res_file = res_base+"/"+ref_size+"/res_floder/res_"+ethnic_type+"_"+af_source+"_"+downsamlpe_type+"/res."+impute_tool+"."+str(af_rate)[2:]+".txt"
        print("将要读取结果文件：",res_file)
        res_pos_list,res_sample_list = read_res_file(res_file)
        res_sample_list_ls.append(res_sample_list)
        #print("结果文件读取完毕：",res_file)
    
    violin_csv_ls = []
    for i in range(len(size_ls)) :
        res_sample_list = res_sample_list_ls[i]
        for j in range(len(res_sample_list[0])) :
            TP_num = 0
            FP_num = 0
            FN_num = 0
            for k in range(len(res_sample_list)) :
                TP_num += res_sample_list[k][j][0]
                FP_num += res_sample_list[k][j][1]
                FN_num += res_sample_list[k][j][3]
            if TP_num + FP_num == 0 :
                precision = 0
            else :
                precision = TP_num/(TP_num+FP_num) 
            if TP_num + FN_num == 0 :
                recall = 0
            else :
                recall = TP_num/(TP_num+FN_num) 
            if precision + recall == 0 :
                f1 = 0
            else :
                f1 = (2*precision*recall) / (precision+recall)
            violin_csv_ls.append(size_names[i]+","+str(TP_num)+","+str(FP_num)+","+str(FN_num)+","+str(f1))
    csv_file = res_csv+"/res_violin_"+af_source+"_"+downsamlpe_type+"_allsize_"+ethnic_show[ethnic_groups.index(ethnic_type)]+"_"+impute_tool+".csv"
    with open(csv_file,'w') as f:
        write_line = "panel_size,TP_num,FP_num,FN_num,F1"
        f.write(write_line + "\n")
        for i in range(len(violin_csv_ls)) :
            write_line = violin_csv_ls[i]
            f.write(write_line + "\n")
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_all_size_csv(downsamlpe_type,af_source,res_base,res_csv) :
    # 该代码的生成结果是基于多组panel和array的，多组生成结果的文件名统一
    impute_tool_ls = ['beagle5','minimac4','beagle4','impute5']
    #impute_tool_ls = ['minimac4']
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    #ethnic_groups = ['gl']
    for impute_tool in impute_tool_ls :
        for ethnic_type in ethnic_groups :
            write_bar_size_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source,res_csv)
            write_line_size_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source,1,res_csv)
            write_line_size_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source,2,res_csv)
            write_box_size_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source,res_csv)
            write_violin_size_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source,res_csv)

# 以下代码是不同的panel下使用同一物种，同一个工具
def write_bar_panel_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source) :
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['Global','Africa','America','Central South Asia','East Asia','Europe','Middle East','Oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    panel_ls = ['1K','ref100','chinamap','ckb','topmed','synthetize']
    panel_names = ['Panel1K','China100K','ChinaMap','CKB','TopMed','synthetize']
    # af_rate单位是0.001，af的列表对应为[0,0.001,0.005,0.05,1]
    af_rate = 0.001
    af_interval_ls = [0,1,5,50,1000]
    af_name = ["AF<0.1","0.1<=AF<0.5","0.5<=AF<5","AF>=5"]
    res_pos_list_ls = []
    for panel_file in panel_ls :
        res_file = res_base+"/"+panel_file+"/res_floder/res_"+ethnic_type+"_"+af_source+"_"+downsamlpe_type+"/res."+impute_tool+"."+str(af_rate)[2:]+".txt"
        print("将要读取结果文件：",res_file)
        res_pos_list,res_sample_list = read_res_file(res_file)
        res_pos_list_ls.append(res_pos_list)
        #print("结果文件读取完毕：",res_file)
    
    bar_csv_ls = []
    for i in range(len(af_interval_ls)-1) :
        for j in range(len(panel_ls)) :
            res_pos_list = res_pos_list_ls[j]
            var_num = 0
            for k in range(len(res_pos_list)) :
                if k < af_interval_ls[i] or k >= af_interval_ls[i+1] :
                    continue
                var_num += sum([1 if r2 >= 0.8 else 0 for r2 in res_pos_list[k][11]])
            if i == 0 :
                bar_csv_ls.append(["AF<"+str(af_interval_ls[i+1]*af_rate*100),panel_names[j],var_num])
            elif i == len(af_interval_ls)-2 :
                bar_csv_ls.append(["AF>="+str(af_interval_ls[i]*af_rate*100),panel_names[j],var_num])
            else :
                bar_csv_ls.append([str(af_interval_ls[i]*af_rate*100)+"<=AF<"+str(af_interval_ls[i+1]*af_rate*100),panel_names[j],var_num])
    csv_file = "res_csv/res_bar_"+af_source+"_"+downsamlpe_type+"_allpanel_"+ethnic_show[ethnic_groups.index(ethnic_type)]+"_"+impute_tool+".csv"
    with open(csv_file,'w') as f:
        write_line = "af_rate,panel_size,var_num"
        f.write(write_line+"\n")
        for i in range(len(bar_csv_ls)) :
            write_line = ""
            for j in range(len(bar_csv_ls[i])) :
                write_line = write_line + str(bar_csv_ls[i][j]) + ","
            write_line = write_line[:-1] + "\n"
            f.write(write_line)
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_line_panel_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source,calcul_type) :
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['global','Africa','America','Central South Asia','East Asia','Europe','Middle East','Oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    panel_ls = ['1K','ref100','chinamap','ckb','topmed','synthetize']
    panel_names = ['Panel1K','China100K','ChinaMap','CKB','TopMed','synthetize']
    # af_rate单位是0.001，af的列表对应为[0,0.001,0.005,0.05,1]
    af_rate = 0.001
    #af_interval_ls = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,1]
    if calcul_type == 1 :
        af_interval_ls = [0,1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,1000]
    else :
        af_interval_ls = [0,1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,950,980,990,995,998,999,1000]
    line_csv_ls = []
    res_pos_list_ls = []
    for ref_size in panel_ls :
        res_file = res_base+"/"+ref_size+"/res_floder/res_"+ethnic_type+"_"+af_source+"_"+downsamlpe_type+"/res."+impute_tool+"."+str(af_rate)[2:]+".txt"
        print("将要读取结果文件：",res_file)
        res_pos_list,res_sample_list = read_res_file(res_file)
        res_pos_list_ls.append(res_pos_list)
        #print("结果文件读取完毕：",res_file)
    for i in range(len(af_interval_ls)-1) :
        csv_rate_ls = []
        for j in range(len(panel_ls)) :
            res_pos_list = res_pos_list_ls[j]
            right_num = 0
            total_num = 0
            maf_sum = 0
            maf_num = 0
            IQS_ls = []
            R2_ls = []
            for k in range(len(res_pos_list)) :
                if k < af_interval_ls[i] or k >= af_interval_ls[i+1] :
                    continue
                right_num += sum(res_pos_list[k][3])
                total_num += sum(res_pos_list[k][4])
                for h in range(len(res_pos_list[k][3])) :
                    if res_pos_list[k][4][h] == 0 :
                        maf_sum += 0
                        maf_num += 1
                    else :
                        maf_sum += res_pos_list[k][3][h] / res_pos_list[k][4][h]
                        maf_num += 1
                IQS_ls.extend(res_pos_list[k][10])
                R2_ls.extend(res_pos_list[k][11])
            if maf_num == 0:
                csv_rate_ls.append(0)
            else :
                csv_rate_ls.append(maf_sum / maf_num)
            if len(IQS_ls) == 0:
                csv_rate_ls.append(0)
            else :
                csv_rate_ls.append(sum(IQS_ls) / len(IQS_ls))
            if len(R2_ls) == 0:
                csv_rate_ls.append(0)
            else :
                csv_rate_ls.append(sum(R2_ls) / len(R2_ls))
        line_csv_ls.append(csv_rate_ls)
    csv_file = "res_csv/res_line_"+af_source+"_"+downsamlpe_type+"_allpanel_"+ethnic_show[ethnic_groups.index(ethnic_type)]+"_"+impute_tool+"_"+str(calcul_type)+".csv"
    with open(csv_file,'w') as f:
        write_line = "af_rate"
        for i in range(len(panel_ls)) :
            write_line = write_line+","+panel_names[i]+"_curr,"+panel_names[i]+"_iqs,"+panel_names[i]+"_r2"
        f.write(write_line+"\n")
        for i in range(len(line_csv_ls)) :
            if calcul_type == 1 :
                if i <= 10 :
                    write_line = str(i+0.5)
                else :
                    write_line = str(11+(i-10)*0.2-0.1)
            else :
                write_line = str(i+0.5)
            for j in range(len(line_csv_ls[i])) :
                write_line = write_line + "," + str(line_csv_ls[i][j])
            write_line = write_line + "\n"
            f.write(write_line)
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_box_panel_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source) :
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['global','Africa','America','Central South Asia','East Asia','Europe','Middle East','Oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    panel_ls = ['1K','ref100','chinamap','ckb','topmed','synthetize']
    panel_names = ['Panel1K','China100K','ChinaMap','CKB','TopMed','synthetize']
    #impute_tools = ['impute5']
    af_rate = 0.001
    af_interval_ls = [0,1,5,50,1000]
    af_name = ["AF<0.1","0.1<=AF<0.5","0.5<=AF<5","AF>=5"]
    res_sample_list_ls = []
    for ref_size in panel_ls :
        res_file = res_base+"/"+ref_size+"/res_floder/res_"+ethnic_type+"_"+af_source+"_"+downsamlpe_type+"/res."+impute_tool+"."+str(af_rate)[2:]+".txt"
        print("将要读取结果文件：",res_file)
        res_pos_list,res_sample_list = read_res_file(res_file)
        res_sample_list_ls.append(res_sample_list)
        #print("结果文件读取完毕：",res_file)
    
    box_csv_ls = []
    for i in range(len(af_interval_ls)-1) :
        for j in range(len(panel_ls)) :
            res_sample_list = res_sample_list_ls[j]
            TP_num_ls = [0 for i in range(len(res_sample_list[0]))]
            FP_num_ls = [0 for i in range(len(res_sample_list[0]))]
            FN_num_ls = [0 for i in range(len(res_sample_list[0]))]
            hell_min_ls = [[] for i in range(len(res_sample_list[0]))]
            hell_mean_ls = [[] for i in range(len(res_sample_list[0]))]
            sen_min_ls = [[] for i in range(len(res_sample_list[0]))]
            sen_mean_ls = [[] for i in range(len(res_sample_list[0]))]
            for k in range(len(res_sample_list)) :
                if k < af_interval_ls[i] or k >= af_interval_ls[i+1] :
                    continue
                for h in range(len(res_sample_list[k])) :
                    TP_num_ls[h] += res_sample_list[k][h][0]
                    FP_num_ls[h] += res_sample_list[k][h][1]
                    FN_num_ls[h] += res_sample_list[k][h][3]
                    hell_min_ls[h].append(res_sample_list[k][h][5])
                    hell_mean_ls[h].append(res_sample_list[k][h][6])
                    sen_min_ls[h].append(res_sample_list[k][h][7])
                    sen_mean_ls[h].append(res_sample_list[k][h][8])
            for k in range(len(TP_num_ls)) :
                csv_rate_ls = []
                csv_rate_ls.append(af_name[i])
                csv_rate_ls.append(panel_names[j])
                if TP_num_ls[k] + FP_num_ls[k] == 0 :
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(TP_num_ls[k]/(TP_num_ls[k] + FP_num_ls[k]))
                if TP_num_ls[k] + FN_num_ls[k] == 0 :
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(TP_num_ls[k]/(TP_num_ls[k] + FN_num_ls[k]))
                if csv_rate_ls[-1] + csv_rate_ls[-2] == 0 :
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append((2*csv_rate_ls[-1]*csv_rate_ls[-2])/(csv_rate_ls[-1]+csv_rate_ls[-2]))
                if len(hell_min_ls[k]) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(hell_min_ls[k])/len(hell_min_ls[k]))
                if len(hell_mean_ls[k]) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(hell_mean_ls[k])/len(hell_mean_ls[k]))
                if len(sen_min_ls[k]) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(sen_min_ls[k])/len(sen_min_ls[k]))
                if len(sen_mean_ls[k]) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(sen_mean_ls[k])/len(sen_mean_ls[k]))
                box_csv_ls.append(csv_rate_ls)
    csv_file = "res_csv/res_box_"+af_source+"_"+downsamlpe_type+"_allpanel_"+ethnic_show[ethnic_groups.index(ethnic_type)]+"_"+impute_tool+".csv"
    with open(csv_file,'w') as f:
        write_line = "af_rate,panel_size,precision,sensitivity,F1,hell_min,hell_mean,sen_min,sen_mean"
        f.write(write_line + "\n")
        for i in range(len(box_csv_ls)) :
            write_line = str(box_csv_ls[i])[1:-1].replace(" ","").replace("'","")
            f.write(write_line + "\n")
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_violin_panel_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source) :
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['global','Africa','America','Central South Asia','East Asia','Europe','Middle East','Oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    panel_ls = ['1K','ref100','chinamap','ckb','topmed','synthetize']
    panel_names = ['Panel1K','China100K','ChinaMap','CKB','TopMed','synthetize']
    #impute_tools = ['impute5']
    af_rate = 0.001
    af_interval_ls = [0,1,5,50,1000]
    af_name = ["AF<0.1","0.1<=AF<0.5","0.5<=AF<5","AF>=5"]
    res_sample_list_ls = []
    for ref_size in panel_ls :
        res_file = res_base+"/"+ref_size+"/res_floder/res_"+ethnic_type+"_"+af_source+"_"+downsamlpe_type+"/res."+impute_tool+"."+str(af_rate)[2:]+".txt"
        print("将要读取结果文件：",res_file)
        res_pos_list,res_sample_list = read_res_file(res_file)
        res_sample_list_ls.append(res_sample_list)
        #print("结果文件读取完毕：",res_file)
    
    violin_csv_ls = []
    for i in range(len(panel_ls)) :
        res_sample_list = res_sample_list_ls[i]
        for j in range(len(res_sample_list[0])) :
            TP_num = 0
            FP_num = 0
            FN_num = 0
            for k in range(len(res_sample_list)) :
                TP_num += res_sample_list[k][j][0]
                FP_num += res_sample_list[k][j][1]
                FN_num += res_sample_list[k][j][3]
            if TP_num + FP_num == 0 :
                precision = 0
            else :
                precision = TP_num/(TP_num+FP_num) 
            if TP_num + FN_num == 0 :
                recall = 0
            else :
                recall = TP_num/(TP_num+FN_num) 
            if precision + recall == 0 :
                f1 = 0
            else :
                f1 = (2*precision*recall) / (precision+recall)
            violin_csv_ls.append(panel_names[i]+","+str(TP_num)+","+str(FP_num)+","+str(FN_num)+","+str(f1))
    csv_file = "res_csv/res_violin_"+af_source+"_"+downsamlpe_type+"_allpanel_"+ethnic_show[ethnic_groups.index(ethnic_type)]+"_"+impute_tool+".csv"
    with open(csv_file,'w') as f:
        write_line = "panel_size,TP_num,FP_num,FN_num,F1"
        f.write(write_line + "\n")
        for i in range(len(violin_csv_ls)) :
            write_line = violin_csv_ls[i]
            f.write(write_line + "\n")
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_all_panel_csv(downsamlpe_type,af_source,res_base) :
    # 该代码的生成结果是基于多组panel和array的，多组生成结果的文件名统一
    #impute_tool_ls = ['beagle5','minimac4','beagle4','impute5']
    impute_tool_ls = ['minimac4']
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    #ethnic_groups = ['gl']
    for impute_tool in impute_tool_ls :
        for ethnic_type in ethnic_groups :
            write_bar_panel_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source)
            write_line_panel_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source,1)
            write_line_panel_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source,2)
            write_box_panel_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source)
            write_violin_panel_csv(res_base,ethnic_type,impute_tool,downsamlpe_type,af_source)

# 以下代码是使用同一个panel，但是使用的array是否是只包含对应样本的
def write_bar_downsample_csv(res_base,impute_tool,af_source,res_csv) :
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['Global','Africa','America','Central South Asia','East Asia','Europe','Middle East','Oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    # af_rate单位是0.001，af的列表对应为[0,0.001,0.005,0.05,1]
    af_rate = 0.001
    af_interval_ls = [0,1,5,50,1000]
    af_name = ["AF<0.1","0.1<=AF<0.5","0.5<=AF<5","AF>=5"]
    downsamlpe_type_ls = ["all","part"]
    res_pos_list_ls = []
    for i in downsamlpe_type_ls :
        res_pos_list_one_downsample = []
        for e in ethnic_groups :
            res_file = res_base + "/res_"+e+"_"+af_source+"_"+i+"/res."+impute_tool+"."+str(af_rate)[2:]+".txt"
            print("将要读取结果文件：",res_file)
            res_pos_list,res_sample_list = read_res_file(res_file)
            res_pos_list_one_downsample.append(res_pos_list)
        res_pos_list_ls.append(res_pos_list_one_downsample)
    #print("结果文件读取完毕：",res_file)
    
    bar_csv_ls = []
    for i in range(len(af_interval_ls)-1) :
        for j in range(len(ethnic_groups)) :
            for h in range(len(downsamlpe_type_ls)) :
                res_pos_list = res_pos_list_ls[h][j]
                var_num = 0
                for k in range(len(res_pos_list)) :
                    if k < af_interval_ls[i] or k >= af_interval_ls[i+1] :
                        continue
                    var_num += sum([1 if r2 >= 0.8 else 0 for r2 in res_pos_list[k][11]])
            if i == 0 :
                bar_csv_ls.append(["AF<"+str(af_interval_ls[i+1]*af_rate*100),ethnic_show[j]+"_"+downsamlpe_type_ls[h],var_num])
            elif i == len(af_interval_ls)-2 :
                bar_csv_ls.append(["AF>="+str(af_interval_ls[i]*af_rate*100),ethnic_show[j]+"_"+downsamlpe_type_ls[h],var_num])
            else :
                bar_csv_ls.append([str(af_interval_ls[i]*af_rate*100)+"<=AF<"+str(af_interval_ls[i+1]*af_rate*100),ethnic_show[j]+"_"+downsamlpe_type_ls[h],var_num])
    csv_file = res_csv+"/res_bar_"+af_source+"_alldownsample_allethnic_"+impute_tool+".csv"
    with open(csv_file,'w') as f:
        write_line = "af_rate,ethnic_downsamlpe,var_num"
        f.write(write_line+"\n")
        for i in range(len(bar_csv_ls)) :
            write_line = ""
            for j in range(len(bar_csv_ls[i])) :
                write_line = write_line + str(bar_csv_ls[i][j]) + ","
            write_line = write_line[:-1] + "\n"
            f.write(write_line)
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_line_downsample_csv(res_base,impute_tool,af_source,calcul_type,res_csv) :
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['global','Africa','America','Central South Asia','East Asia','Europe','Middle East','Oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    # af_rate单位是0.001，af的列表对应为[0,0.001,0.005,0.05,1]
    af_rate = 0.001
    #af_interval_ls = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,1]
    if calcul_type == 1 :
        af_interval_ls = [0,1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,1000]
    else :
        af_interval_ls = [0,1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,950,980,990,995,998,999,1000]
    line_csv_ls = []
    downsamlpe_type_ls = ["all","part"]
    res_pos_list_ls = []
    for i in downsamlpe_type_ls :
        res_pos_list_one_downsample = []
        for e in ethnic_groups :
            res_file = res_base + "/res_"+e+"_"+af_source+"_"+i+"/res."+impute_tool+"."+str(af_rate)[2:]+".txt"
            print("将要读取结果文件：",res_file)
            res_pos_list,res_sample_list = read_res_file(res_file)
            res_pos_list_one_downsample.append(res_pos_list)
        res_pos_list_ls.append(res_pos_list_one_downsample)
            #print("结果文件读取完毕：",res_file)
    for i in range(len(af_interval_ls)-1) :
        csv_rate_ls = []
        for j in range(len(ethnic_groups)) :            
            for m in range(len(downsamlpe_type_ls)) :
                res_pos_list = res_pos_list_ls[m][j]
                right_num = 0
                total_num = 0
                maf_sum = 0
                maf_num = 0
                IQS_ls = []
                R2_ls = []
                for k in range(len(res_pos_list)) :
                    if k < af_interval_ls[i] or k >= af_interval_ls[i+1] :
                        continue
                    right_num += sum(res_pos_list[k][3])
                    total_num += sum(res_pos_list[k][4])
                    for h in range(len(res_pos_list[k][3])) :
                        if res_pos_list[k][4][h] == 0 :
                            maf_sum += 0
                            maf_num += 1
                        else :
                            maf_sum += res_pos_list[k][3][h] / res_pos_list[k][4][h]
                            maf_num += 1
                    IQS_ls.extend(res_pos_list[k][10])
                    R2_ls.extend(res_pos_list[k][11])
                if maf_num == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(maf_sum / maf_num)
                if len(IQS_ls) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(IQS_ls) / len(IQS_ls))
                if len(R2_ls) == 0:
                    csv_rate_ls.append(0)
                else :
                    csv_rate_ls.append(sum(R2_ls) / len(R2_ls))
        line_csv_ls.append(csv_rate_ls)
    csv_file = res_csv+"/res_line_"+af_source+"_alldownsample_allethnic_"+impute_tool+"_"+str(calcul_type)+".csv"
    with open(csv_file,'w') as f:
        write_line = "af_rate"
        for i in range(len(ethnic_groups)) :
            for j in range(len(downsamlpe_type_ls)) :
                write_line = write_line+","+ethnic_show[i]+"_"+downsamlpe_type_ls[j]+"_curr,"+ethnic_show[i]+"_"+downsamlpe_type_ls[j]+"_iqs,"+ethnic_show[i]+"_"+downsamlpe_type_ls[j]+"_r2"
        f.write(write_line+"\n")
        for i in range(len(line_csv_ls)) :
            if calcul_type == 1 :
                if i <= 10 :
                    write_line = str(i+0.5)
                else :
                    write_line = str(11+(i-10)*0.2-0.1)
            else :
                write_line = str(i+0.5)
            for j in range(len(line_csv_ls[i])) :
                write_line = write_line + "," + str(line_csv_ls[i][j])
            write_line = write_line + "\n"
            f.write(write_line)
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_box_downsample_csv(res_base,impute_tool,af_source,res_csv) :
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['global','Africa','America','Central South Asia','East Asia','Europe','Middle East','Oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    #impute_tools = ['impute5']
    af_rate = 0.001
    af_interval_ls = [0,1,5,50,1000]
    af_name = ["AF<0.1","0.1<=AF<0.5","0.5<=AF<5","AF>=5"]
    downsamlpe_type_ls = ["all","part"]
    res_sample_list_ls = []
    for i in downsamlpe_type_ls :
        res_pos_list_one_downsample = []
        for e in ethnic_groups :
            res_file = res_base + "/res_"+e+"_"+af_source+"_"+i+"/res."+impute_tool+"."+str(af_rate)[2:]+".txt"
            print("将要读取结果文件：",res_file)
            _,res_sample_list = read_res_file(res_file)
            res_pos_list_one_downsample.append(res_sample_list)
        res_sample_list_ls.append(res_pos_list_one_downsample)
    
    box_csv_ls = []
    for i in range(len(af_interval_ls)-1) :
        for j in range(len(ethnic_groups)) :
            for m in range(len(downsamlpe_type_ls)) :
                res_sample_list = res_sample_list_ls[m][j]
                TP_num_ls = [0 for i in range(len(res_sample_list[0]))]
                FP_num_ls = [0 for i in range(len(res_sample_list[0]))]
                FN_num_ls = [0 for i in range(len(res_sample_list[0]))]
                hell_min_ls = [[] for i in range(len(res_sample_list[0]))]
                hell_mean_ls = [[] for i in range(len(res_sample_list[0]))]
                sen_min_ls = [[] for i in range(len(res_sample_list[0]))]
                sen_mean_ls = [[] for i in range(len(res_sample_list[0]))]
                for k in range(len(res_sample_list)) :
                    if k < af_interval_ls[i] or k >= af_interval_ls[i+1] :
                        continue
                    for h in range(len(res_sample_list[k])) :
                        TP_num_ls[h] += res_sample_list[k][h][0]
                        FP_num_ls[h] += res_sample_list[k][h][1]
                        FN_num_ls[h] += res_sample_list[k][h][3]
                        hell_min_ls[h].append(res_sample_list[k][h][5])
                        hell_mean_ls[h].append(res_sample_list[k][h][6])
                        sen_min_ls[h].append(res_sample_list[k][h][7])
                        sen_mean_ls[h].append(res_sample_list[k][h][8])
                for k in range(len(TP_num_ls)) :
                    csv_rate_ls = []
                    csv_rate_ls.append(af_name[i])
                    csv_rate_ls.append(ethnic_show[j]+"_"+downsamlpe_type_ls[m])
                    if TP_num_ls[k] + FP_num_ls[k] == 0 :
                        csv_rate_ls.append(0)
                    else :
                        csv_rate_ls.append(TP_num_ls[k]/(TP_num_ls[k] + FP_num_ls[k]))
                    if TP_num_ls[k] + FN_num_ls[k] == 0 :
                        csv_rate_ls.append(0)
                    else :
                        csv_rate_ls.append(TP_num_ls[k]/(TP_num_ls[k] + FN_num_ls[k]))
                    if csv_rate_ls[-1] + csv_rate_ls[-2] == 0 :
                        csv_rate_ls.append(0)
                    else :
                        csv_rate_ls.append((2*csv_rate_ls[-1]*csv_rate_ls[-2])/(csv_rate_ls[-1]+csv_rate_ls[-2]))
                    if len(hell_min_ls[k]) == 0:
                        csv_rate_ls.append(0)
                    else :
                        csv_rate_ls.append(sum(hell_min_ls[k])/len(hell_min_ls[k]))
                    if len(hell_mean_ls[k]) == 0:
                        csv_rate_ls.append(0)
                    else :
                        csv_rate_ls.append(sum(hell_mean_ls[k])/len(hell_mean_ls[k]))
                    if len(sen_min_ls[k]) == 0:
                        csv_rate_ls.append(0)
                    else :
                        csv_rate_ls.append(sum(sen_min_ls[k])/len(sen_min_ls[k]))
                    if len(sen_mean_ls[k]) == 0:
                        csv_rate_ls.append(0)
                    else :
                        csv_rate_ls.append(sum(sen_mean_ls[k])/len(sen_mean_ls[k]))
                    box_csv_ls.append(csv_rate_ls)
    csv_file = res_csv+"/res_box_"+af_source+"_alldownsample_allethnic_"+impute_tool+".csv"
    with open(csv_file,'w') as f:
        write_line = "af_rate,ethnic_downsamlpe,precision,sensitivity,F1,hell_min,hell_mean,sen_min,sen_mean"
        f.write(write_line + "\n")
        for i in range(len(box_csv_ls)) :
            write_line = str(box_csv_ls[i])[1:-1].replace(" ","").replace("'","")
            f.write(write_line + "\n")
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_violin_downsample_csv(res_base,impute_tool,af_source,res_csv) :
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['global','Africa','America','Central South Asia','East Asia','Europe','Middle East','Oceania']
    ethnic_show = ['GLB','AFR','AMR','CSA','EAS','EUR','ME','OCE']
    #impute_tools = ['impute5']
    af_rate = 0.001
    af_interval_ls = [0,1,5,50,1000]
    af_name = ["AF<0.1","0.1<=AF<0.5","0.5<=AF<5","AF>=5"]
    downsamlpe_type_ls = ["all","part"]
    res_sample_list_ls = []
    for i in downsamlpe_type_ls :
        res_pos_list_one_downsample = []
        for e in ethnic_groups :
            res_file = res_base + "/res_"+e+"_"+af_source+"_"+i+"/res."+impute_tool+"."+str(af_rate)[2:]+".txt"
            print("将要读取结果文件：",res_file)
            _,res_sample_list = read_res_file(res_file)
            res_pos_list_one_downsample.append(res_sample_list)
        res_sample_list_ls.append(res_pos_list_one_downsample)
    violin_csv_ls = []
    for i in range(len(ethnic_groups)) :
        for h in range(len(downsamlpe_type_ls)) :
            res_sample_list = res_sample_list_ls[h][i]
            for j in range(len(res_sample_list[0])) :
                TP_num = 0
                FP_num = 0
                FN_num = 0
                for k in range(len(res_sample_list)) :
                    TP_num += res_sample_list[k][j][0]
                    FP_num += res_sample_list[k][j][1]
                    FN_num += res_sample_list[k][j][3]
                if TP_num + FP_num == 0 :
                    precision = 0
                else :
                    precision = TP_num/(TP_num+FP_num) 
                if TP_num + FN_num == 0 :
                    recall = 0
                else :
                    recall = TP_num/(TP_num+FN_num) 
                if precision + recall == 0 :
                    f1 = 0
                else :
                    f1 = (2*precision*recall) / (precision+recall)
                violin_csv_ls.append(ethnic_show[i]+"_"+downsamlpe_type_ls[h]+","+str(TP_num)+","+str(FP_num)+","+str(FN_num)+","+str(f1))
    csv_file = res_csv+"/res_violin_"+af_source+"_alldownsample_allethnic_"+impute_tool+".csv"
    with open(csv_file,'w') as f:
        write_line = "ethnic_downsamlpe,TP_num,FP_num,FN_num,F1"
        f.write(write_line + "\n")
        for i in range(len(violin_csv_ls)) :
            write_line = violin_csv_ls[i]
            f.write(write_line + "\n")
        f.close()
    print("绘图文件写入完毕：",csv_file)

def write_all_downsample_csv(af_source,impute_tool,res_base,res_csv) :
    write_bar_downsample_csv(res_base,impute_tool,af_source,res_csv)
    write_line_downsample_csv(res_base,impute_tool,af_source,1,res_csv)
    write_line_downsample_csv(res_base,impute_tool,af_source,2,res_csv)
    write_box_downsample_csv(res_base,impute_tool,af_source,res_csv)
    write_violin_downsample_csv(res_base,impute_tool,af_source,res_csv)

# 形成绘制全部的r语言图像的数据文件
def write_all_draw_csv(tools_ls) :
    downsamlpe_type,af_source,res_base = "all","panelaf","XXX/ref100/res_floder"
    write_all_impute_csv(downsamlpe_type,af_source,res_base,tools_ls)