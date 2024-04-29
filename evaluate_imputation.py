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

# 统计imputation输出相对于答案的各种指标值
def get_impute_accuracy_common(vcf_ans_ls,vcf_out_ls,af_file,af_source,af_interval,af_ceiling,af_floor,sample_filter_ls,res_file,impute_tool) :
    #print("答案文件中数据量为：",len(vcf_ans_ls))
    #print("输出文件中数据量为：",len(vcf_out_ls))
    af_max_index = int((af_ceiling-af_floor)/af_interval)
    samples_num = len(vcf_ans_ls[0]) - 9

    # res_pos_list为三维，第一维为af rate分类，第二维为各种信息分类，第三维为pos的信息，包含信息为:
    # 7/8/9没用，可以做备用空位
    # 0-pos,1-all_concordance_right_num,2-all_concordance_total_num,3-maf_concordance_right_num,4-maf_concordance_total_num,5-ref,6-alt,7-well_right_all_num,8-average_SEN_score,9-min_SEN_score,10-IQS,11-R2,12-af_rate
    # res_sample_list为三维，第一维为af rate分类，第二维为sample分类，第三维为各种信息分类
    # 包含信息为:0-TP Number,1-FP Number,2-TN Number,3-FN Numer,4-total number,5-min hell,6-mean-hell,7-min sen,8-mean sen
    res_pos_list = [[[] for j in range(13)] for i in range(af_max_index)]
    res_sample_list = [[[0 for k in range(9)] for j in range(samples_num)] for i in range(af_max_index)]
    sample_score_list = [[[[],[]] for j in range(samples_num)] for i in range(af_max_index)]
    out_num_ls = [0 for i in range(af_max_index)]
    ans_num_ls = [0 for i in range(af_max_index)]
    af_panel = []
    r2_ls = []
    r2_ori_ls = []
    fre_ls = []
    fre_ori_ls = []
    with open(af_file,'r') as f:
        while(True) :
            f_line = f.readline().strip()
            if f_line == "" :
                break
            f_line_ls = f_line.split('\t')
            f_line_ls[3] = float(f_line_ls[3])
            af_panel.append(f_line_ls)
    af_list = []
    i = j = 0
    vcf_ans_filter = []
    vcf_out_filter = []
    af_list = []
    while(i < len(vcf_out_ls) and j < len(af_panel)) :
        if int(vcf_out_ls[i][1]) < int(af_panel[j][0]) :
            i += 1
            continue
        elif int(vcf_out_ls[i][1]) > int(af_panel[j][0]) :
            j += 1
            continue
        elif vcf_out_ls[i][3] < af_panel[j][1] :
            i += 1
            continue
        elif vcf_out_ls[i][3] > af_panel[j][1] :
            j += 1
            continue
        elif vcf_out_ls[i][4] < af_panel[j][2] :
            i += 1
            continue
        elif vcf_out_ls[i][4] > af_panel[j][2] :
            j += 1
            continue
        vcf_ans_filter.append(vcf_ans_ls[i])
        vcf_out_filter.append(vcf_out_ls[i])
        af_list.append(af_panel[j])
        i += 1
        j += 1
    vcf_ans_ls = vcf_ans_filter
    vcf_out_ls = vcf_out_filter
    print("输出数量：",len(vcf_out_ls),"\t答案数量：",len(vcf_ans_ls),"\t频率数量：",len(af_list))
    #print("af文件中数据量为：",len(af_list))
    #print("样本总数量为：",samples_num)
    for i in range(len(vcf_out_ls)) :
        af_str = vcf_out_ls[i][7][vcf_out_ls[i][7].find("AF=")+3:]
        af_str = af_str[:af_str.index(";") if af_str.index(";")!=-1 else len(af_str)]
        frequency_value = float(af_str)
        if af_source == "panelaf" :
            af_rate = af_list[i][3]
        elif af_source == "imputeaf" :
            af_rate = frequency_value
        af_index = math.floor((af_rate - af_floor)/af_interval)
        if af_index == af_max_index :
            af_index = af_max_index - 1
        # 以下是评价指标，如果修改了评价方式，只需要修改以下代码即可
        right_maf_num = 0
        total_maf_num = 0
        right_all_num = 0
        well_right_all_num = 0
        IQS_ls = [0 for x in range(9)]
        Hell_ls = []
        Sen_ls = []
        variants_num = 0
        total_num = 0
        # 计算R2
        # 因为beagle4和minimac4不直接提供ap1和ap2，只提供gp和ds，而因为只保留三位小数的限制，导致在计算ap的过程中会报错，所以除了impute5，其他的r2只能直接取
        # 这样就会出现当只计算一部分样本下的r2的时候，只能取整体的r2值做替代。所以如果想要求部分样本值下的准确的r2，就必须使用downsample之后的部分array文件做impute
        if vcf_out_ls[i][7].find("R2=") != -1 :
            r2 = vcf_out_ls[i][7][vcf_out_ls[i][7].find("R2="):]
            r2 = r2.split(";")[0][3:]
            r2 = float(r2)
        elif frequency_value == 0 :
            r2 = 0
        elif frequency_value == 1 :
            r2 = 0
        else :
            r2 = 0
            for k in range(9,len(vcf_ans_ls[i])) :
                if sample_filter_ls[k-9] == False :
                    continue
                ap_ls = vcf_out_ls[i][k].split(":")[2].split(",")
                r2 = r2 + (float(ap_ls[0])-frequency_value)*(float(ap_ls[0])-frequency_value) + (float(ap_ls[1])-frequency_value)*(float(ap_ls[1])-frequency_value)
            r2 = r2 / (2*(len(vcf_ans_ls[i])-9))
            r2 = r2 / ((1-frequency_value)*frequency_value)

        gp_loc = vcf_out_ls[i][8].split(':').index("GP")
        #print(len(vcf_ans_ls[i]))
        #print(len(sample_filter_ls))
        for k in range(9,len(vcf_ans_ls[i])) :
            if sample_filter_ls[k-9] == False :
                continue
            #print(vcf_out_ls[i][k][0:3],"\t",vcf_ans_ls[i][k][0:3])
            if af_rate <= 0.5 :
                # 计算涉及1的准确度
                if vcf_ans_ls[i][k][0:3] not in ['0/0','0|0'] or vcf_out_ls[i][k][0:3] not in ['0/0','0|0'] :
                    total_maf_num += 1
                    if vcf_ans_ls[i][k][0:3] == './.' :
                        right_maf_num += 1
                    elif (vcf_ans_ls[i][k][0:3][0] == vcf_out_ls[i][k][0:3][0] and vcf_ans_ls[i][k][0:3][2] == vcf_out_ls[i][k][0:3][2]) or (vcf_ans_ls[i][k][0:3][0] == vcf_out_ls[i][k][0:3][2] and vcf_ans_ls[i][k][0:3][2] == vcf_out_ls[i][k][0:3][0]):
                        right_maf_num += 1
            else :
                # 计算涉及0的准确度
                if vcf_ans_ls[i][k][0:3] not in ['1/1','1|1'] or vcf_out_ls[i][k][0:3] not in ['1/1','1|1']:
                    total_maf_num += 1
                    if vcf_ans_ls[i][k][0:3] == './.' :
                        right_maf_num += 1
                    elif (vcf_ans_ls[i][k][0:3][0] == vcf_out_ls[i][k][0:3][0] and vcf_ans_ls[i][k][0:3][2] == vcf_out_ls[i][k][0:3][2]) or (vcf_ans_ls[i][k][0:3][0] == vcf_out_ls[i][k][0:3][2] and vcf_ans_ls[i][k][0:3][2] == vcf_out_ls[i][k][0:3][0]):
                        right_maf_num += 1
            # 计算全部的准确度
            if vcf_ans_ls[i][k][0:3] == './.' :
                right_all_num += 1
            elif (vcf_ans_ls[i][k][0:3][0] == vcf_out_ls[i][k][0:3][0] and vcf_ans_ls[i][k][0:3][2] == vcf_out_ls[i][k][0:3][2]) or (vcf_ans_ls[i][k][0:3][0] == vcf_out_ls[i][k][0:3][2] and vcf_ans_ls[i][k][0:3][2] == vcf_out_ls[i][k][0:3][0]):
                right_all_num += 1
            # 计算IQS需要的列表
            if vcf_ans_ls[i][k][0:3] in ['0/0','0|0'] :
                raw_n = 0
            elif vcf_ans_ls[i][k][0:3] in ['0/1','0|1','1/0','1|0'] :
                raw_n = 1
            elif vcf_ans_ls[i][k][0:3] in ['1/1','1|1'] :
                raw_n = 2
            else :
                raw_n = random.randint(0, 2)
            if vcf_out_ls[i][k][0:3] in ['0/0','0|0'] :
                col_n = 0
            elif vcf_out_ls[i][k][0:3] in ['0/1','0|1','1/0','1|0'] :
                col_n = 1
            elif vcf_out_ls[i][k][0:3] in ['1/1','1|1'] :
                col_n = 2
            else :
                col_n = random.randint(0, 2)
            IQS_ls[raw_n*3+col_n] += 1
            # 计算Hell和SEN分数
            if vcf_ans_ls[i][k][0:3] in ['0/0','0|0'] :
                obs_vector = [1,0,0]
            elif vcf_ans_ls[i][k][0:3] in ['0/1','0|1','1/0','1|0'] :
                obs_vector = [0,1,0]
            elif vcf_ans_ls[i][k][0:3] in ['1/1','1|1'] :
                obs_vector = [0,0,1]
            else :
                obs_vector = [0,0,0]
                obs_vector[random.randint(0, 2)] = 1
            
            #print(gp_loc)
            imp_vector = [float(gp) for gp in vcf_out_ls[i][k].split(':')[gp_loc].split(',')]
            Hell_score = 0
            for x in range(len(obs_vector)) :
                Hell_score += math.sqrt(obs_vector[x]*imp_vector[x])
            Hell_score = 1-math.sqrt(1-Hell_score)
            Hell_ls.append(Hell_score)
            #print(Hell_score)
            #print(len(vcf_ans_ls[i]))
            #print(len(sample_filter_ls))
            M_obs_socre = 2-(obs_vector[1]+2*obs_vector[0])
            M_imp_score = 2-(imp_vector[1]+2*imp_vector[0])
            Sen_score = 1-(M_obs_socre-M_imp_score)*(M_obs_socre-M_imp_score)/4
            Sen_ls.append(Sen_score)

            # 计算well impute的snp的数量
            if vcf_ans_ls[i][k][0:3] == './.' and r2 >=0.8 :
                well_right_all_num += 1
            elif ((vcf_ans_ls[i][k][0:3][0] == vcf_out_ls[i][k][0:3][0] and vcf_ans_ls[i][k][0:3][2] == vcf_out_ls[i][k][0:3][2]) or (vcf_ans_ls[i][k][0:3][0] == vcf_out_ls[i][k][0:3][2] and vcf_ans_ls[i][k][0:3][2] == vcf_out_ls[i][k][0:3][0])) and r2 >=0.8:
                well_right_all_num += 1
            
            if vcf_ans_ls[i][k][0].isdigit() and vcf_ans_ls[i][k][2].isdigit() and vcf_out_ls[i][k][0].isdigit() and vcf_out_ls[i][k][2].isdigit():
                ans_gt_1 = min(int(vcf_ans_ls[i][k][0]),int(vcf_ans_ls[i][k][2]))
                ans_gt_2 = max(int(vcf_ans_ls[i][k][0]),int(vcf_ans_ls[i][k][2]))
                out_gt_1 = min(int(vcf_out_ls[i][k][0]),int(vcf_out_ls[i][k][2]))
                out_gt_2 = max(int(vcf_out_ls[i][k][0]),int(vcf_out_ls[i][k][2]))
                TP_num,FP_num,TN_num,FN_num = 0,0,0,0
                if out_gt_1 == 1 :
                    if ans_gt_1 == 1 :
                        TP_num += 1
                    else :
                        FP_num += 1
                else :
                    if ans_gt_1 == 0 :
                        TN_num += 1
                    else :
                        FN_num += 1
                if out_gt_2 == 1 :
                    if ans_gt_2 == 1 :
                        TP_num += 1
                    else :
                        FP_num += 1
                else :
                    if ans_gt_2 == 0 :
                        TN_num += 1
                    else :
                        FN_num += 1
                res_sample_list[af_index][k-9][0] += TP_num
                res_sample_list[af_index][k-9][1] += FP_num
                res_sample_list[af_index][k-9][2] += TN_num
                res_sample_list[af_index][k-9][3] += FN_num
                res_sample_list[af_index][k-9][4] += (TP_num+FP_num+TN_num+FN_num)
            sample_score_list[af_index][k-9][0].append(Hell_score)
            sample_score_list[af_index][k-9][1].append(Sen_score)
        
        res_pos_list[af_index][0].append(int(vcf_ans_ls[i][1]))
        #print(int(vcf_ans_ls[i][1]),"/",i)
        res_pos_list[af_index][1].append(right_all_num)
        res_pos_list[af_index][2].append(len(vcf_ans_ls[i])-9)
        res_pos_list[af_index][3].append(right_maf_num)
        res_pos_list[af_index][4].append(total_maf_num)
        res_pos_list[af_index][5].append(vcf_ans_ls[i][3])
        res_pos_list[af_index][6].append(vcf_ans_ls[i][4])
        res_pos_list[af_index][7].append(well_right_all_num)
        res_pos_list[af_index][8].append(sum(Sen_ls)/len(Sen_ls))
        res_pos_list[af_index][9].append(min(Sen_ls))
        p0 = (IQS_ls[0]+IQS_ls[4]+IQS_ls[8])/sum(IQS_ls)
        pc_ls = [IQS_ls[0]+IQS_ls[1]+IQS_ls[2],IQS_ls[3]+IQS_ls[4]+IQS_ls[5],IQS_ls[6]+IQS_ls[7]+IQS_ls[8],IQS_ls[0]+IQS_ls[3]+IQS_ls[6],IQS_ls[1]+IQS_ls[4]+IQS_ls[7],IQS_ls[2]+IQS_ls[5]+IQS_ls[8]]
        pc = (pc_ls[0]*pc_ls[3]+pc_ls[1]*pc_ls[4]+pc_ls[2]*pc_ls[5])/(sum(IQS_ls)*sum(IQS_ls))
        if 1-pc == 0 :
            res_pos_list[af_index][10].append(0)
        else :
            res_pos_list[af_index][10].append((p0-pc)/(1-pc))
        res_pos_list[af_index][11].append(r2)
        res_pos_list[af_index][12].append(af_rate)
        # 以上是评价指标，包括街廓列表的形成方式，如果修改了评价方式，只需要修改以上代码即可
        ans_num_ls[af_index] += 1
        out_num_ls[af_index] += 1
    
    for i in range(len(res_sample_list)) :
        for j in range(len(res_sample_list[i])) :
            if len(sample_score_list[i][j][0]) == 0:
                res_sample_list[i][j][5] = 0
                res_sample_list[i][j][6] = 0
            else :
                res_sample_list[i][j][5] = min(sample_score_list[i][j][0])
                res_sample_list[i][j][6] = sum(sample_score_list[i][j][0]) / len(sample_score_list[i][j][0])
            if len(sample_score_list[i][j][1]) == 0:
                res_sample_list[i][j][7] = 0
                res_sample_list[i][j][8] = 0
            else :
                res_sample_list[i][j][7] = min(sample_score_list[i][j][1])
                res_sample_list[i][j][8] = sum(sample_score_list[i][j][1]) / len(sample_score_list[i][j][1])

    res_sample_list_fin = [[] for i in range(af_max_index)]
    for i in range(len(res_sample_list)) :
        for j in range(len(res_sample_list[i])) :
            if sample_filter_ls[j] == True :
                res_sample_list_fin[i].append(res_sample_list[i][j])
    samples_num = sum([1 if sample else 0 for sample in sample_filter_ls])
    res_sample_list = res_sample_list_fin

    #print("结果统计完毕")
    write_res_file([af_max_index,af_interval,af_ceiling,af_floor,samples_num],res_pos_list,res_sample_list,res_file)
    print("结果存储完毕：",res_file)
    return res_pos_list

#downsamlpe_type中的all指的是用全部的array做impute，在针对全部结果中的部分样本子集进行计算；part指的是先按照部分样本进行降采样，得到更小的array文件，在进行impute和计算
#af_source设置的时af数值的来源，panelaf对应af数值通过panel文件统计，imputeaf对应af数值由impute工具生成的结果文件提供
def calcul_impute_common_quality(downsamlpe_type,file_base_path,af_source,af_file,impute_tool_ls) :
    # 0-pos,1-all_concordance_right_num,2-all_concordance_total_num,3-maf_concordance_right_num,4-maf_concordance_total_num,5-average_Hellinger_score,6-min_Hellinger_score,7-average_SEN_score,8-min_SEN_score,9-IQS,10-R2,11-af_rate
    ethnic_groups = ['gl','af','am','csa','ea','eu','me','oc']
    ethnic_names = ['global','africa','america','centralsouthasia','eastasia','europe','middleeast','oceania']
    sample_ethnic_names = ['global','Africa (HGDP)','America (HGDP)','Central South Asia (HGDP)','East Asia (HGDP)','Europe (HGDP)','Middle East (HGDP)','Oceania (HGDP)']
    #ethnic_groups = ['gl']
    #ethnic_names = ['global']
    #sample_ethnic_names = ['global']
    #ethnic_groups = ['ea']
    #ethnic_names = ['eastasia']
    #sample_ethnic_names = ['East Asia (HGDP)']
    sample_file=file_base_path+"/sample_file.txt"
    af_file_base = file_base_path[:file_base_path.rfind("/")]
    af_file = af_file_base+"/"+af_file
    hgdp_file_base = file_base_path[:file_base_path.rfind("/")]
    af_ceiling,af_floor = 1.0,0
    #af_interval_ls = [0.001,0.002,0.005,0.01]
    af_interval_ls = [0.001]
    #impute_tool_ls = ['synthetize','beagle5','minimac4','beagle4','impute5']
    #impute_tool_ls = ['beagle5','minimac4','beagle4','impute5']
    #impute_tool_ls = ['synthetize_r2']
    res_floder_name = "res_floder"
    #impute_tool_ls = ['beagle5']
    start_time = time.time()
    if downsamlpe_type == "all" :
        for i in range(len(ethnic_names)) :
            write_hgdp_list(sample_ethnic_names[i],None,sample_file,file_base_path+"/global/hgdp.phased.global.ans.chr22.vcf.gz",hgdp_file_base)
            sample_filter_ls=read_hgdp_list(vcf_file=file_base_path+"/global/hgdp.phased.global.ans.chr22.vcf.gz",sample_file=file_base_path+"/sample_file.txt",is_filter=True)
            for impute_tool in impute_tool_ls :
                ans_file = file_base_path+"/global/hgdp.phased.global.common.ans."+impute_tool+".vcf.gz"
                out_file = file_base_path+"/global/hgdp.phased.global.common.out."+impute_tool+".vcf.gz"
                vcf_ans_ls = read_vcf_file(ans_file,None,False,None)
                vcf_out_ls = read_vcf_file(out_file,None,False,None)
                for af_interval in af_interval_ls :
                    res_file = file_base_path+"/"+res_floder_name+"/res_"+ethnic_groups[i]+"_"+af_source+"_all/res."+impute_tool+"."+str(af_interval)[2:]+".txt"
                    res_list = get_impute_accuracy_common(vcf_ans_ls,vcf_out_ls,af_file,af_source,af_interval,af_ceiling,af_floor,sample_filter_ls,res_file,impute_tool)
    else :
        for i in range(len(ethnic_names)) :
            sample_filter_ls=read_hgdp_list(file_base_path+"/"+ethnic_names[i]+"/hgdp.phased."+ethnic_names[i]+".ans.chr22.vcf.gz",None,is_filter=False)
            for impute_tool in impute_tool_ls :
                ans_file = file_base_path+"/"+ethnic_names[i]+"/hgdp.phased."+ethnic_names[i]+".common.ans."+impute_tool+".vcf.gz"
                out_file = file_base_path+"/"+ethnic_names[i]+"/hgdp.phased."+ethnic_names[i]+".common.out."+impute_tool+".vcf.gz"
                vcf_ans_ls = read_vcf_file(ans_file,None,False,None)
                vcf_out_ls = read_vcf_file(out_file,None,False,None)
                for af_interval in af_interval_ls :
                    res_file = file_base_path+"/"+res_floder_name+"/res_"+ethnic_groups[i]+"_"+af_source+"_part/res."+impute_tool+"."+str(af_interval)[2:]+".txt"
                    res_list = get_impute_accuracy_common(vcf_ans_ls,vcf_out_ls,af_file,af_source,af_interval,af_ceiling,af_floor,sample_filter_ls,res_file,impute_tool)
    end_time = time.time()
    print(file_base_path,"下的",downsamlpe_type,"+",af_source,"的impute效果计算完毕")
    print("耗时：",end_time-start_time)
