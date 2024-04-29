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
# 该函数处理array_file文件，如果is_cover_pos为true，会按比例将源文件分为ans_file，tar_file，比例为cover_pos_rate：1-cover_pos_rate
# cover_rate是将vcf中的gt进行mask的比例，一般为0
def random_cover_gt(array_file,cover_rate,tar_file,ans_file,is_cover_pos,cover_pos_rate) :
    if array_file[-4:] == ".vcf" :
        tar_line_ls = []
        ans_line_ls = []
        line_num = 0
        with open(array_file,'r') as f:
            while(True) :
                f_line = f.readline().strip()
                if f_line == '' :
                    break
                if f_line[0] == '#' :
                    tar_line_ls.append(f_line+'\n')
                    ans_line_ls.append(f_line+'\n')
                    continue
                f_line_ls = f_line.split('\t')
                vcf_line = ''
                vcf_ls = []
                for i in range(9) :
                    vcf_ls.append(f_line_ls[i])
                for i in range(9,len(f_line_ls)) :
                    if f_line_ls[i] not in ['./.','0/0','1/0','0/1','1/1','0|0','0|1','1|0','1|1'] :
                        vcf_ls.append('./.')
                    else :
                        vcf_ls.append(f_line_ls[i])
                if is_cover_pos == True :
                    if random.random() < cover_pos_rate :
                        for i in range(len(vcf_ls)) :
                            vcf_line += vcf_ls[i] + '\t'
                        vcf_line = vcf_line[:-1] + '\n'
                        ans_line_ls.append(vcf_line)
                    else :
                        for i in range(len(vcf_ls)) :
                            if i >= 9 and random.random() < cover_rate :
                                vcf_line += './.' + '\t'
                            else :
                                vcf_line += vcf_ls[i] + '\t'
                        vcf_line = vcf_line[:-1] + '\n'
                        tar_line_ls.append(vcf_line)
                else :
                    for i in range(len(vcf_ls)) :
                        if i >= 9 and random.random() < cover_rate :
                            vcf_line += './.' + '\t'
                        else :
                            vcf_line += vcf_ls[i] + '\t'
                    vcf_line = vcf_line[:-1] + '\n'
                    tar_line_ls.append(vcf_line)
                line_num += 1
                #print("已处理vcf条数为",line_num)
            f.close()
    if array_file[-3:] == ".gz" :
        tar_line_ls = []
        ans_line_ls = []
        line_num = 0
        with gzip.open(array_file, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                if row[0][0:2] == '##' :
                    tar_line_ls.append(row[0].strip()+'\n')
                    ans_line_ls.append(row[0].strip()+'\n')
                    continue
                vcf_line = ''
                vcf_ls = []
                if row[0][0] == "#" :
                    for i in range(len(row)) :
                        vcf_line = vcf_line + row[i] + "\t"
                    vcf_line = vcf_line[:-1]
                    ans_line_ls.append(vcf_line+"\n")
                    tar_line_ls.append(vcf_line+"\n")
                    continue
                for i in range(9) :
                    vcf_ls.append(row[i])
                for i in range(9,len(row)) :
                    if row[i] not in ['./.','0/0','1/0','0/1','1/1','0|0','0|1','1|0','1|1'] :
                        vcf_ls.append('./.')
                    else :
                        vcf_ls.append(row[i])
                if is_cover_pos == True :
                    if random.random() < cover_pos_rate :
                        for i in range(len(vcf_ls)) :
                            vcf_line += vcf_ls[i] + '\t'
                        vcf_line = vcf_line[:-1] + '\n'
                        ans_line_ls.append(vcf_line)
                    else :
                        for i in range(len(vcf_ls)) :
                            if i >= 9 and random.random() < cover_rate :
                                vcf_line += './.' + '\t'
                            else :
                                vcf_line += vcf_ls[i] + '\t'
                        vcf_line = vcf_line[:-1] + '\n'
                        tar_line_ls.append(vcf_line)
                else :
                    for i in range(len(vcf_ls)) :
                        if i >= 9 and random.random() < cover_rate :
                            vcf_line += './.' + '\t'
                        else :
                            vcf_line += vcf_ls[i] + '\t'
                    vcf_line = vcf_line[:-1] + '\n'
                    tar_line_ls.append(vcf_line)
                line_num += 1
                #print("已处理vcf条数为",line_num)
            csvfile.close()
    print("已处理vcf条数为",line_num)
    if tar_file != None :
        with gzip.open(tar_file, 'wt') as f:
            for line in tar_line_ls :
                f.write(line)
            f.close()
        cmd_list_1 = "gunzip " + tar_file
        cmd_list_2 = "bgzip -f " + tar_file[:-3] + " > " + tar_file
        cmd_list_3 = "tabix " + tar_file
        subprocess.run(cmd_list_1,shell=True,capture_output=True)
        subprocess.run(cmd_list_2,shell=True,capture_output=True)
        subprocess.run(cmd_list_3,shell=True,capture_output=True)
    if ans_file != None :
        if is_cover_pos == True :
            with gzip.open(ans_file, 'wt') as f:
                for line in ans_line_ls :
                    f.write(line)
                f.close()
            cmd_list_1 = "gunzip " + ans_file
            cmd_list_2 = "bgzip -f " + ans_file[:-3] + " > " + ans_file
            cmd_list_3 = "tabix " + ans_file
            subprocess.run(cmd_list_1,shell=True,capture_output=True)
            subprocess.run(cmd_list_2,shell=True,capture_output=True)
            subprocess.run(cmd_list_3,shell=True,capture_output=True)
    return

# 将所有数据行分割为列表，在组装为一个高维列表
# 运行改代码时需要注意：由于vcf.gz文件的特殊性，需要为该代码预留出大量的内存空间，单独运行该函数需要50-100g左右，运行完整的get_impute_accuracy函数需要150-200g左右。好在该代码一般在几分钟之内就能结束。
# 如果exculde_info为True，在加载vcf文件时，会按照vcf文件中info列记录的是impute新推断出的信息来去除数据，如minimac4中是“TYPED;IMPUTED;”，beagle5中是没有“IMP”，满足这些的都是推断前的文件就有的变异，在验证过程中可以不使用这些信息  
# exclude_vcf_file和exculde_info可以同时起筛选作用
# ans文件不能经过exculde_info筛选，实际上只要把ans和tar文件分开，ans文件就不需要筛选了
# impute_tool是不同impute工具的参数，用来区分不同工具的不同输出格式
def read_vcf_file(vcf_file,exclude_vcf_file,exculde_info,impute_tool) :
    if vcf_file[-4:] == ".vcf" :
        vcf_line_ls = []
        line_num = 0
        with open(vcf_file,'r') as f:
            while(True) :
                f_line = f.readline().strip()
                if f_line == '' :
                    break
                if f_line[0] == '#' :
                    continue
                if exculde_info :
                    if impute_tool in ['beagle5','beagle4','impute5'] and f_line[7].find('IMP') == -1 :
                        continue
                    if impute_tool == 'minimac4' and (f_line[7].find('IMPUTED') == -1 or (f_line[7].find('IMPUTED') != -1 and f_line[7].find('TYPED') != -1)) :
                        continue
                vcf_line_ls.append(f_line.split('\t'))
                line_num += 1
                #print("从文件",vcf_file,"已读取数据量为：",line_num)
            f.close()
        #print("从文件",vcf_file,"已读取数据量为：",len(vcf_line_ls))
    if vcf_file[-3:] == ".gz" :
        vcf_line_ls = []
        line_num = 0
        with gzip.open(vcf_file, 'rt') as csvfile:
            #print(vcf_file)
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                if row[0][0] == '#' :
                    continue
                if exculde_info :
                    if impute_tool in ['beagle5','beagle4','impute5'] and row[7].find('IMP') == -1 :
                        continue
                    if impute_tool == 'minimac4' and (row[7].find('IMPUTED') == -1 or (row[7].find('IMPUTED') != -1 and row[7].find('TYPED') != -1)) :
                        continue
                vcf_line_ls.append(row)
                line_num += 1
                #print(len(vcf_line_ls))
            #print("从文件",vcf_file,"已读取数据量为：",line_num)
            csvfile.close()
    #print("vcf file 1读取完毕")
    # 如果exclude_vcf_file是None，那么直接返回vcf_file文件读取的内容；如果不是None，那么返回vcf_file中有，但是exclude_vcf_file没有的文件内容
    if exclude_vcf_file == None :
        return vcf_line_ls
    
    if exclude_vcf_file[-4:] == ".vcf" :
        exclude_vcf_line_ls = []
        line_num = 0
        with open(exclude_vcf_file,'r') as f:
            while(True) :
                f_line = f.readline().strip()
                if f_line == '' :
                    break
                if f_line[0] == '#' :
                    continue
                exclude_vcf_line_ls.append(f_line.split('\t'))
                line_num += 1
                #print("从文件",vcf_file,"已读取数据量为：",line_num)
            f.close()
        #print("从文件",vcf_file,"已读取数据量为：",len(vcf_line_ls))
    if exclude_vcf_file[-3:] == ".gz" :
        exclude_vcf_line_ls = []
        line_num = 0
        with gzip.open(exclude_vcf_file, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                if row[0][0] == '#' :
                    continue
                exclude_vcf_line_ls.append(row)
                line_num += 1
                #print("从文件",vcf_file,"已读取数据量为：",line_num)
            csvfile.close()
    #print("vcf file 2读取完毕")
    # 将vcf_file中的内容按照exclude_vcf_file排除
    i = j = 0
    res_line_ls = []
    while(i<len(vcf_line_ls) and j<len(exclude_vcf_line_ls)) :
        if vcf_line_ls[i][1] < exclude_vcf_line_ls[j][1] :
            res_line_ls.append(vcf_line_ls[i])
            i += 1
            continue
        elif vcf_line_ls[i][1] == exclude_vcf_line_ls[j][1] :
            if vcf_line_ls[i][3] == exclude_vcf_line_ls[j][3] and vcf_line_ls[i][4] == exclude_vcf_line_ls[j][4] :
                i += 1
                continue
            else :
                k=j
                find_flag = False
                while(k<len(exclude_vcf_line_ls)) :
                    if vcf_line_ls[i][1] == exclude_vcf_line_ls[k][1] and vcf_line_ls[i][3] == exclude_vcf_line_ls[k][3] and vcf_line_ls[i][4] == exclude_vcf_line_ls[k][4] :
                        find_flag = True
                        break
                    elif vcf_line_ls[i][1] != exclude_vcf_line_ls[k][1] :
                        break
                    k += 1
                if find_flag == False :
                    res_line_ls.append(vcf_line_ls[i])
                    i += 1
                    continue
                else :
                    i += 1
                    continue
        else :
            j += 1
            continue
    return res_line_ls

# 获取vcf文件中的各种信息
# 包括样本列表，各种标签位置等等
def read_vcf_file_info(vcf_file) :
    #print(vcf_file)
    if vcf_file[-4:] == ".vcf" :
        with open(vcf_file,'r') as f:
            while(True) :
                f_line = f.readline().strip()
                if f_line == '' :
                    break
                elif f_line[0:2] == '##' :
                    continue
                elif f_line[0] == '#' and f_line[1] != '#' :
                    sample_ls = f_line.split('\t')[9:]
                else :
                    if "AP" in f_line.split('\t')[8].split(":") :
                        ap_index = f_line.split('\t')[8].split(":").index("AP")
                    elif "HDS" in f_line.split('\t')[8].split(":") :
                        ap_index = f_line.split('\t')[8].split(":").index("HDS")
                    else :
                        ap_index = -1*f_line.split('\t')[8].split(":").index("AP1")
                    gp_index = f_line.split('\t')[8].split(":").index("GP")
                    gt_index = f_line.split('\t')[8].split(":").index("GT")
                    break
            f.close()
        #print("从文件",vcf_file,"已读取数据量为：",len(vcf_line_ls))
    if vcf_file[-3:] == ".gz" :
        with gzip.open(vcf_file, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                if row[0][0:2] == '##' :
                    continue
                elif row[0][0] == '#' and row[0][1] != '#' :
                    sample_ls = row[9:]
                else :
                    #print(row[8])
                    if "AP" in row[8].split(":") :
                        ap_index = row[8].split(":").index("AP")
                    elif "HDS" in row[8].split(":") :
                        ap_index = row[8].split(":").index("HDS")
                    else :
                        ap_index = -1*row[8].split(":").index("AP1")
                    gp_index = row[8].split(":").index("GP")
                    gt_index = row[8].split(":").index("GT")
                    break
            csvfile.close()
    return sample_ls,gp_index,gt_index,ap_index

# 将内存中的vcf内容写入文件中
def write_vcf_file(output_vcf_file,head_vcf_file,vcf_line_ls) :
    if head_vcf_file != None :
        head_line_ls = []
        with gzip.open(head_vcf_file, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                if row[0][0] == '#' :
                    head_line_ls.append(row)
            csvfile.close()
        vcf_line_ls = head_line_ls + vcf_line_ls
    with gzip.open(output_vcf_file, 'wt') as f:
        for line in vcf_line_ls :
            write_line = ""
            for i in range(len(line)) :
                write_line += line[i]+'\t'
            f.write(write_line[:-1]+'\n')
        f.close()
    #cmd_list_1 = "gunzip " + output_vcf_file
    #cmd_list_2 = "bgzip -f " + output_vcf_file[:-3] + " > " + output_vcf_file
    #cmd_list_3 = "tabix " + output_vcf_file

def read_vcf_snp_sample_num(vcf_file) :
    snp_num = 0
    if vcf_file[-3:] == ".gz" :
        with gzip.open(vcf_file, 'rt') as f:
            while(True) :
                f_line = f.readline().rstrip('\n')
                if f_line == "" :
                    break
                elif f_line[0:2] == "##" :
                    continue
                elif f_line[0] == "#" and f_line[1] != "#" :
                    sample_ls = f_line.split("\t")[9:]
                else :
                    snp_num += 1
            f.close()
    return sample_ls,snp_num

# vcf_line_ls是计算af频率来源的列表，fliter_line_ls是一个过滤列表，vcf_line_ls中的变异，只有pos,ref,alt同时能够在fliter_line_ls找到一致的变异的时候，才会参与af频率计算，fliter_line_ls为none，则不起筛选作用
def get_af_list(vcf_line_ls,fliter_line_ls,af_interval,af_ceiling,af_floor) :
    # pos list + ref list + alt list + af rate index list + af rate +af information list
    af_list = [[],[],[],[],[],[]]
    af_max_index = int((af_ceiling-af_floor)/af_interval)
    j = 0
    for var_line in vcf_line_ls :
        is_same_flag = False
        if fliter_line_ls != None and j < len(fliter_line_ls) :
            is_same_flag = False
            while(int(fliter_line_ls[j][1]) < int(var_line[1])) :
                j += 1
                if j >= len(fliter_line_ls) :
                    break
            if j >= len(fliter_line_ls) :
                break
            if int(var_line[1]) < int(fliter_line_ls[j][1]) :
                continue
            else :
                k = j
                while(int(var_line[1]) == int(fliter_line_ls[k][1])) :
                    if var_line[3] == fliter_line_ls[k][3] and var_line[4] == fliter_line_ls[k][4] :
                        is_same_flag = True
                        break
                    k += 1
                    if k >= len(fliter_line_ls) :
                        break
        if fliter_line_ls == None :
            is_same_flag = True
        if not is_same_flag :
            continue
        var_num = 0
        total_num = 0
        for i in range(9,len(var_line)) :
            if var_line[i][0] == '0' :
                total_num += 1
            elif var_line[i][0] == '1' :
                total_num += 1
                var_num += 1
            if var_line[i][2] == '0' :
                total_num += 1
            elif var_line[i][2] == '1' :
                total_num += 1
                var_num += 1
        af_rate = var_num / total_num
        # 上限给最后一个区间，其他区间左闭右开
        if af_rate > af_ceiling :
            continue
        elif af_rate == af_ceiling :
            af_index = af_max_index - 1
        else :
            af_index = math.floor((af_rate - af_floor)/af_interval)
        af_list[0].append(int(var_line[1]))
        af_list[1].append(var_line[3])
        af_list[2].append(var_line[4])
        af_list[3].append(af_index)
        af_list[4].append(af_rate)
    af_list[5].append(af_max_index)
    af_list[5].append(af_interval)
    af_list[5].append(af_ceiling)
    af_list[5].append(af_floor)
    return af_list
