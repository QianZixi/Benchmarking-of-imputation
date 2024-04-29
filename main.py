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
from data_process import *
from evaluate_imputation import *
from make_csv import *

def evaluate_imputation_res() :
    af_file = "chr22.concat.all.af.txt"
    tools_ls = ["synthetizenext","beagle4","beagle5","minimac4","impute5"]
    evaluate_imputation.calcul_impute_common_quality("all","XXX/ref100","panelaf",af_file,tools_ls)

# 实验1：测试不同工具的impute效果
def fun_to_res_1() :
    af_file = "ref100/chr22.concat.all.af.txt"
    tools_ls = ["beagle4","beagle5","impute5","minimac4"]
    tools_names = ["beagle4","beagle5","impute5","minimac4"]
    res_csv = "res_csv_1"
    #calcul_impute_common_quality("all","XXX/ref100","panelaf",af_file,tools_ls)
    downsamlpe_type,af_source,res_base = "all","panelaf","XXX/ref100/res_floder"
    make_csv.write_all_impute_csv(downsamlpe_type,af_source,res_base,tools_ls,tools_names,True,res_csv)

# 实验2：测试不同panel样本数量的impute效果
def fun_to_res_2() :
    af_file = "ref100/chr22.concat.all.af.txt"
    downsamlpe_type,af_source,res_base = "all","panelaf","XXX/"
    res_csv = "res_csv_2"
    make_csv.write_all_size_csv(downsamlpe_type,af_source,res_base,res_csv)
    
    #tools_ls = ["array01","array02","array03","array04","array05","array06","array07","array08","array09"]
    #calcul_impute_common_quality("all","XXX/ref100","panelaf",af_file,tools_ls)

# 实验2：计算十个不同尺寸子panel的样本数量
def fun_to_res_2_sample_num() :
    vcf_file = "XXX/ref100/ref/chr22.concat.all.vcf.gz"
    sample_ls,snp_num = data_process.read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(len(sample_ls))
    vcf_file = "XXX/ref90/ref/chr22.concat.all.vcf.gz"
    sample_ls,snp_num = data_process.read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(len(sample_ls))
    vcf_file = "XXX/ref80/ref/chr22.concat.all.vcf.gz"
    sample_ls,snp_num = data_process.read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(len(sample_ls))
    vcf_file = "XXX/ref70/ref/chr22.concat.all.vcf.gz"
    sample_ls,snp_num = data_process.read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(len(sample_ls))
    vcf_file = "XXX/ref60/ref/chr22.concat.all.vcf.gz"
    sample_ls,snp_num = data_process.read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(len(sample_ls))
    vcf_file = "XXX/ref50/ref/chr22.concat.all.vcf.gz"
    sample_ls,snp_num = data_process.read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(len(sample_ls))
    vcf_file = "XXX/ref40/ref/chr22.concat.all.vcf.gz"
    sample_ls,snp_num = data_process.read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(len(sample_ls))
    vcf_file = "XXX/ref30/ref/chr22.concat.all.vcf.gz"
    sample_ls,snp_num = data_process.read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(len(sample_ls))
    vcf_file = "XXX/ref20/ref/chr22.concat.all.vcf.gz"
    sample_ls,snp_num = data_process.read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(len(sample_ls))
    vcf_file = "XXX/ref10/ref/chr22.concat.all.vcf.gz"
    sample_ls,snp_num = data_process.read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(len(sample_ls))
    #downsamlpe_type,af_source,res_base = "all","panelaf","XXX/ref100/res_floder"
    #write_all_impute_csv(downsamlpe_type,af_source,res_base,tools_ls)

# 实验3：测试是否使用array部分样本引起的impute效果变化
def fun_to_res_3() :
    af_file = "ref100/chr22.concat.all.af.txt"
    tools_ls = ["beagle4","beagle5","impute5","minimac4"]
    res_csv = "res_csv_3"
    #calcul_impute_common_quality("part","XXX/ref100","panelaf",af_file,tools_ls)
    for impute_tool in tools_ls :
        af_source,res_base = "panelaf","XXX/ref100/res_floder"
        write_all_downsample_csv(af_source,impute_tool,res_base,res_csv)
    #downsamlpe_type,af_source,res_base = "all","panelaf","XXX/ref100/res_floder"
    #write_all_impute_csv(downsamlpe_type,af_source,res_base,tools_ls)

# 实验4：测试不同array变异数量的impute效果
def fun_to_res_4() :
    af_file = "ref100/chr22.concat.all.af.txt"
    tools_ls = ["array01","array02","array03","array04","array05","array06","array07","array08","array09","array10"]
    tools_names = ["10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"]
    res_csv = "res_csv_4"
    #calcul_impute_common_quality("all","XXX/ref100","panelaf",af_file,tools_ls)
    downsamlpe_type,af_source,res_base = "all","panelaf","XXX/ref100/res_floder"
    write_all_impute_csv(downsamlpe_type,af_source,res_base,tools_ls,tools_names,True,res_csv)

# 实验4：获取不同尺寸子array的变异数量
def fun_to_res_4_snp_num() :
    vcf_file = "XXX/ref100/global/hgdp.phased.global.tar.01.chr22.vcf.gz"
    sample_ls,snp_num = read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(snp_num)
    vcf_file = "XXX/ref100/global/hgdp.phased.global.tar.02.chr22.vcf.gz"
    sample_ls,snp_num = read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(snp_num)
    vcf_file = "XXX/ref100/global/hgdp.phased.global.tar.03.chr22.vcf.gz"
    sample_ls,snp_num = read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(snp_num)
    vcf_file = "XXX/ref100/global/hgdp.phased.global.tar.04.chr22.vcf.gz"
    sample_ls,snp_num = read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(snp_num)
    vcf_file = "XXX/ref100/global/hgdp.phased.global.tar.05.chr22.vcf.gz"
    sample_ls,snp_num = read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(snp_num)
    vcf_file = "XXX/ref100/global/hgdp.phased.global.tar.06.chr22.vcf.gz"
    sample_ls,snp_num = read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(snp_num)
    vcf_file = "XXX/ref100/global/hgdp.phased.global.tar.07.chr22.vcf.gz"
    sample_ls,snp_num = read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(snp_num)
    vcf_file = "XXX/ref100/global/hgdp.phased.global.tar.08.chr22.vcf.gz"
    sample_ls,snp_num = read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(snp_num)
    vcf_file = "XXX/ref100/global/hgdp.phased.global.tar.09.chr22.vcf.gz"
    sample_ls,snp_num = read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(snp_num)
    vcf_file = "XXX/ref100/global/hgdp.phased.global.tar.chr22.vcf.gz"
    sample_ls,snp_num = read_vcf_snp_sample_num(vcf_file)
    print(vcf_file)
    print(snp_num)
    #downsamlpe_type,af_source,res_base = "all","panelaf","XXX/ref100/res_floder"
    #write_all_impute_csv(downsamlpe_type,af_source,res_base,tools_ls)

def main(argv):
    fun_to_res_1()
    fun_to_res_2()
    fun_to_res_4()

if __name__ == "__main__":
    main(sys.argv[1:])