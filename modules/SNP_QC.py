## Parte 3 Integração de codigos FOXCONN

## Processo de QC para dados após imputação

# Bibliotecas necessárias

import pandas as pd
import os
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import sys
import argparse
import textwrap
import shutil
import time
import gzip

###############
## Functions ##
###############

tty_colors = {
    'green' : '\033[0;32m%s\033[0m',
    'yellow' : '\033[0;33m%s\033[0m',
    'red' : '\033[0;31m%s\033[0m'
}

# # or, example if wanting the ones from before with background highlighting
# tty_colors = {
#     'green' : '\x1b[6;37;42m%s\x1b[0m',
#     'red' : '\x1b[0;37;41m%s\x1b[0m'
# }

#Color texto de acordo com o "Warning"

def color_text(text, color='green'):

    if sys.stdout.isatty():
        return tty_colors[color] % text
    else:
        return text


def wprint(text):

    print(textwrap.fill(text, width=80, initial_indent="\n  ", 
          subsequent_indent="    ", break_on_hyphens=False))

# Identificar se o arquivo de fato existe

def check_file_exists(file):

	if not os.path.exists(file):
		wprint(color_text("The specified input file '" + str(file) + "' does not seem to exist :(", "red"))
		print("\n  Exiting for now.\n")
		exit(1)

################################
### Help and argument parser ###
################################

arg_parser = argparse.ArgumentParser(description = "This is a script runs standard QC process for imputed datasets", 
    epilog = "Ex. usage: script.py  ") ## AJUSTAR A DESCRIÇÃO

arg_parser.add_argument("-vcf", "--vcf_file", help = "File for processing, requierd for script execution",required=True)
arg_parser.add_argument("-bcftools", "--bcftools_path", help = "Path for the bcftools executable, requierd for script execution -- default is to look for the variable on path")
arg_parser.add_argument("-plink2", "--plink2_path", help = "Path for the plink2 executable, requierd for script execution -- default is to look for the variable on path")
arg_parser.add_argument("-o", "--output_folder", help = "Wanted output folder (default: current output folder)")
arg_parser.add_argument("-geno", "--geno_plink", help = "Threshold value for SNP with missing genotype data -- default=0.05", default="0.05")
#arg_parser.add_argument("-mind", "--mind_plink", help = "Threshold value for individuals with missing genotype data -- default=0.1")
arg_parser.add_argument("-maf", "--maf_plink", help = "Threshold value for minor allele frequency (MAF) -- default=0.01", default="0.01")
arg_parser.add_argument("-HWE", "--hardy", help = "Check for SNPs which are not in Hardy-Weinberg equilibrium (HWE) -- default=1e-6", default="1e-6")
arg_parser.add_argument("-use_HWE", "--use_hardy", help = "Define if the HWE analysis will be run -- default:False", default="False", action="store_true")
arg_parser.add_argument("-R2", "--r_squared", help = "Imputation r-squared threshold value -- default >= 0.8 (Use this flag when dataset was imputed using MIS (Michigan Imputation Server))",default="0.8")
arg_parser.add_argument("-INFO", "--INFO_SCORE", help = "Imputation INFO score threshold value -- default >= 0.5 (Use this flag when dataset was imputed using IMPUTE5)", default="0.5")
arg_parser.add_argument("--score_type", help = "Select r2 or info for imputation score filter -- default: r2", default="r2")
arg_parser.add_argument("--no_score", help = "Dataset with no imputation score -- default: False", default=False, action="store_true")
arg_parser.add_argument("--threads", help = "Number of computer threads -- default = 1", default="1")

#Se nenhum comando foi dado ao script, automaticamente é mostrado o "help"

if len(sys.argv)==1:
    arg_parser.print_help(sys.stderr)
    sys.exit(0)


##################################
### Setting starting variables ###
##################################

# getting primary script full path
path = os.path.realpath(__file__)


# getting primary script directory full path
primary_script_path = path.split("/")[:-1]
primary_script_path = "/".join(primary_script_path)


# setting database full path (assuming they are in the same directory as the primary script)
database_path = os.path.join(primary_script_path, "database")

# setting scripts path (assuming they are in the same directory as the primary script)
script_path = os.path.join(primary_script_path, "scripts")


args = arg_parser.parse_args()
args_dict = vars(arg_parser.parse_args())

vcf_file = args_dict["vcf_file"]
bcftools_path = args_dict["bcftools_path"]
output_folder = args_dict["output_folder"]
plink2_path = args_dict["plink2_path"]
geno_value = args_dict["geno_plink"]
#mind_value = args_dict["mind_plink"]
maf_value = args_dict["maf_plink"]
hwe = args_dict["hardy"]
r2 = args_dict["r_squared"]
info_score = args_dict["INFO_SCORE"]
threads = args_dict["threads"]
use_hwe = args_dict["use_hardy"]
score_type = args_dict["score_type"]
no_score = args_dict["no_score"]

#######################
## Pre-flight checks ##
#######################

file_path = os.path.abspath(vcf_file)

check_file_exists(file_path)

file_name = file_path.split("/")[-1]

base_name = file_name.split(".")[0] #Equivalente ao $G do script!

print("Working on ",base_name)

# Se for dada uma path para output
if output_folder:

    provided_output = os.path.abspath(output_folder)

    # Criar a pasta do usuário na path providenciada
    try:
        os.mkdir(provided_output)
    except:
        print("\n")

    out_dir_path = provided_output

    print(color_text("Using specified directory for output: " + output_folder, "green"))

else:
    #Se não for dado um output, usar o diretório atual
    output_folder = os.getcwd()

    out_dir_path = output_folder

    print(color_text("No output directory specified, using current working directory: " + out_dir_path, "yellow"))

temp_files = os.path.join(out_dir_path, "tmp")
try:
    os.mkdir(temp_files)
except:
    print(color_text("tmp folder exists, will keep using it"))

#Geno value
print(color_text("Using specified value for SNP with missing genotype data of "+str(geno_value),"green"))

#MAF value
print(color_text("Using specified MAF value of "+str(maf_value),"green"))

#R2 value
print(color_text("Using specified R2 threshold value of "+str(r2),"green"))

#Threads
print(color_text("Using "+threads+" threads"))

if not bcftools_path:
    bcftools_look_for_path = subprocess.run(["which", "bcftools"], stdout=subprocess.PIPE, text=True)

    bcftools_path = bcftools_look_for_path.stdout.strip()

if not plink2_path:
    plink2_look_for_path = subprocess.run(["which", "plink2"], stdout=subprocess.PIPE, text=True)

    plink2_path = plink2_look_for_path.stdout.strip()

#######################
## Starting analysis ##
#######################

#tempos de execução
exec_times = []

## Processo de QC com plink2
print(color_text("Starting Plink2 QC process"))

plinkQC_start = time.time()

plink_out_file = file_name.replace(".vcf.gz", ".R2andQC")

plink_process_output = os.path.join(out_dir_path, plink_out_file)

qc_process_err = os.path.join(temp_files, "QC_process.err")
qc_process_out = os.path.join(temp_files, "QC_process.out")

if not no_score:

    if score_type == "r2":

        try:
            _try = subprocess.run([plink2_path, "--extract-if-info","R2",">=",r2,"--allow-extra-chr","--geno",geno_value,"--keep-allele-order","--maf",maf_value,
                "--recode", "vcf","bgz","--vcf",vcf_file,"--out",plink_process_output, "--threads",threads],stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True)
            with open(qc_process_err, "w") as err:
                err.write(_try.stderr)
            with open(qc_process_out, "w") as out:
                out.write(_try.stdout)
            if _try.stderr:
                print(color_text("ERROR: Plink2. Check error log file "+qc_process_err, "red"))
                exit(1)

        except:
            print(color_text("Error on Plink2 execution", "red"))
            print(color_text("Path used for Plink2 executable = "+str(plink2_path), "red"))
            print(color_text("Error log is stored in "+qc_process_err, "yellow"))
            exit(1)


    if score_type == "info":
        r2 = info_score
        try:
            _try = subprocess.run([plink2_path, "--extract-if-info","INFO",">=",r2,"--allow-extra-chr","--geno",geno_value,"--keep-allele-order","--maf",maf_value,
            "--recode", "vcf","bgz","--vcf",vcf_file,"--out",plink_process_output, "--threads",threads],stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True)
            with open(qc_process_err, "w") as err:
                err.write(_try.stderr)
            with open(qc_process_out, "w") as out:
                out.write(_try.stdout)
            if _try.stderr:
                print(color_text("ERROR: Plink2. Check error log file "+qc_process_err, "yellow"))
        except:
            print(color_text("Error on Plink2 execution", "red"))
            print(color_text("Path used for Plink2 executable = "+str(plink2_path), "red"))
            print(color_text("Error log is stored in "+qc_process_err, "yellow"))
            exit(1)

if no_score:
        try:
            _try = subprocess.run([plink2_path,"--allow-extra-chr","--geno",geno_value,"--keep-allele-order","--maf",maf_value,
                "--recode", "vcf","bgz","--vcf",vcf_file,"--out",plink_process_output, "--threads",threads],stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True)
            with open(qc_process_err, "w") as err:
                err.write(_try.stderr)
            with open(qc_process_out, "w") as out:
                out.write(_try.stdout)
            if _try.stderr:
                print(color_text("WARNING: Plink2. Check error log file "+qc_process_err, "yellow"))

        except:
            print(color_text("Error on Plink2 execution", "red"))
            print(color_text("Path used for Plink2 executable = "+str(plink2_path), "red"))
            print(color_text("Error log is stored in "+qc_process_err, "yellow"))
            exit(1)

if use_hwe == "True":
    print(color_text("Using HWE value for control of variant quality"+"\n"+"Using hwe value of "+str(hwe),"green"))
    if score_type == "r2":

        try:
            _try = subprocess.run([plink2_path, "--extract-if-info","R2",">=",r2,"--allow-extra-chr","--geno",geno_value,"--keep-allele-order","--maf",maf_value,
                "--recode", "vcf","bgz","--vcf",vcf_file,"--out",plink_process_output, "--threads",threads],stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True)
            with open(qc_process_err, "w") as err:
                err.write(_try.stderr)
            with open(qc_process_out, "w") as out:
                out.write(_try.stdout)
            if _try.stderr:
                print(color_text("ERROR: Plink2. Check error log file "+qc_process_err, "red"))
                exit(1)

        except:
            print(color_text("Error on Plink2 execution", "red"))
            print(color_text("Path used for Plink2 executable = "+str(plink2_path), "red"))
            print(color_text("Error log is stored in "+qc_process_err, "yellow"))
            exit(1)

    if score_type == "info":
        r2 = info_score
        try:
            _try = subprocess.run([plink2_path, "--extract-if-info","INFO",">=",r2,"--allow-extra-chr","--geno",geno_value,"--keep-allele-order","--maf",maf_value,
            "--recode", "vcf","bgz","--vcf",vcf_file,"--out",plink_process_output, "--threads",threads],stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True)
            with open(qc_process_err, "w") as err:
                err.write(_try.stderr)
            with open(qc_process_out, "w") as out:
                out.write(_try.stdout)
            if _try.stderr:
                print(color_text("ERROR: Plink2. Check error log file "+qc_process_err, "red"))
                exit(1)
        except:
            print(color_text("Error on Plink2 execution", "red"))
            print(color_text("Path used for Plink2 executable = "+str(plink2_path), "red"))
            print(color_text("Error log is stored in "+qc_process_err, "yellow"))
            exit(1)


plinkQC_end = time.time()

plink_time = (plinkQC_end - plinkQC_start)

exec_times.append(["Plink2QC", plink_time])


print(color_text("Compressing and Indexing file"))

reindex_start = time.time()

index_input = plink_process_output+".vcf.gz"

try:

    subprocess.run([bcftools_path,"index","-f","--threads",threads,"-t",index_input])
except:
    print(color_text("Error on bcftools execution", "red"))
    print(color_text("Path used for bcftools executable = "+str(bcftools_path), "red"))
    exit(1)

reindex_end = time.time()

reindex_time = (reindex_end - reindex_start)

exec_times.append(["Reindex_QC", reindex_time])

exec_times_df = pd.DataFrame.from_records(exec_times, columns=["Task", "Time"])

exec_out = os.path.join(out_dir_path,base_name+"_QC_execution_times.tsv")

exec_times_df.to_csv(exec_out, index=False, sep="\t")