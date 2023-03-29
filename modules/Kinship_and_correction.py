# -- coding: utf-8 --
## Integração de codigos FOXCONN

## Kinship e correção de valores de kinship a partir destrutura populacional

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

arg_parser = argparse.ArgumentParser(description = "This is a script to run kinship analysis and correct the values using population stratification", 
	epilog = "Ex. usage: script.py  ") ## AJUSTAR A DESCRIÇÃO

arg_parser.add_argument("-vcf", "--vcf_file", help = "File for processing, requierd for script execution",required=True)
arg_parser.add_argument("-plink", "--plink_path", help = "Path for the plink(1.9) executable, requierd for script execution -- default is to look for the variable on path")
arg_parser.add_argument("-o", "--output_folder", help = "Wanted output folder (default: current output folder)")
arg_parser.add_argument("--window_size", help = "Window size for prunning step -- default = 1000", default="1000")
arg_parser.add_argument("--sliding_window_step", help = "Sliding Window step -- default = 50", default="50")
arg_parser.add_argument("--prune_r2", help = "R2 value for prunning-- default = 0.03", default="0.03")
arg_parser.add_argument("--degree", help = "Degree for relatedeness (INT --> 1, 2 or 3) -- default = 2nd degree [2]", default="2")
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
output_folder = args_dict["output_folder"]
plink_path = args_dict["plink_path"]
threads = args_dict["threads"]
win_size = args_dict["window_size"]
step_size = args_dict["sliding_window_step"]
prune_r2 = args_dict["prune_r2"]
degree = args_dict["degree"]


#######################
## Pre-flight checks ##
#######################

file_path = os.path.abspath(vcf_file)

check_file_exists(file_path)

file_name = file_path.split("/")[-1]

base_name = file_name.split(".")[0]

print("Working on ",base_name)

# Se for dada uma path para output
if output_folder:

    provided_output = os.path.abspath(output_folder)

    # Criar a pasta do usuário na path providenciada
    try:
        os.mkdir(provided_output)
    except:
        print("")

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

#Window size

print(color_text("Using window size of "+str(win_size)))

#Sliding window step

print(color_text("Using step size of "+str(step_size)))

# R2 threshold

print(color_text("Using R2 threshold of "+str(prune_r2)))

print(color_text("Using relatadeness degree of "+str(degree)))

print(color_text("Using "+str(threads)+" Threads"))

if not plink_path:
    plink_look_path = subprocess.run(["which", "plink1.9"], stdout=subprocess.PIPE, text=True)
    plink_path = plink_look_path.stdout.strip()


#######################
## Starting analysis ##
#######################

#tempos de execução
exec_times = []

## Prunning short range

prunning_start = time.time()

print(color_text("Starting prunning short range step"))

indep_tmp_file = os.path.join(temp_files, base_name+"_tmp")

indep_err = os.path.join(temp_files, "indep_short_range.err")
indep_out = os.path.join(temp_files, "indep_short_range.out")

try:
    _try = subprocess.run([plink_path,"--vcf", vcf_file, "--keep-allele-order","--id-delim", "_","--indep-pairwise",win_size,step_size,prune_r2, "--allow-extra-chr","--out", indep_tmp_file, "--make-bed","--threads", threads], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with open(indep_err, "w") as err:
        err.write(_try.stderr)
    with open(indep_out, "w") as out:
        out.write(_try.stdout)
    if _try.stderr:
        print(color_text("WARNING: Plink1.9. Check error log file "+indep_err, "red"))
except:
    print(color_text("Error on Plink1.9 execution", "red"))
    print(color_text("Path used for Plink1.9 executable = "+str(plink_path), "red"))
    print(color_text("Error log is stored in "+indep_err, "yellow"))
    exit(1)

prunning_end = time.time()

prunning_time = (prunning_end - prunning_start)

exec_times.append(["Kinship_prunning", prunning_time])


## Extracting prunned regions

print(color_text("Extracting prunning regions to VCF"))

extract_start = time.time()

prune_in = indep_tmp_file+".prune.in"

for_kinship = os.path.join(temp_files, base_name+".prunned.r2_"+str(prune_r2)+".for_kinship")

prune_err = os.path.join(temp_files, "prune_step.err")
prune_out = os.path.join(temp_files, "prune_step.out")


try:
    _try = subprocess.run([plink_path, "--bfile", indep_tmp_file, "--allow-extra-chr","--extract",prune_in,"--recode", "vcf", "--out",for_kinship], stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, text=True)
    with open(prune_err, "w") as err:
        err.write(_try.stderr)
    with open(prune_out, "w") as out:
        out.write(_try.stdout)
    if _try.stderr:
        print(color_text("WARNING: Plink1.9. Check error log file "+prune_err, "red"))
except:
    print(color_text("Error on Plink1.9 execution", "red"))
    print(color_text("Path used for Plink1.9 executable = "+str(plink_path), "red"))
    print(color_text("Error log is stored in "+prune_err, "yellow"))
    exit(1)

extract_end = time.time()

extract_time = (extract_end - extract_start)

exec_times.append(["Extracting prunned", extract_time])

## Starting Kinship analysis

R_kinship_start = time.time()

print(color_text("Starting Kinship analysis and correction"))

Rscript = os.path.join(script_path, "Kinship_tirando_PCAs_de_Estruturacao.R")
Rscript_err = os.path.join(temp_files, "Rscript_run.err")

try:
    _try = subprocess.run(["Rscript", Rscript, "--vcf", for_kinship+".vcf", "--output", out_dir_path], stderr=subprocess.PIPE, text=True)
    with open(Rscript_err, "w") as err:
        err.write(_try.stderr)
except:
    print(color_text("ERROR: Rscript error, check log error file "+Rscript_err, "red"))

R_kinship_end = time.time()

R_kinship_time = (R_kinship_end - R_kinship_start)

exec_times.append(["Kinship", R_kinship_time])

## Getting related individuals --> essa lista é passada para o plink remover os aparentados antes das analises de pca

rel_start = time.time()

kingship_data = os.path.join(out_dir_path,"Kinship_corrected.tsv")

#MZ = [0.354, 0.5]

first_degree = [0.177, 0.354]

second_degree = [0.0884, 0.177]

third_degree = [0.0442, 0.0884]

king_data = pd.read_csv(kingship_data, sep="\t")

if degree == "1":

    print(color_text("Filtering by degree "+str(degree)))

    related = king_data[king_data["kin"].between(first_degree[0],first_degree[1], inclusive="both") == True]

    related = related["ID1"].unique().tolist()

if degree == "2":

    print(color_text("Filtering by degree "+str(degree)))

    related = king_data[king_data["kin"].between(second_degree[0],second_degree[1], inclusive="both") == True]

    related = related["ID1"].unique().tolist()

if degree == "3":

    print(color_text("Filtering by degree "+str(degree)))

    related = king_data[king_data["kin"].between(third_degree[0],third_degree[1], inclusive="both") == True]

    related = related["ID1"].unique().tolist()

related_output = os.path.join(out_dir_path, "Related_at_degree"+str(degree)+".txt")

print(color_text("Total number of related individuals = "+str(len(related)), "yellow"))

with open(related_output, "w") as output:
    for l in related:
        split_l = l.split("_")
        output.write(split_l[0]+"\t"+split_l[1]+"\n")

rel_end = time.time()

rel_time = (rel_end - rel_start)

exec_times.append(["Relatedeness filter", rel_time])

## Getting VCF with data of related and unrelated individuals

## 1 Unrelated

#/temporario/fpnrossi/bin/bin/bcftools view -S related_indiv_list.txt -Oz -o Pelotas_merged.unicos.R2andQC.RSIDs.MIND.HET.RELATED.vcf.gz Pelotas_merged.unicos.R2andQC.RSIDs.MIND.HET.vcf.gz

#Output for exec times

exec_times_df = pd.DataFrame.from_records(exec_times, columns=["Task", "Time"])

exec_out = os.path.join(out_dir_path,base_name+"_Kinship_execution_times.tsv")

exec_times_df.to_csv(exec_out, index=False, sep="\t")


