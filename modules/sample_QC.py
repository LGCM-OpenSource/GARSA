## Processo de QC para individuos com missing data ou erros de heterozigosidade

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

#Função criada para analise de heterozigosidade - Fernando P N Rossi

def Het_check(het_txt, fail_het_txt):
    with open(het_txt, "r") as h:
        h_list = []
        for _row in h:
            het = _row.split()
            h_list.append(het)
    het_dataset = pd.DataFrame.from_records(h_list[1:], columns=h_list[0])

    ## High levels of heterozygosity within an individual might be an indication of low sample quality

    ### Off levels of heterozygosity --> values that are < mean(HET_RATE) -3sd(HET_RATE) | values that are > mean(HET_RATE) +3sd(HET_RATE)

    ### It starts caculating (N(NM) - O(HOM))/N(NM) --> creating the HET_RATE columns 

    ### Number of non-missing genotypes - observed number of homozygotes / Number of non-missing genotypes = Heterozygosity Rate

    het_dataset["HET_RATE"] = (het_dataset["N(NM)"].astype("float") - het_dataset["O(HOM)"].astype("float"))/het_dataset["N(NM)"].astype("float")

    ## filetring the dataset for off levels of heterozygosity

    het_fail = het_dataset.loc[(het_dataset["HET_RATE"] < het_dataset["HET_RATE"].mean() - (3*het_dataset["HET_RATE"].std())) | (het_dataset["HET_RATE"] > het_dataset["HET_RATE"].mean() + (3*het_dataset["HET_RATE"].std()))]
    #saving the data --> for plink the data need to be saved having just the first 2 columns

    het_headers = het_fail[["FID", "IID"]].astype("str").columns.values.tolist()
    het_data = het_fail[["FID", "IID"]].astype("str").values.tolist()
    het_fail_data = [het_headers] + het_data

    with open(fail_het_txt, "w") as het:
        for _line in het_fail_data:
            het.write("\t".join(_line)+"\n")


################################
### Help and argument parser ###
################################

arg_parser = argparse.ArgumentParser(description = "This is a script runs standard QC process for imputed datasets", 
    epilog = "Ex. usage: script.py  ") ## AJUSTAR A DESCRIÇÃO

arg_parser.add_argument("-vcf", "--vcf_file", help = "File for processing, requierd for script execution",required=True)
arg_parser.add_argument("-plink", "--plink_path", help = "Path for the plink(1.9) executable, requierd for script execution -- default is to look for the variable on path")
arg_parser.add_argument("-mind", "--mind_plink", help = "Threshold value for individuals with missing genotype data -- default=0.1", default="0.1")
arg_parser.add_argument("--threads", help = "Number of computer threads -- default = 1")
arg_parser.add_argument("-o", "--output_folder", help = "Wanted output folder (default: current output folder)")

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
mind_value = args_dict["mind_plink"]
output_folder = args_dict["output_folder"]
plink_path = args_dict["plink_path"]
threads = args_dict["threads"]



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

#Mind value
print(color_text("Using value for individuals with missing genotype data of "+str(mind_value),"yellow"))

print(color_text("Using "+threads+" threads"))


if not plink_path:
    plink_look_path = subprocess.run(["which", "plink1.9"], stdout=subprocess.PIPE, text=True)
    plink_path = plink_look_path.stdout.strip()


#######################
## Starting analysis ##
#######################

#tempos de execução
exec_times = []


#Removing individuals with high missing data

mind_start = time.time()

print(color_text("Starting removal of individuals with high missing data"))

mind_out = os.path.join(temp_files, base_name+".MIND")

mind_err = os.path.join(temp_files, "mind_check.err")
mind_out_log = os.path.join(temp_files, "mind_check.out")

try:
    _try = subprocess.run([plink_path, "--vcf", vcf_file,"--keep-allele-order","--id-delim", "_","--mind", mind_value, "--allow-extra-chr","--make-bed", "--out", mind_out, "--threads", threads], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with open(mind_err) as err:
        err.write(_try.stderr)
    with open(mind_out_log) as out:
        out.write(_try.stdout)
    if _try.stderr:
        print(color_text("WARNING: Plink1.9. Check error log file "+mind_err, "yellow"))
except:
    print("")
    # print(color_text("Error on Plink1.9 execution", "red"))
    # print(color_text("Path used for Plink1.9 executable = "+str(plink_path), "red"))
    # print(color_text("Error log is stored in "+mind_err, "yellow"))
    # exit(1)


mind_end = time.time()

mind_time = (mind_end - mind_start)

exec_times.append(["Mind", mind_time])

## Prcesso de filtro de Heterozigosidade

#het input = plink_process_output --> output gerado pelo processamento acima

het_start = time.time()

print(color_text("Starting heterozygosity rate analysis", "green"))

Het_output = os.path.join(temp_files, base_name+"_Het")

het_err = os.path.join(temp_files, "het_check.err")
het_out = os.path.join(temp_files, "het_check.out")

#Het_input = mind_out+".vcf.gz"

try:
    _try = subprocess.run([plink_path, "--bfile", mind_out, "--allow-extra-chr","--het", "--threads",threads,"--out", Het_output], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    with open(het_err, "w") as err:
        err.write(_try.stderr)
    with open(het_out, "w") as out:
        out.write(_try.stdout)
    if _try.stderr:
        print(color_text("WARNING: Plink1.9. Check error log file "+het_err, "red"))
except:
    print(color_text("Error on Plink1.9 execution", "red"))
    print(color_text("Path used for Plink1.9 executable = "+str(plink_path), "red"))
    print(color_text("Error log is stored in "+het_err, "yellow"))
    exit(1)

#Creating the file with all failed individuals

Het_txt = Het_output+".het"

het_fail_output = os.path.join(temp_files, base_name+"_fail-het.txt")

print(color_text("Creating file with all individuals with heterozygosity rate deviating more than 3 sd from the mean", "yellow"))

Het_check(Het_txt, het_fail_output)

print(color_text("File created as "+het_fail_output))

print(color_text("Removing all individuals that failed tthe Heterozygosity analysis", "yellow"))

het_rate_output_file = file_name.replace(".vcf.gz", ".MIND.HET")

het_rate_output = os.path.join(out_dir_path, het_rate_output_file)

het_filter_err = os.path.join(temp_files, "het_filter.err")
het_filter_out = os.path.join(temp_files, "het_filter.out")

try:
    _try = subprocess.run([plink_path,"--bfile",mind_out,"--threads",threads,"--allow-extra-chr","--remove",het_fail_output,"--recode","vcf", "bgz","--out",het_rate_output], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with open(het_filter_err, "w") as err:
        err.write(_try.stderr)
    with open(het_filter_out, "w") as out:
        out.write(_try.stdout)
    if _try.stderr:
        print(color_text("WARNING: Plink1.9. Check error log file "+het_filter_err, "red"))
except:
    print(color_text("Error on Plink1.9 execution", "red"))
    print(color_text("Path used for Plink1.9 executable = "+str(plink_path), "red"))
    print(color_text("Error log is stored in "+het_filter_err, "yellow"))

het_end = time.time()
het_exec = (het_end - het_start)
exec_times.append(["Heterozygosity_Rate", het_exec])


exec_times_df = pd.DataFrame.from_records(exec_times, columns=["Task", "Time"])

exec_out = os.path.join(out_dir_path,base_name+"_MIND_HET_execution_times.tsv")

exec_times_df.to_csv(exec_out, index=False, sep="\t")