## Parte 1 integração de codigos FOXCONN

## Identificação e remoção de SNPs duplicados

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

arg_parser = argparse.ArgumentParser(description = "This is a script to identify and remove duplicated SNPs", 
	epilog = "Ex. usage: script.py  ") ## AJUSTAR A DESCRIÇÃO

arg_parser.add_argument("-vcf", "--vcf_file", help = "File for processing, requierd for script execution",required=True)
arg_parser.add_argument("-bcftools", "--bcftools_path", help = "Path for the bcftools executable, requierd for script execution -- default is to look for the variable on path")
arg_parser.add_argument("-o", "--output_folder", help = "Wanted output folder (default: current output folder)")
arg_parser.add_argument("-plink2", "--plink2_path", help = "Path for the plink2 executable, requierd for script execution -- default is to look for the variable on path")
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

# getting scripts folder path --> pasta com todos os scripts que serão rodados em bash (scripts do José Patane)
script_path = os.path.join(primary_script_path, "scripts")


#Inicializando as variaveis

args = arg_parser.parse_args()
args_dict = vars(arg_parser.parse_args())

vcf_file = args_dict["vcf_file"]
output_folder = args_dict["output_folder"]
bcftools_path = args_dict["bcftools_path"]
plink2_path = args_dict["plink2_path"]
threads = args_dict["threads"]


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

if not bcftools_path:
	bcftools_look_for_path = subprocess.run(["which", "bcftools"], stdout=subprocess.PIPE, text=True)

	bcftools_path = bcftools_look_for_path.stdout.strip()

if not plink2_path:
	plink2_look_for_path = subprocess.run(["which", "plink2"], stdout=subprocess.PIPE, text=True)

	plink2_path = plink2_look_for_path.stdout.strip()


#######################
## Starting analysis ##
#######################

#First step --> Identification of duplicated SNPs --> Aqui usamos $1 = arquivo de entrada e $2 = prefixo (base_name), $3 = output path

print(color_text("Starting identification of duplicated SNPs"))

exec_times = [] #List with execution times

snp_dedup_list_start = time.time()

# desdup_script = os.path.join(script_path,"desdup_snps_list.sh")

snp_list_output = os.path.join(temp_files,"Unique_SNP_list_"+base_name+".txt") #Output do script 1
temp_bim = os.path.join(temp_files,base_name+"_temp_bim")
max_alleles_out_file = os.path.join(out_dir_path, base_name+".unique")
transform_error = os.path.join(temp_files, "plink_bim_convert.err")
transform_out = os.path.join(temp_files, "plink_bim_convert.out")

#Step 1 -- transform to plink bim file

try:
	_try = subprocess.run([plink2_path, "--vcf", vcf_file,"--allow-extra-chr","--make-just-bim", "--out", temp_bim], stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True)
	with open(transform_error, "w") as err:
		err.write(_try.stderr)
	with open(transform_out, "w") as out:
		out.write(_try.stdout)
	if _try.stderr:
		print(color_text("ERROR: Plink2. Check error log file "+multiallele_remove_err, "red"))
		exit(1)

except:
    print(color_text("Error on Plink2 execution", "red"))
    print(color_text("Path used for Plink2 executable = "+str(plink2_path), "red"))
    print(color_text("Error log is stored in "+transform_error, "yellow"))
    exit(1)

#Reading bim

bim_file = pd.read_csv(temp_bim+".bim", sep="\t", header=None, names=["Chr", "rsID_in","Dist_centimorgans_in", "Position", "REF", "ALT"], dtype="str")

total_snps = len(bim_file["rsID_in"])

desduplicate_bim = bim_file.drop(bim_file[bim_file.duplicated(subset=["Chr", "Position"], keep=False)].index)

unique_SNPs = desduplicate_bim["rsID_in"].values.tolist()

snp_dedup_list_end = time.time()

desdup_time = (snp_dedup_list_end-snp_dedup_list_start)

exec_times.append(["Finding all duplicated SNPs", round(desdup_time)])

if len(unique_SNPs) < total_snps:
	with open(snp_list_output, "w") as no_dup_file:
		for row in unique_SNPs:
			no_dup_file.write(row+"\n")

	print(color_text("Starting the removal of the identified duplications and multiallelic", "green"))

	dup_multiallele_err = os.path.join(temp_files, "remove_duplications.err")
	dup_multiallele_out = os.path.join(temp_files, "remove_duplications.out")

	rm_dup_start = time.time()

	try:
		_try = subprocess.run([plink2_path, "--vcf", vcf_file,"--allow-extra-chr","--recode", "vcf", "bgz", "--extract", snp_list_output, "--max-alleles","2","--out", max_alleles_out_file,
			"--threads", threads], stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True)
		with open (dup_multiallele_err, "w") as err:
			err.write(_try.stderr)
		with open (dup_multiallele_out, "w") as out:
			out.write(_try.stdout)

		if _try.stderr:
			print(color_text("ERROR: Plink2. Check error log file "+multiallele_remove_err, "red"))
			exit(1)

	except:
	    print(color_text("Error on Plink2 execution", "red"))
	    print(color_text("Path used for Plink2 executable = "+str(plink2_path), "red"))
	    print(color_text("Error log is stored in "+dup_multiallele_err, "yellow"))
	    exit(1)


	index_err = os.path.join(temp_files, "index.err")
	index_out = os.path.join(temp_files, "index.out")
	try:
		print(color_text("Indexing"))
		_try = subprocess.run([bcftools_path,"index","--threads",threads,"-f","-t",max_alleles_out_file+".vcf.gz"],text=True)
		# with open(index_err, "w") as err:
		# 	err.wrtie(_try.stderr)
		# 	,stderr=subprocess.PIPE
		
	except:
	    print(color_text("Error on bcftools execution", "red"))
	    print(color_text("Path used for bcftools executable = "+str(bcftools_path), "red"))
	    exit(1)

if len(unique_SNPs) == total_snps:

	multi_var_start = time.time()

	print(color_text("No duplicated SNPs found! Removing multiallelic variants only", "yellow"))

	multiallele_remove_err = os.path.join(temp_files, "multiallelic_remove.err")
	multiallele_remove_out = os.path.join(temp_files, "multiallelic_remove.out")

	try:
		_try = subprocess.run([plink2_path, "--vcf", vcf_file,"--allow-extra-chr","--recode", "vcf", "bgz", "--max-alleles","2","--out", max_alleles_out_file,
			"--threads", threads],stdout=subprocess.PIPE,stderr=subprocess.PIPE,  text=True)
		with open(multiallele_remove_err, "w") as err:
			err.write(_try.stderr)
		with open(multiallele_remove_out, "w") as out:
			out.write(_try.stdout)
		if _try.stderr:
			print(color_text("ERROR: Plink2. Check error log file "+multiallele_remove_err, "red"))
			exit(1)

	except:
	    print(color_text("Error on Plink2 execution", "red"))
	    print(color_text("Path used for Plink2 executable = "+str(plink2_path), "red"))
	    print(color_text("Error log is stored in "+multiallele_remove_err, "yellow"))
	    exit(1)

	index_err = os.path.join(temp_files, "index.err")
	index_out = os.path.join(temp_files, "index.out")
	try:
		print(color_text("Indexing"))
		_try = subprocess.run([bcftools_path,"index","--threads",threads,"-f","-t",max_alleles_out_file+".vcf.gz"], text=True)
		# with open(index_err, "w") as err:
		# 	err.write(_try.stderr)
		# 	 stderr=subprocess.PIPE,
	except:
	    print(color_text("Error on bcftools execution", "red"))
	    print(color_text("Path used for bcftools executable = "+str(bcftools_path), "red"))
	    exit(1)

	multi_var_end = time.time()

	muti_var_time = (multi_var_end - multi_var_start)

	exec_times.append(["Multiallelic removal", muti_var_time])
	

exec_times_df = pd.DataFrame.from_records(exec_times, columns=["Task", "Time"])

exec_out = os.path.join(out_dir_path,base_name+"_Desdup_execution_times.tsv")

exec_times_df.to_csv(exec_out, index=False, sep="\t")


