# -- coding: utf-8 --
## Integracao de codigos FOXCONN

## PRS usando LD Pred2

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
import seaborn as sns

#Script for PRS extrapolation

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

arg_parser = argparse.ArgumentParser(description = "This is a script to GWAS analysis and plot the results with Manhattam plot", 
	epilog = "Ex. usage: script.py  ") ## AJUSTAR A DESCRIÇÃO

## A partir de VCFs de pops independentes extrapolar os betas obtidos e calcular um PRS para uma população externa às etapas de teste e validação

arg_parser.add_argument("-vcf", "--vcf_file", help = "File for PRS analysis, required if user dont have Plink binary files (Same file as used for GWAS)", default=None)
arg_parser.add_argument("-plink2", "--plink2_path", help = "Path for the Plink2 executable, requierd for script execution -- default is to look for the variable on path")
arg_parser.add_argument("-weight", "--LDpred_weights", help = "Path for the weights_LDPRED2.tsv file generated at the PRS step", required=True)
arg_parser.add_argument("-o", "--output_folder", help = "Wanted output folder (default: current output folder)")
arg_parser.add_argument("-bfile", "--plink_binary_prefix", help = "Path for the plink binary file, provide only the prefix (no extensions)")
arg_parser.add_argument("--colsum", help = "Choose only the sum of PRS to be reported", action="store_true")
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
plink2_path = os.path.join(script_path, "plink2")
threads = args_dict["threads"]
LD_pred_weight = args_dict["LDpred_weights"]
bfile = args_dict["plink_binary_prefix"]
colsum = args_dict["colsum"]

#######################
## Pre-flight checks ##
#######################
if vcf_file:

	file_path = os.path.abspath(vcf_file)

	check_file_exists(file_path)

	file_name = file_path.split("/")[-1]

	base_name = file_name.split(".")[0]

	print("Working on ",base_name)

if bfile:
	file_path = os.path.abspath(bfile)

	file_name = file_path.split("/")[-1]

	base_name = file_name

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

print(color_text("Using "+str(threads)+" Threads"))

## VCF to Bfile Plink
if vcf_file:
    print(color_text("Converting VCF to Plink file"))

    in_bfile = os.path.join(temp_files, "extrapolation_pop")

    subprocess.run([plink2_path, "--vcf", file_path, "--max-alleles", "2", "--rm-dup","exclude-all","--make-bed", "--out", in_bfile, "--threads", threads])

if bfile:
	in_bfile = bfile

print(color_text(f"Calculating allele frequencies from {base_name} dataset"))
## ADD ESSA PARTE!!
#plink2 --bfile /home/fernando/lgcm/projects/Pipeline_FOXCONN/dev/scripts/testes_pipe/tmp/GARSA_final_example_for_LDPRED --out /home/fernando/lgcm/projects/Pipeline_FOXCONN/dev/scripts/testes_pipe/tmp/TESTE_LDPRED2_freqs --freq
freq_out_plink = os.path.join(temp_files, "extrapolation_pop_frequencies") #+.afreq

freq_err = os.path.join(temp_files, f"{base_name}_extrapolation_pop_freq.err")
freq_out = os.path.join(temp_files, f"{base_name}_extrapolation_pop_freq.out")

_try = subprocess.run([plink2_path, "--bfile", in_bfile, "--out", freq_out_plink, "--freq", "--threads", threads], 
stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

with open(freq_err, "w") as err:
	err.write(_try.stderr)
with open(freq_out, "w") as out:
	out.write(_try.stdout)

if _try.stderr:
	print(color_text("WARNING: Plink2. Check error log file "+freq_err, "yellow"))

#starting extrapolation of betas 

print(color_text("Extrapolation for PRS calcualtion in new cohort"))

PRS_err = os.path.join(temp_files, f"{base_name}_Extrapolation_score.err")
PRS_out = os.path.join(temp_files, f"{base_name}_Extrapolation_score.out")

output_PRS = os.path.join(out_dir_path, f"{base_name}_Extrapolation_scores")

if colsum:

	try:
		_try = subprocess.run([plink2_path, "--bfile", in_bfile, "--out", output_PRS, "--score", LD_pred_weight, "1", "2", "3", "header-read", "cols=scoresums", 
		"--threads", threads, "--read-freq", freq_out_plink+".afreq"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		with open(PRS_err, "w") as err:
			err.write(_try.stderr)
		with open(PRS_out, "w") as out:
			out.write(_try.stdout)
		if _try.stderr:
			print(color_text("WARNING: Plink2. Check error log file "+PRS_err, "yellow"))

	except:
		print(color_text("Error on Plink2 execution", "red"))
		print(color_text("Path used for Plink2 executable = "+str(plink2_path), "red"))
		print(color_text("Error log is stored in "+PRS_err, "yellow"))
		exit(1)

if not colsum:
	try:
		_try = subprocess.run([plink2_path, "--bfile", in_bfile, "--out", output_PRS, "--score", LD_pred_weight, "1", "2", "3", "header-read",
						"--threads", threads, "--read-freq", freq_out_plink+".afreq"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		with open(PRS_err, "w") as err:
			err.write(_try.stderr)
		with open(PRS_out, "w") as out:
			out.write(_try.stdout)
		if _try.stderr:
			print(color_text("WARNING: Plink2. Check error log file "+PRS_err, "yellow"))

	except:
		print(color_text("Error on Plink2 execution", "red"))
		print(color_text("Path used for Plink2 executable = "+str(plink2_path), "red"))
		print(color_text("Error log is stored in "+PRS_err, "yellow"))
		exit(1)

