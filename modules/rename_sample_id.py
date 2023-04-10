# -- coding: utf-8 --
## Parte da Integração de codigos FOXCONN

## manipulação e atualização de sample ids para caso o usuário deseje fazer esse tipo de alteração

# Bibliotecas necessárias

import pandas as pd
import os
import subprocess
import argparse
import textwrap
import shutil
import time
import sys
import re

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

arg_parser = argparse.ArgumentParser(description = "This script updates the Sample IDs of a VCF file, for this the user must provide a tab or comma separated file with Old sample ID on the first column and the New sample ID in the seconda column"
	, epilog = "Ex. usage: script.py  ") ## AJUSTAR A DESCRIÇÃO

arg_parser.add_argument("-vcf", "--vcf_file", help = "File for processing, requierd for script execution",required=True)
arg_parser.add_argument("-table", "--sample_table", help = "File with OLD_SAMPLE_ID<tab>NEW_SAMPLE_ID",required=True)
arg_parser.add_argument("-bcftools", "--bcftools_path", help = "Path for the bcftools executable, requierd for script execution -- default is to look for the variable on path")
arg_parser.add_argument("-o", "--output_folder", help = "Wanted output folder (default: current output folder)")
arg_parser.add_argument("--threads", help = "Number of computer threads -- default = 1", default="1")

#Se nenhum comando foi dado ao script, automaticamente é mostrado o "help"

if len(sys.argv)==1:
	arg_parser.print_help(sys.stderr)
	sys.exit(0)

def update_header(header_output, new_header_output):
	with open(header_output,'r') as main:
		with open(new_header_output, 'w') as out:
			input_data = main.read()
			for key,value in dict_to_replace.items():
				# input_data = input_data.replace(fr'\b{key}\b',value)
				input_data = re.sub(fr'\b{key}\b',value, input_data)
			out.write(input_data)

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
input_table = args_dict["sample_table"]
threads = args_dict["threads"]

#######################
## Pre-flight checks ##
#######################

# VCF file

file_path = os.path.abspath(vcf_file)

check_file_exists(file_path)

file_name = file_path.split("/")[-1]

base_name = file_name.split(".")[0]

# Table file

table_file_path = os.path.abspath(input_table)

check_file_exists(table_file_path)

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

print(color_text("Using "+threads+" threads"))

if not bcftools_path:
	bcftools_look_for_path = subprocess.run(["which", "bcftools"], stdout=subprocess.PIPE, text=True)

	bcftools_path = bcftools_look_for_path.stdout.strip()

#######################
## Starting analysis ##
#######################

#tempos de execução
exec_times = []

## Get VCF header

vcf_header_start = time.time()

print(color_text("Getting VCF file header"))

header_output = os.path.join(temp_files, base_name+"_OLD_HEADER.txt")

subprocess.run([bcftools_path,"view","--threads",threads,"-h","-o",header_output,vcf_file])

vcf_header_end = time.time()

vcf_header_time = (vcf_header_end - vcf_header_start)

exec_times.append(["Get Header", vcf_header_time])

## Update sample names

print(color_text("Updating Sample IDs"))

update_nsamples_start = time.time()

## Reading table file

input_tsv = pd.read_csv(input_table, dtype="str",sep=None, engine="python")
dict_to_replace = dict(zip(input_tsv.iloc[:,0], input_tsv.iloc[:,1]))

## Function for Sample ID update

new_header_output = os.path.join(temp_files,base_name+"_NEW_HEADER.txt")

update_header(header_output, new_header_output)

update_nsamples_end = time.time()

update_nsamples_time = (update_nsamples_end - update_nsamples_start)

exec_times.append(["Update Sample ID", update_nsamples_time])

## Write updated VCF file

print(color_text("Writing new VCF file"))

update_vcf_start = time.time()

new_id_output_file = file_name.replace(".vcf.gz", ".Nsamples.vcf.gz")

new_id_output_file_path = os.path.join(out_dir_path, new_id_output_file)

subprocess.run([bcftools_path, "reheader","-h",new_header_output,"-o",new_id_output_file_path, vcf_file])

update_vcf_end = time.time()

update_vcf_time = (update_vcf_end - update_vcf_start)

exec_times.append(["Update VCF", update_vcf_time])

#Time stamp output

exec_times_df = pd.DataFrame.from_records(exec_times, columns=["Task", "Time"])

exec_out = os.path.join(out_dir_path,base_name+"_Nsample_execution_times.tsv")

exec_times_df.to_csv(exec_out, index=False, sep="\t")





