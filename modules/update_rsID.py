## Parte 2 Integração de codigos FOXCONN

## Anotação de SNPs usando rsID

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
from Bio.Seq import Seq

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

arg_parser = argparse.ArgumentParser(description = "This is a script to update SNP rsIDs (for hg19). This script assumes that your file name have the pattern chr[1-22], e.g project_name_chr12.extensions", 
	epilog = "Ex. usage: script.py  ") ## AJUSTAR A DESCRIÇÃO

arg_parser.add_argument("-vcf", "--vcf_file", help = "File for processing, requierd for script execution")
arg_parser.add_argument("-ref_hg", "--ref_build", help = "Select the human genome build version -- hg37 or hg38, default=hg37", default="hg37")
arg_parser.add_argument("-bcftools", "--bcftools_path", help = "Path for the bcftools executable, requierd for script execution -- default is to look for the variable on path")
arg_parser.add_argument("-plink2", "--plink2_path", help = "Path for the Plink2 executable, requierd for script execution -- default is to look for the variable on path")
arg_parser.add_argument("-plink", "--plink_path", help = "Path for the Plink1.9 executable, requierd for script execution -- default is to look for the variable on path")
arg_parser.add_argument("-o", "--output_folder", help = "Wanted output folder (default: current output folder)")
arg_parser.add_argument("-rm_tmp", "--rm_temp_files", help = "Force keeping temporary files (Files may be quite large) -- default: Delete temporary files", action="store_true")
arg_parser.add_argument("--threads", help = "Number of computer threads -- default = 1", default=1)

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

# fastas_path = os.path.join(database_path, "hg_19_fasta")

# setting scripts path (assuming they are in the same directory as the primary script)
script_path = os.path.join(primary_script_path, "scripts")


args = arg_parser.parse_args()
args_dict = vars(arg_parser.parse_args())

vcf_file = args_dict["vcf_file"]
hg_build = args_dict["ref_build"]
output_folder = args_dict["output_folder"]
bcftools_path = args_dict["bcftools_path"]
plink2_path = os.path.join(script_path, "plink2")
plink_path = args_dict["plink2_path"]
remove_temp = args_dict["rm_temp_files"]
threads = args_dict["threads"]

#######################
## Pre-flight checks ##
#######################

file_path = os.path.abspath(vcf_file)

check_file_exists(file_path)

file_name = file_path.split("/")[-1]

base_name = file_name.split(".")[0] #Equivalente ao $G do script!

# for_regex_name = base_name.lower()

# chr_number = re.findall("chr[0-9]*", for_regex_name)[0]

print("Working on ",base_name)

# Se for dada uma path para output
if output_folder:

    provided_output = os.path.abspath(output_folder)

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


#Verificando se os arquivos de database existem!

if hg_build == "hg37":
	database_file_1 = os.path.join(database_path, "dbsnp_hg37p13_b151", "hg37_b151_dbSNP.vcf.gz")
	databse_file_2 = os.path.join(database_path, "dbsnp_hg37p13_b151", "hg37_b151_dbSNP.vcf.gz.tbi")

if hg_build == "hg38":
	database_file_1 = os.path.join(database_path, "dbsnp_hg38p7_b151", "hg38_b151_dbSNP.vcf.gz")
	databse_file_2 = os.path.join(database_path, "dbsnp_hg38p7_b151", "hg38_b151_dbSNP.vcf.gz.tbi")

# database_file_1 = os.path.join(database_path, "HRC.r1-1.GRCh37.wgs.mac5.sites.semDuplicatas.vcf.gz")
# databse_file_2 = os.path.join(database_path, "HRC.r1-1.GRCh37.wgs.mac5.sites.semDuplicatas.vcf.gz.tbi")

if not os.path.exists(database_file_1):
	print(color_text("[WARNING] Required annotation file"+database_file_1+" not found", "red"))
	print(color_text("Download it using the download_database.py script", "yellow"))
	# print(color_text("Reamember to download the index (.tbi) file aswell", "yellow"))

else:
	print(color_text("Database files found!"))

if not bcftools_path:
    bcftools_look_for_path = subprocess.run(["which", "bcftools"], stdout=subprocess.PIPE, text=True)

    bcftools_path = bcftools_look_for_path.stdout.strip()
if not plink_path:
	plink_look_path = subprocess.run(["which", "plink1.9"], stdout=subprocess.PIPE, text=True)
	plink_path = plink_look_path.stdout.strip()


#######################
## Starting analysis ##
#######################
#tempos de execução
exec_times = []

#Primeira coisa, corrigir possiveis erros do arquivo em relação a referencia -- vou passar a usar o dbsnp

correct_flip_start = time.time()

correct_output_file = file_name.replace(".vcf.gz", ".CORRECTED.vcf.gz")

correct_output = os.path.join(out_dir_path, correct_output_file)

#1 Step - Compare vcf input file and reference for reference filtering

temp_filtered_file = os.path.join(temp_files, "filtered_temp_ref_tab.vcf.gz")

temp_filetered_vcf_file = os.path.join(temp_files, "filtered_temp_ref.vcf.gz")

print(color_text("Filtering snps from database"))


subprocess.run([bcftools_path, "view", "-R", vcf_file, database_file_1, "-Oz", "-o", temp_filtered_file, "--no-header", "--threads", threads])

print(color_text("Table creation done"))

subprocess.run([bcftools_path, "view", "-R", vcf_file, database_file_1, "-Oz", "-o", temp_filetered_vcf_file, "--threads", threads])

subprocess.run([bcftools_path, "index", "-t", temp_filetered_vcf_file, "--threads", threads])

print(color_text("databse VCF creation done"))

#2 Transform user input to bim file

print(color_text("Creating Plink .bim file for annotation step"))

temp_user_bim = os.path.join(temp_files, "temp_bim_to_annot")

form_bim_err = os.path.join(temp_files, "form_bim.err")
form_bim_out = os.path.join(temp_files, "form_bim.out")

try:
	_try = subprocess.run([plink2_path, "--vcf", vcf_file,"--id-delim", "_","--allow-extra-chr", "--make-just-bim", "--keep-allele-order","--out", temp_user_bim], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	with open(form_bim_err, "w") as err:
		err.write(_try.stderr)
	with open(form_bim_out, "w") as out:
		out.write(_try.stdout)
	if _try.stderr:
		print(color_text("WARNING: Plink2. Check error log file "+form_bim_err, "red"))

except:
    print(color_text("Error on Plink2 execution", "red"))
    print(color_text("Path used for Plink2 executable = "+str(plink2_path), "red"))
    print(color_text("Error log is stored in "+form_bim_err, "yellow"))
    exit(1)

print(color_text("Looking for swapped and flipped snps", "yellow"))

#1 Swapped

#get bim file

user_bim = pd.read_csv(temp_user_bim+".bim", sep="\t", header=None, names=["Chr", "rsID_in","Dist_centimorgans_in", "Position", "REF", "ALT"], dtype="str")

#Get ref vcf

ref_vcf = pd.read_csv(temp_filtered_file, sep="\t", header=None, names=["Chr", "Position", "rsID", "REF", "ALT", "coisa", "coisa1", "INFO"], dtype="str")
ref_vcf_explode = ref_vcf.assign(ALT=ref_vcf["ALT"].str.split(",")).explode("ALT")
ref_vcf_explode

#Merge both datasets for analysis

intersect_data = pd.merge(user_bim, ref_vcf_explode, on=["Chr", "Position"], how="inner")

#Checking for swapps

intersect_data["swapped"] = intersect_data.apply(lambda x: "TRUE" if x["REF_x"] == x["ALT_y"] and x["ALT_x"] == x["REF_y"] else "FALSE", axis=1)

swapped_count = intersect_data["swapped"].values.tolist().count("TRUE")

if swapped_count != 0:

    #Criar arquivo com os SNPs para swap
    input_swap = vcf_file
    swap_file = os.path.join(temp_files, "ref_allele_swap.txt")
    swapped_out = os.path.join(temp_files, base_name+"_swapped")

    swap_err = os.path.join(temp_files, "swapped_snps.err")
    swap_out = os.path.join(temp_files, "swapped_snps.out")

    print(color_text(str(swapped_count)+" swapped SNPs found", "yellow"))

    to_swap = intersect_data[intersect_data["swapped"] == "TRUE"]
    to_swap = to_swap[["ALT_y", "rsID_in"]]
    to_swap = to_swap.drop_duplicates(subset="rsID_in")
    to_swap.to_csv(swap_file, sep="\t", index=False, header=False)

    #Rodar correção pelo plink
    try:
    	_try = subprocess.run([plink2_path, "--vcf", input_swap, "--keep-allele-order","--id-delim","_","--allow-extra-chr","--alt1-allele", "force", swap_file, "1", "2", "--make-bed", "--out", swapped_out, "--threads", threads],stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    	with open(swap_err, "w") as err:
    		err.write(_try.stderr)
    	with open(swap_out, "w") as out:
    		out.write(_try.stdout)
    	if _try.stderr:
    		print(color_text("WARNING: Plink2. Check error log file "+swap_err, "red"))
    except:
    	print(color_text("Error on Plink2 execution", "red"))
    	print(color_text("Path used for Plink2 executable = "+str(plink2_path), "red"))
    	print(color_text("Error log is stored in "+swap_err, "yellow"))
    	exit(1)

if swapped_count == 0:
    print(color_text("No swapped SNPs found, skipping correction"))
    swapped_out = vcf_file


#Checking for Flipps

intersect_data["flipped"] = intersect_data.apply(lambda x: "TRUE" if Seq(str(x["REF_x"]+x["ALT_x"])).complement() == str(x["REF_y"]+x["ALT_y"]) else "FALSE", axis=1)

flipped_count = intersect_data["flipped"].values.tolist().count("TRUE")

if flipped_count != 0:

    print(color_text(str(flipped_count)+" flipped SNPs found", "yellow"))

    # Criar arquivo com os SNPs para flip
    flip_file = os.path.join(temp_files, "flip_allele.txt")
    flipped_out = os.path.join(temp_files, base_name+"_flipped")

    flip_err = os.path.join(temp_files, "flipped_snps.err")
    flip_out = os.path.join(temp_files, "flipped_snps.out")

    to_flip = intersect_data[intersect_data["flipped"] == "TRUE"]
    to_flip = to_flip[["rsID_in"]]
    to_flip.to_csv(flip_file, sep="\t", index=False, header=False)

    #Corrgir usando plink

    try:
    	_try = subprocess.run([plink_path, "--bfile", swapped_out, "--flip", flip_file, "--allow-extra-chr","--recode", "vcf", "bgz", "--out", flipped_out,"--threads", threads],
    		stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    	with open(flip_err, "w") as err:
    		err.write(_try.stderr)
    	with open(flip_out, "w") as out:
    		out.write(_try.stdout)
    	if _try.stderr:
    		print(color_text("WARNING: Plink2. Check error log file "+flip_err, "red"))
    except:
    	print(color_text("Error on Plink2 execution", "red"))
    	print(color_text("Path used for Plink2 executable = "+str(plink2_path), "red"))
    	print(color_text("Error log is stored in "+flip_err, "yellow"))
    	exit(1)


else:
    print(color_text("No flipped SNPs found, skipping correction"))
    try:
    	_try = subprocess.run([plink2_path, "--bfile", swapped_out,"--recode", "vcf", "bgz", "--out", flipped_out,"--threads", threads],
    		stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    	with open(flip_err, "w") as err:
    		err.write(_try.stderr)
    	with open(flip_out, "w") as out:
    		out.write(_try.stdout)
    	if _try.stderr:
    		print(color_text("WARNING: Plink2. Check error log file "+flip_err, "red"))
    except:
    	print(color_text("Error on Plink2 execution", "red"))
    	print(color_text("Path used for Plink2 executable = "+str(plink2_path), "red"))
    	print(color_text("Error log is stored in "+flip_err, "yellow"))
    	exit(1)




print(color_text("Flip and swap corretion done"))

correct_flip_end = time.time()

correct_flip_time = (correct_flip_end - correct_flip_start)

exec_times.append(["Flip Correction", correct_flip_time])

indexing_start = time.time()

#Indexando o arquivo comprimido
print(color_text("Starting index"))

subprocess.run([bcftools_path,"index","-f","--threads",threads,"-t",flipped_out+".vcf.gz"])

indexing_end = time.time()

indexing_time = (indexing_end - indexing_start)

exec_times.append(["Indexing", indexing_time])

#Step 3 --> Annotation of rsIDs

#arquivo para anotação é o VCF da database = database_file_1

#Annotation
annotation_start = time.time()

print(color_text("Starting annotation"))
annot_output_file = file_name.replace(".vcf.gz", ".RSIDs.vcf.gz")
annot_output = os.path.join(out_dir_path,annot_output_file)
subprocess.run([bcftools_path,"annotate","--threads",threads,"-a",temp_filetered_vcf_file, "-c","CHROM,POS,ID", "-Oz","-o",annot_output,flipped_out+".vcf.gz"])

print(color_text("Indexing compressed file"))
subprocess.run([bcftools_path,"index","-f","--threads",threads,"-t",annot_output])

annotation_end = time.time()

annotation_time = (annotation_end - annotation_start)

exec_times.append(["Annotation",annotation_time])

#remove temp files if user specify

# if remove_temp == True:
# 	print(color_text("Keeping temporary files"))
# else:
# 	#primeiro remover o vcf não comprimido gerado
# 	print(color_text("Removing uncompressed file "+annot_output_file))
# 	os.remove(annot_output)
# 	#segundo remover os _temp do awk
# 	print(color_text("Removing other temporary files"))
# 	os.remove(sort_output)
# 	os.remove(correct_output)


exec_times_df = pd.DataFrame.from_records(exec_times, columns=["Task", "Time"])

exec_out = os.path.join(out_dir_path,base_name+"_RSIDs_execution_times.tsv")

exec_times_df.to_csv(exec_out, index=False, sep="\t")
