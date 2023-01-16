## Integração de codigos FOXCONN

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

#LD Pred usa os binarios do Plink (bed, bim, fam) para rodar as análises, além do arquivo de saida do GWAS -- Damos duas opções 
#1 vcf --> gerar a partir do VCF os binários
#2 bfile --> usuário já fornecer os binários

##Usuário deve fornecer o N de individuos que passaram pelo GWAS -- lembrando que para UKBB vamos usar 2/3

##Para implementar -- flag para covars, qcovars e pheno / Plots / Summary output!

arg_parser.add_argument("-vcf", "--vcf_file", help = "File for PRS analysis, required if user dont have Plink binary files (Same file as used for GWAS)", default=None)
arg_parser.add_argument("-plink", "--plink_path", help = "Path for the plink(1.9) executable -- default is to look for the variable on path")
arg_parser.add_argument("-plink2", "--plink2_path", help = "Path for the Plink2 executable, requierd for script execution -- default is to look for the variable on path")
arg_parser.add_argument("-bfile", "--plink_binary_prefix", help = "Path for the plink(1.9) binary file, provide only the prefix (no extensions) -- Same used in the GWAS setp")
arg_parser.add_argument("-mlma", "--GWAS_mlma", help = "Output file from de GWAS step -- the extension of this file is .mlma for GCTA and .stats for BOLT-LMM", required=True)
arg_parser.add_argument("--BOLT", help = "Use this flag if the BOLT-LMM output (.stats) was provided", action="store_true")
arg_parser.add_argument("-pheno", "--phenotype_file", help = "Path for the phenotype file, this file must have FID and IID (like the .fam file) and must be separated by tab or space. Same used on the GWAS setp")
arg_parser.add_argument("--pheno_col", help = "Name of the columns contaning the Phenotype data -- Default is to look for 'Phenotype' as the column name", default="Phenotype")
arg_parser.add_argument("-qcovar", "--quantitative_covar", help = "Path for the quantitative covariables, e.g. PCs, age, and other continuous variables. The same used on the GWAS step")
arg_parser.add_argument("-n_pcs", "--number_of_pcs", help = "Number of PCs to use on model evaluation -- default = 4", default="4")
arg_parser.add_argument("-covar", "--covar_file", help = "Path for the covariables file, e.g. Sex. The same used on the GWAS step")
arg_parser.add_argument("-o", "--output_folder", help = "Wanted output folder (default: current output folder)")
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
plink2_path = os.path.join(script_path, "plink2")
threads = args_dict["threads"]
bfile = args_dict["plink_binary_prefix"]
mlma = args_dict["GWAS_mlma"]
pheno = args_dict["phenotype_file"]
qcovar = args_dict["quantitative_covar"]
covar = args_dict["covar_file"]
n_pcs = args_dict["number_of_pcs"]
bolt = args_dict["BOLT"]
pheno_col = args_dict["pheno_col"]

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

if not plink_path:
	plink_look_path = subprocess.run(["which", "plink1.9"], stdout=subprocess.PIPE, text=True)
	plink_path = plink_look_path.stdout.strip()

#######################
## Starting analysis ##
#######################

#tempos de execução
exec_times = []

##PRIMEIRA COISA -- Checar se os SNPs do HapMap3 (ou outra ref) batem com os dados de entrada --> usar extract do plink

#Check SNPs

convert_snps_start = time.time()

#Se o usuário fornecer um VCF

if vcf_file:

	print(color_text("Starting convertion to bfile format for LD Pred2", "yellow"))

	plink_out = os.path.join(temp_files, base_name+"_for_LDPRED")

	plink_convert_err = os.path.join(temp_files, "plink_convert.err")
	plink_convert_out = os.path.join(temp_files, "plink_convert.out")

	try:
		_try = subprocess.run([plink_path,"--vcf", vcf_file, "--pheno", pheno,"--make-bed", "--out",plink_out, "--threads", threads], stdout=subprocess.PIPE,
			stderr=subprocess.PIPE, text=True)
		with open(plink_convert_err, "w") as err:
			err.write(_try.stderr)
		with open(plink_convert_out, "w") as out:
			out.write(_try.stdout)
		if _try.stderr:
			print(color_text("WARNING: Plink1.9. Check error log file "+plink_convert_err, "yellow"))
	except:
		print(color_text("Error on Plink1.9 execution", "red"))
		print(color_text("Path used for Plink1.9 executable = "+str(plink_path), "red"))
		print(color_text("Error log is stored in "+plink_convert_err, "yellow"))
		exit(1)

	print(color_text("Done with VCF convertion", "yellow"))


#Se o usuário fornecer um bfile pronto

if bfile:
	
	plink_out = bfile


print(color_text("Starting the analysis with file"+plink_out, "yellow"))



convert_snps_end = time.time()

convert_snps_time = (convert_snps_end - convert_snps_start)

exec_times.append(["LDPRED SNP convertion", convert_snps_time])

#Before PRS ajust mlma file --> order: rsid,chr,pos,a0,a1,beta,beta_se,N,p
#N = number of samples -- Can find that info on the .fam file

print(color_text("Adjusting mlma table", "yellow"))

adjust_start = time.time()

#Reading mlma file

mlma_to_adjust = pd.read_csv(mlma, sep="\t")

#Getting n of samples from .fam

fam_file = pd.read_csv(plink_out+".fam",sep=" ", header=None)

n_of_indiv = len(fam_file[0])

#Adjusting the table



if bolt == True:
	try:
		adjusted = mlma_to_adjust.drop(columns=["A1FREQ", "GENPOS", "F_MISS"])
		adjusted = adjusted.rename(columns={"CHR": "chr", "SNP" : "rsid", "BP" : "pos", "ALLELE0" : "a0", "ALLELE1" : "a1", "BETA": "beta",
	"SE" : "beta_se"})
	except:
		print(color_text("ERROR: Wrong columns names in input file "+str(mlma), "red"))
		exit(1)

else:

	try:
		adjusted = mlma_to_adjust.drop(columns="Freq")
		adjusted = adjusted.rename(columns={"Chr": "chr", "SNP" : "rsid", "bp" : "pos", "A1" : "a0", "A2" : "a1", "b": "beta",
		"se" : "beta_se"})
	except:
		print(color_text("ERROR: Wrong columns names in input file "+str(mlma), "red"))
		print(color_text("If the file provided is from BOLT-LMM output, pĺease use the flag '--BOLT'", "yellow"))
		exit(1)


adjusted["n_eff"] = n_of_indiv

#Putting in the right order

adjusted = adjusted[["rsid","chr","pos","a0","a1","beta","beta_se", "n_eff", "p"]]

print(color_text("Writing adjusted mlma file", "yellow"))

adjusted_file = mlma.split("/")[-1]

if bolt == True:
	adjusted_file = adjusted_file.replace(".stats", ".adjusted.stats")

else:
	adjusted_file = adjusted_file.replace(".mlma", ".adjusted.mlma")

adjusted_file_output = os.path.join(temp_files, adjusted_file)

adjusted.to_csv(adjusted_file_output, index=False)

adjust_end = time.time()

adjust_time = (adjust_end - adjust_start)

exec_times.append(["Adjust of mlma file", adjust_time])

#Start PRS

prs_start = time.time()

LDPRED_Rscript = os.path.join(script_path, "LDPred2.R")

bfile_input = plink_out+".bed"

mlma_input = adjusted_file_output

database = os.path.join(database_path, "LDPRED2")

hapmap3_ld_matrix = os.path.join(database, "map_hm3_ldpred2.rds")

out_put_LDpred = out_dir_path

t = threads

print(color_text("Starting LDPred2"))

ldpred_err = os.path.join(temp_files, "LDPred.err")

try:
	_try = subprocess.run(["Rscript", LDPRED_Rscript, "--bfile", bfile_input, "--mlma", mlma_input, "--database", hapmap3_ld_matrix, "--outdir", out_put_LDpred, "--threads", t, "--pheno", pheno, 
	"--covar", covar, "--qcovar", qcovar, "--PCs", n_pcs], stderr=subprocess.PIPE, text=True)
	with open(ldpred_err, "w") as err:
		err.write(_try.stderr)
except:
	print(color_text("ERROR: LDPred Rscript. Check the error log file "+str(ldpred_err), "red"))



## Remover arquivos *.interpolated_genetic_map

for file in os.listdir(out_put_LDpred):
	if file.endswith(".interpolated_genetic_map") or file.endswith(".sbk") or file.endswith(".bk"):
		to_remove = os.path.join(out_put_LDpred, file)
		try:
			os.remove(to_remove)
		except:
			print("File already removed")


#PRS calculation using Plink

print(color_text("Starting PRS calculation from the LDpred2 weights"))

in_bfile = plink_out
in_weight_scores = os.path.join(out_put_LDpred, "weights_LDPRED2.tsv")

output_PRS = os.path.join(out_dir_path, "PRS_scores")

PRS_err = os.path.join(temp_files, "PRS_score.err")
PRS_out = os.path.join(temp_files, "PRS_score.out")

print(color_text("Calculating allele frequencies"))
## ADD ESSA PARTE!!
#plink2 --bfile /home/fernando/lgcm/projects/Pipeline_FOXCONN/dev/scripts/testes_pipe/tmp/GARSA_final_example_for_LDPRED --out /home/fernando/lgcm/projects/Pipeline_FOXCONN/dev/scripts/testes_pipe/tmp/TESTE_LDPRED2_freqs --freq
freq_out_plink = os.path.join(temp_files, "LDPred_frequencies") #+.afreq

freq_err = os.path.join(temp_files, "freq.err")
freq_out = os.path.join(temp_files, "freq.out")

_try = subprocess.run([plink2_path, "--bfile", in_bfile, "--out", freq_out_plink, "--freq", "--threads", threads], 
stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

with open(freq_err, "w") as err:
	err.write(_try.stderr)
with open(freq_out, "w") as out:
	out.write(_try.stdout)

if _try.stderr:
	print(color_text("WARNING: Plink2. Check error log file "+freq_err, "yellow"))

try:
	_try = subprocess.run([plink2_path, "--bfile", in_bfile, "--out", output_PRS, "--score", in_weight_scores, "1", "2", "3", "header-read", "cols=scoresums", 
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

prs_end = time.time()

prs_time = (prs_end - prs_start)

exec_times.append(["PRS", prs_time])

#PLOTS

#O output do PRS sai como output_PRS+".sscore"

print(color_text("Plotting boxplot"))

prs_data = pd.read_csv(output_PRS+".sscore", names=["IID", "PRS"], header=None, sep="\t", skiprows=1)

pheno_data = pd.read_csv(pheno, sep=None, engine="python")

covars = pd.read_csv(covar, sep=None, engine="python")

qcovars = pd.read_csv(qcovar, sep=None, engine="python")

#1 merge
pheno_total = pd.merge(pheno_data, covars, on=["FID", "IID"], how="inner")
#2 merge
pheno_total = pd.merge(pheno_total, qcovars, on=["FID", "IID"], how="inner")

#3 Join PRS scores and phenotype data

prs_pheno = pd.merge(prs_data, pheno_total, on="IID", how="inner")

#Calculating PRS deciles

prs_pheno["Deciles"] = pd.qcut(prs_pheno["PRS"], 10, lables=False)

#Saving decile data
prs_deciles_out = os.path.join(out_dir_path, "PRS_deciles_distribution.tsv")
prs_pheno.to_csv(prs_deciles_out, sep="\t", index=False)

#Plotting boxplot
boxplot_png = os.path.join(out_dir_path, "PRS_boxplot.png")
boxplot_svg = os.path.join(out_dir_path, "PRS_boxplot.svg")

fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(x="Deciles", y=pheno_col, data=prs_pheno, ax=ax, color="lightblue")
sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
ax.set_xlabel("Deciles", fontsize=16)
ax.set_ylabel("Phenotype distribution", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.savefig(boxplot_png, dpi=300, facecolor="white")
plt.savefig(boxplot_svg, facecolor="white")

#Plot distribution
print(color_text("Plotting distribution"))

dist_plot_png = os.path.join(out_dir_path, "PRS_distribution_plot.png")
dist_plot_svg = os.path.join(out_dir_path, "PRS_distribution_plot.svg")

sns.displot(prs_pheno, x="PRS", kind="kde", fill=True, pallete="pastel")

plt.savefig(dist_plot_png, dpi=300, facecolor="white")
plt.savefig(dist_plot_svg, facecolor="white")

#Salvando os tempos

exec_times_df = pd.DataFrame.from_records(exec_times, columns=["Task", "Time"])

exec_out = os.path.join(out_dir_path,base_name+"_PRS_execution_times.tsv")

exec_times_df.to_csv(exec_out, index=False, sep="\t")