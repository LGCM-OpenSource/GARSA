# -- coding: utf-8 --
## Processo de PCA 1 (não aparentados) e PCA 2 (extrapolação para aparentados)

# Bibliotecas necessárias

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import numpy as np
import sys
import argparse
import textwrap
import shutil
import time
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


arg_parser = argparse.ArgumentParser(description = "This script runs PCA for non-related individualas and projects to related individuals", 
    epilog = "Ex. usage: script.py  ") ## AJUSTAR A DESCRIÇÃO

arg_parser.add_argument("-vcf", "--vcf_file", help = "File for processing, requierd for script execution",required=True)
arg_parser.add_argument("-plink", "--plink_path", help = "Path for the plink(1.9) executable, requierd for script execution -- default is to look for the variable on path")
arg_parser.add_argument("-o", "--output_folder", help = "Wanted output folder (default: current output folder)")
arg_parser.add_argument("-related", "--related_file", help = "File from the kinship module with all related individuals", required=True)
arg_parser.add_argument("--window_size", help = "Window size for prunning step -- default = 1000", default="1000")
arg_parser.add_argument("--sliding_window_step", help = "Sliding Window step -- default = 50", default="50")
arg_parser.add_argument("--prune_r2", help = "R2 value for prunning-- default = 0.03", default="0.03")
arg_parser.add_argument("--Hg", help = "Hg version for use -- 38 or 37. Default=37", default="37")
# arg_parser.add_argument("--N_pop", help = "Number of expected populations in sample, if provided no testing for best K is performed -- The default is for automatic look up using admixture")
arg_parser.add_argument("--threads", help = "Number of computer threads -- default = 1", default="1")
# arg_parser.add_argument("--garsa_path", help = "Path to main script GARSA -- always provided by default")


#Se nenhum comando foi dado ao script, automaticamente é mostrado o "help"

if len(sys.argv)==1:
    arg_parser.print_help(sys.stderr)
    sys.exit(0)


#For the admixture problem, use sys.argv[-1] to get the GARSA.py main path

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
plink_path = args_dict["plink_path"]
related_file = args_dict["related_file"]
admixture_path = os.path.join(script_path, "admixture")
flashPCA_path = os.path.join(script_path, "flashpca")
win_size = args_dict["window_size"]
step_size = args_dict["sliding_window_step"]
prune_r2 = args_dict["prune_r2"]
threads = args_dict["threads"]
output_folder = args_dict["output_folder"]
hg = args_dict["Hg"]
# GARSA_path = args_dict["garsa_path"]
GARSA_path = os.getcwd()
# N_pop = args_dict["N_pop"]

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

print(color_text("Using window size of"+str(win_size)))

#Sliding window step

print(color_text("Using step size of"+str(step_size)))

# R2 threshold

print(color_text("Using R2 threshold of"+str(prune_r2)))

#Threads 
print(color_text("Using "+threads+" threads"))


if not plink_path:
    plink_look_path = subprocess.run(["which", "plink1.9"], stdout=subprocess.PIPE, text=True)
    plink_path = plink_look_path.stdout.strip()


#######################
## Starting analysis ##
#######################

#tempos de execução
exec_times = []

#Primeira etapa --> Remover LD long range e fazer Prunning

print(color_text("Starting Prunning LD short range and LD long-range"))

#1 Prunning
## Prunning short range

prunning_start = time.time()

print(color_text("Starting prunning step"))

indep_tmp_file = os.path.join(temp_files, base_name+"_tmp")

indep_err = os.path.join(temp_files, "indep_PCA.err")
indep_out = os.path.join(temp_files, "indep_PCA.out")

try:
    _try = subprocess.run([plink_path,"--vcf", vcf_file, "--keep-allele-order","--id-delim", "_","--indep-pairwise",win_size,step_size,prune_r2, "--allow-extra-chr","--out", indep_tmp_file, "--make-bed","--threads", threads], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
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


## Extracting prunned regions --> PCA 1 == usando não aparentados

print(color_text("Extracting prunning regions to VCF"))

extract_start = time.time()

prune_in = indep_tmp_file+".prune.in"

prune_err = os.path.join(temp_files, "short_range_prune.err")
prune_out = os.path.join(temp_files, "short_range_prune.out")

for_PCA1 = os.path.join(temp_files, base_name+".prunned.r2_"+str(prune_r2)+".for_PCA")

try:
    _try = subprocess.run([plink_path, "--bfile", indep_tmp_file, "--extract",prune_in, "--allow-extra-chr","--make-bed","--out",for_PCA1], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with open(prune_err, "w") as err:
        err.write(_try.stderr)
    with open(prune_out, "w") as out:
        out.write(_try.stdout)
    if _try.stderr:
        print(color_text("WARNING: Plink1.9. Check error log file "+indep_err, "red"))
except:
    print(color_text("Error on Plink1.9 execution", "red"))
    print(color_text("Path used for Plink1.9 executable = "+str(plink_path), "red"))
    print(color_text("Error log is stored in "+prune_err, "yellow"))
    exit(1)

extract_end = time.time()

extract_time = (extract_end - extract_start)

exec_times.append(["Extracting prunned", extract_time])

#2 Long range
#Removind LD long range

print(color_text("Removing Long Range LD regions"))

long_ld_start = time.time() 

if hg == "37":
    ld_long_databse = os.path.join(database_path, "long_range_hg19.txt")
if hg == "38":
    ld_long_databse = os.path.join(database_path, "long_range_hg38.txt")

set_to_remove = os.path.join(temp_files, "Long_range_LD.to_remove") #O plink acrescenta a extensão .set nesse arquivo (--write-set)

set_LD_long_err = os.path.join(temp_files, "Long_range_set.err")
set_LD_long_out = os.path.join(temp_files, "Long_range_set.out")


try:
    _try = subprocess.run([plink_path, "--bfile", for_PCA1, "--allow-extra-chr","--make-set", ld_long_databse, "--write-set", "--out", set_to_remove], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with open(set_LD_long_err, "w") as err:
        err.write(_try.stderr)
    with open(set_LD_long_out, "w") as out:
        out.write(_try.stdout)
    if _try.stderr:
        print(color_text("WARNING: Plink1.9. Check error log file "+set_LD_long_err, "red"))
except:
    print(color_text("Error on Plink1.9 execution", "red"))
    print(color_text("Path used for Plink1.9 executable = "+str(plink_path), "red"))
    print(color_text("Error log is stored in "+prune_err, "yellow"))
    exit(1)

#Agora removendo as regiões identificadas

# LD_out_file = vcf_file.replace(".vcf.gz", ".PRUNNED.LONG_LD")

# vcf_LD_out = os.path.join(temp_files, LD_out_file)

output_long_LD = os.path.join(temp_files, "Long_LD_prunned")

prunned_vcf_err = os.path.join(temp_files, "vcf_prunned.err")
prunned_vcf_out = os.path.join(temp_files, "vcf_prunned.out")

try:
    _try = subprocess.run([plink_path,"--bfile", for_PCA1, "--allow-extra-chr","--exclude", set_to_remove+".set", "--make-bed", "--out", output_long_LD], stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, text=True)
    with open(prunned_vcf_err, "w") as err:
        err.write(_try.stderr)
    with open(prunned_vcf_out, "w") as out:
        out.write(_try.stdout)
    if _try.stderr:
        print(color_text("WARNING: Plink1.9. Check error log file "+prunned_vcf_err, "red"))
except:
    print(color_text("Error on Plink1.9 execution", "red"))
    print(color_text("Path used for Plink1.9 executable = "+str(plink_path), "red"))
    print(color_text("Error log is stored in "+prunned_vcf_err, "yellow"))
    exit(1)

long_ld_end = time.time()

long_ld_times = (long_ld_end - long_ld_start)

exec_times.append(["Long-LD", long_ld_times])


#Split 1 --> BED for non-related --> Aqui vale a pena separar apenas os não relacionados pra colocar no flash PCA, também vale gerar apenas o .bed 
#por ser mais leve e a unica coisa necessária para a etapa do PCA1

split1_start = time.time()

print(color_text("First dataset split into non-related samples"))

#input_split_1 = LD_out_file+".vcf.gz"

non_related_output = os.path.join(temp_files, "non_related_split_1")

non_related_err = os.path.join(temp_files, "non_related.err")
non_related_out = os.path.join(temp_files, "non_related.out")

try:
    _try = subprocess.run([plink_path, "--bfile", output_long_LD, "--allow-extra-chr","--make-bed", "--remove", related_file, "--out",non_related_output], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    with open(non_related_err, "w") as err:
        err.write(_try.stderr)
    with open(non_related_output, "w") as out:
        out.write(_try.stdout)
    if _try.stderr:
        print(color_text("WARNING: Plink1.9. Check error log file "+prunned_vcf_err, "red"))
except:
    print(color_text("Error on Plink1.9 execution", "red"))
    print(color_text("Path used for Plink1.9 executable = "+str(plink_path), "red"))
    print(color_text("Error log is stored in "+non_related_err, "yellow"))
    exit(1)

split1_end = time.time()

split_1_time = (split1_end - split1_start)

exec_times.append(["Split 1 -- Non related", split_1_time])

## First Run PCA -- PCA 1

##A entrada é o bed file do plink!!

pca1_start = time.time()

print(color_text("Runing first PCA analysis"))

loadings_1 = os.path.join(temp_files, "SNP_loadings_PCA1.txt")

out_pc1 = os.path.join(temp_files, "PCA_1_RUN_1.txt")

laodings_1_err = os.path.join(temp_files, "loadings_1.err")
laodings_1_out = os.path.join(temp_files, "loadings_1.out")

try:
    _try = subprocess.run([flashPCA_path, "--bfile", non_related_output, "--outload", loadings_1, "--outpc", out_pc1, "-n", threads], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    with open(laodings_1_err, "w") as err:
        err.write(_try.stderr)
    with open(laodings_1_out, "w") as out:
        out.write(_try.stdout)
    if _try.stderr:
        print(color_text("WARNING: flashpca. Check error log file "+str(laodings_1_err), "red"))
except:
    print(color_text("Error on FlashPCA execution", "red"))
    print(color_text("Check if the executable is in "+str(flashPCA_path), "red"))
    print(color_text("Error log is stored in "+laodings_1_err, "yellow"))
    exit(1)

pca1_end = time.time()

pca1_time = (pca1_end - pca1_start)

exec_times.append(["PCA first run", pca1_time])

##Search for outiler loading values

load_start = time.time()

print(color_text("Looking for outlier load values"))

#Loading SNPs and loading values into numpy array

numpy_snps = np.genfromtxt(loadings_1,usecols=[0], skip_header=1, dtype="str")

numpy_loads = np.genfromtxt(loadings_1, usecols=[2], skip_header=1, dtype="float")

#calculating mean and standard deviation of the load values

load_mean = np.mean(numpy_loads)

load_std = np.std(numpy_loads)


#Finding indexes where loads deviates more than 3 times from the mean

to_remove_below_sd = np.argwhere(numpy_loads < load_mean - (2*load_std))

to_remove_up_sd = np.argwhere(numpy_loads > load_mean + (2*load_std))

#Concatenating and soting the arrays

to_remove = np.concatenate((to_remove_below_sd,to_remove_up_sd))

to_remove = np.sort(to_remove, kind="quicksort")

#Filtering SNPs

new_snps = np.delete(numpy_snps, to_remove, axis=0)

#Writing file of snps to keep for the next PCA 1 analysis

SNPs_to_keep = os.path.join(temp_files, "SNPs_to_keep_in_PCA.txt")

with open(SNPs_to_keep, "w") as keep:
    for l in new_snps:
        keep.write(l+"\n")

print(color_text("It was found "+str(len(to_remove))+" SNPs with outiler loading values", "yellow"))


load_end = time.time()

load_time = (load_end - load_start)

exec_times.append(["Get outlier loadings", load_time])



## Removing outlier SNPs from the bed file and running new PCA

remove_snps_start = time.time()

print(color_text("Removing outlier SNPs"))

bfile_to_remove_snps = non_related_output

clean_bfile = os.path.join(temp_files, "non_related_split_1_clean")
clean_loads_err = os.path.join(temp_files, "clean_outlier_loads.err")
clean_loads_out = os.path.join(temp_files, "clean_outlier_loads.out")

try:
    _try = subprocess.run([plink_path, "--bfile", bfile_to_remove_snps ,"--allow-extra-chr","--make-bed", "--extract", SNPs_to_keep, "--out", clean_bfile], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)
    with open(clean_loads_err, "w") as err:
        err.write(_try.stderr)
    with open(clean_loads_out, "w") as out:
        out.write(_try.stdout)
    if _try.stderr:
        print(color_text("WARNING: flashpca. Check error log file "+str(clean_loads_err), "red"))
except:
    print(color_text("Error on FlashPCA execution", "red"))
    print(color_text("Check if the executable is in "+str(flashPCA_path), "red"))
    print(color_text("Error log is stored in "+laodings_1_err, "yellow"))
    exit(1)


remove_snps_end = time.time()

remove_snps_time = (remove_snps_end - remove_snps_start)

exec_times.append(["Removing outlier SNPs", remove_snps_time])


##Running second PCA1, now with no outlier SNPs

pca1_run2_start = time.time()

out_pc1_run2 = os.path.join(temp_files, "PCA_1_RUN_2.txt")

out_mean_sd = os.path.join(temp_files, "PCA_1_out_mean_sd.txt")

out_load_2 = os.path.join(temp_files, "PCA_1_RUN_2_outloads.txt")

pca_1_run_2_err = os.path.join(temp_files, "PCA_1_RUN_2.err")
pca_1_run_2_out = os.path.join(temp_files, "PCA_1_RUN_2.out")

try:
    _try = subprocess.run([flashPCA_path, "--bfile", clean_bfile, "--outpc", out_pc1_run2, "--outmeansd", out_mean_sd, "--outload", out_load_2,"-n", threads], stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE, text=True)
    with open(pca_1_run_2_err, "w") as err:
        err.write(_try.stderr)
    with open(pca_1_run_2_out, "w") as out:
        out.write(_try.stdout)
    if _try.stderr:
        print(color_text("WARNING: flashpca. Check error log file "+str(pca_1_run_2_err), "red"))
except:
    print(color_text("Error on FlashPCA execution", "red"))
    print(color_text("Check if the executable is in "+str(flashPCA_path), "red"))
    print(color_text("Error log is stored in "+laodings_1_err, "yellow"))
    exit(1)

pca1_run2_end = time.time()

pca1_rune_2_time = (pca1_run2_start - pca1_run2_end)

exec_times.append(["PCA 1 Second Run", pca1_rune_2_time])

## Running admixture for denovo population separation




admixture_start = time.time()

print(color_text("Starting Admixture run"))

#input é o .bed do plink --> nesse caso entra o bed sem aparentados e com os SNPs outliers retirados

input_for_admixture = clean_bfile+".bed"

# if N_pop:
#     try:
#         admix_run = subprocess.run([admixture_path,"-j"+str(threads), "--cv", input_for_admixture, str(i)], capture_output=True)
#     except:
#         print(color_text("ERROR: Admixture.", "red"))

#     Q_file = os.path.join(out_dir_path, clean_bfile+"."+str(N_pop)+".Q")
#     n_pops = int(N_pop)


#Vamos fazer um for para rodar até 10 populãções
list_results = []

try:
    for i in range(2,11):
        print(color_text("Running analysis with K populations = "+str(i), "yellow"))
        admix_run = subprocess.run([admixture_path,"-j"+str(threads), "--cv", input_for_admixture, str(i)], capture_output=True)
        admix_output = admix_run.stdout.decode()
        parse_step_1 = admix_output.split("\n") #gerar uma lista com todas as linhas recuperadas do stdout
        for j in parse_step_1:
            if j.startswith("CV error"): #A partir da lista pegamos apenas a informação desejada --> o erro calculado pela ferramenta
                parse_step_2 = j.split(":") #aqui dividimos o resultado encontrado em ":", ficando parse_step_2[0] = K=* e parse_step_2[1] = valor do erro
                parse_step_3 = parse_step_2[0] #Manter apenas a primeira parte para outro parse usando Regex
                regex_1 = re.findall(r'=[0-9]*', parse_step_3) #recuperamos apenas o valor de K
                final_regex = regex_1[0].replace("=","")
                list_results.append([final_regex, float(parse_step_2[1])])
except:
    print(color_text("ERROR in admixture populaton analysis"))
    exit(1)

list_results.sort(key=lambda x: x[1])

best_results = list_results[0]

print(color_text("Best number of K populations found is "+str(best_results[0])+" with CV error = "+str(best_results[1])))

#admixture files are always outputed on the script path! 
print("Moving files")
admix_output = GARSA_path

for file in os.listdir(admix_output):
    if file.endswith(".txt"):
        values_current = os.path.join(admix_output, file)
        values_new = os.path.join(temp_files, file)
        shutil.move(values_current, values_new)
    if file.endswith(".P") or file.endswith(".Q"):
        current = os.path.join(admix_output, file)
        new = os.path.join(temp_files, file)
        shutil.move(current, new)

#We also need the ID of all samples

ids_input= os.path.join(out_dir_path, clean_bfile+".fam")

fam_df = pd.read_csv(ids_input, header=None, sep=" ") #lendo o arquivo .fam

#O arquivo de saida é sempre o nome do arquivo de entrada (sem extensão) + .número + .Q --> numero nesse caso vai ser o de melhor K == best_results[0]

Q_file = os.path.join(out_dir_path, clean_bfile+"."+str(best_results[0])+".Q")

n_pops = int(best_results[0]) #Numero total de populações a serem usdas

Pop_columns = ["Pop"+str(x) for x in range(1, n_pops+1)] #Aqui criamos automaticamente as colunas referentes as populações 

q_df = pd.read_csv(Q_file, names= Pop_columns, sep=" ")

admixture_end = time.time()

admixture_time = (admixture_end - admixture_start)

exec_times.append(["Admixture Run", admixture_time])

print(color_text("Preparing table output with population, sample ID and PCs"))

pandas_start = time.time()

#Adciona população com maior probabilidade

q_df["Best_POP"] = q_df.idxmax(axis=1)

#Adicionando uma coluna de IDs no q_df

q_df["Sample_ID"] = fam_df.apply(lambda x: str(x[0])+"_"+str(x[1]), axis=1)


#Getting PCA 1 run 2 data

pca_data = pd.read_csv(out_pc1_run2, sep="\t")

pca_data = pca_data.astype({"FID": "str", "IID": "str"})

pca_data["Sample_ID"] = pca_data.apply(lambda x: str(x[0])+"_"+str(x[1]), axis=1)

pca_data.drop(["FID", "IID"], axis=1, inplace=True)

#Unir todos os DFs para gerar o df para plot

complete_pc_data = pd.merge(q_df, pca_data, on="Sample_ID", how="left")

to_plot = os.path.join(temp_files, "table_for_plot.tsv")

complete_pc_data.to_csv(to_plot, index=False, sep="\t")

pandas_end = time.time()

pandas_time = (pandas_end - pandas_start)

exec_times.append(["PC table", pandas_time])

## Plotting PCA scatter

plot_start = time.time()

print(color_text("Plotting PCA scatterplots, please check all plots for informative PCs before using them as covariates in the GWAS step", "yellow"))

#Definir a coluna fixa que é o PC1

x_col = "PC1"

#Definir as colunas que variam, de PC2 até PC10

y_col = ["PC"+str(c) for c in range(2,11)]

#Identificar quais PCs estão contidos na tabela

pc_cols = complete_pc_data.columns.tolist()

to_pop = []
for i in range(len(y_col)):
    if y_col[i] not in pc_cols:
        to_pop.append(i)
if to_pop:
    del y_col[to_pop[0]:to_pop[-1]+1]


#Definições de tamanho de figura

figure = plt.figure(figsize=(15,10))

#contagem do index do plot
count = 1

#loop para plot

for y in y_col:
    plt.subplot(3,3,count) #Definido como 3 plots por linha com 3 linhas

    sns.scatterplot(x=complete_pc_data[x_col], y=complete_pc_data[y], hue=complete_pc_data["Best_POP"])

    count+=1

#Saving the plot

output_plot = os.path.join(out_dir_path, "PC_plots_PCA1.pdf")

plt.savefig(output_plot, dpi=300)

plot_end = time.time()

plot_time = (plot_end - plot_start)

exec_times.append(["Plotting", plot_time])


#Starting PCA 2

if os.path.getsize(related_file) > 0:
    pca_2_start = time.time()

    print(color_text("Starting PCA projection to related individuals"))

    #Precisa entrar loads do pca 1 run 2, meansd do pca 1 run 2 e bfile apenas com os aparentados

    print(color_text("Getting realted individuals", "yellow"))

    related_output = os.path.join(temp_files, "related_split_1")

    related_err = os.path.join(temp_files, "related_split.err")
    related_out = os.path.join(temp_files, "related_split.out")


    try:
        _try = subprocess.run([plink_path, "--bfile", output_long_LD,"--allow-extra-chr","--make-bed", "--keep", related_file, "--out",related_output], stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, text=True)
        with open(related_err, "w") as err:
            err.write(_try.stderr)
        with open(related_out, "w") as out:
            out.write(_try.stdout)
        if _try.stderr:
            print(color_text("WARNING: Plink1.9. Check error log file "+related_err, "red"))
    except:
        print(color_text("Error on Plink1.9 execution", "red"))
        print(color_text("Path used for Plink1.9 executable = "+str(plink_path), "red"))
        print(color_text("Error log is stored in "+related_err, "yellow"))
        exit(1)

    print(color_text("Starting projection"))

    out_pc2 = os.path.join(temp_files, "PCA_2_projection.txt")

    projection_err = os.path.join(temp_files, "projection.err")
    projection_out = os.path.join(temp_files, "projection.out")


    try:
        _try = subprocess.run([flashPCA_path, "--bfile", related_output, "--project","--outproj", out_pc2, "--inmeansd", out_mean_sd, "--inload", out_load_2,"-n", threads, "-v"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with open(projection_err, "w") as err:
            err.write(_try.stderr)
        with open(projection_out, "w") as out:
            out.write(_try.stdout)
        if _try.stderr:
            print(color_text("WARNING: flashpca. Check error log file "+str(projection_err), "red"))
    except:
        print(color_text("Error on FlashPCA execution", "red"))
        print(color_text("Check if the executable is in "+str(flashPCA_path), "red"))
        print(color_text("Error log is stored in "+projection_err, "yellow"))


if os.path.getsize(related_file) == 0:
    print(color_text("No realted individuals skpping PCA 2", "yellow"))


if os.path.getsize(related_file) > 0:
    #Unir PCA 1 e PCA 2

    #Getting PCA 2 data

    print(color_text("Concatenating PCA data -- PCA 1 and PCA 2"))

    pca2_data = pd.read_csv(out_pc2, sep="\t")

    pca2_data = pca2_data.astype({"FID": "str", "IID": "str"})

    pca2_data["Sample_ID"] = pca2_data.apply(lambda x: str(x[0])+"_"+str(x[1]), axis=1)

    pca2_data.drop(["FID", "IID"], axis=1, inplace=True)

    pca_total = pd.concat([pca_data, pca2_data])

    mask_df = pca_total["Sample_ID"].str.split("_", expand=True)

    pca_total["FID"] = mask_df[0]

    pca_total["IID"] = mask_df[1]

    pca_total.drop(["Sample_ID"], axis=1 ,inplace=True)

    pca_total.insert(0, "IID", pca_total.pop("IID"))

    pca_total.insert(0, "FID", pca_total.pop("FID"))

    output_pca_total = os.path.join(out_dir_path, base_name+"_PCA_total.txt") #PCA em formato tabular que pode ser usado de diferentes formas

    output_pca_gwas = os.path.join(out_dir_path, base_name+"_PCA_GWAS_GCTA.txt") #PCA com espaço como separador para ser usado no GCTA

    pca_total.to_csv(output_pca_total, index=False, sep="\t")

    pca_total.to_csv(output_pca_gwas, index=False, sep=" ", header=False)

    print(color_text("PCA file for covar in GWAS generated as "+output_pca_gwas))

    pca_2_end = time.time()

    pca_2_time = (pca_2_end - pca_2_start)

    exec_times.append(["PCA 2 - Related", pca_2_time])

if os.path.getsize(related_file) == 0:
    output_pca_total = os.path.join(out_dir_path, base_name+"_PCA_total.txt")

    output_pca_gwas = os.path.join(out_dir_path, base_name+"_PCA_GWAS_GCTA.txt")

    mask_df = pca_data["Sample_ID"].str.split("_", expand=True)

    pca_data["FID"] = mask_df[0]

    pca_data["IID"] = mask_df[1]

    pca_data.drop(["Sample_ID"], axis=1 ,inplace=True)

    pca_data.insert(0, "IID", pca_data.pop("IID"))

    pca_data.insert(0, "FID", pca_data.pop("FID"))

    pca_data.to_csv(output_pca_total, index=False, sep="\t")

    pca_data.to_csv(output_pca_gwas, index=False, sep=" ")

#Salvando arquivo de tempos

exec_times_df = pd.DataFrame.from_records(exec_times, columns=["Task", "Time"])

exec_out = os.path.join(out_dir_path,base_name+"_PCA_execution_times.tsv")

exec_times_df.to_csv(exec_out, index=False, sep="\t")