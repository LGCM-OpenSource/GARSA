# -- coding: utf-8 --
## Inscript principal FOXCONN

## Uniao dos scripts em uma única ferramenta

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

#####################
## Parse arguments ##
#####################
print(color_text(r""" __          ________ _      _____ ____  __  __ ______   _______ ____     _____          _____   _____         
 \ \        / /  ____| |    / ____/ __ \|  \/  |  ____| |__   __/ __ \   / ____|   /\   |  __ \ / ____|  /\    
  \ \  /\  / /| |__  | |   | |   | |  | | \  / | |__       | | | |  | | | |  __   /  \  | |__) | (___   /  \   
   \ \/  \/ / |  __| | |   | |   | |  | | |\/| |  __|      | | | |  | | | | |_ | / /\ \ |  _  / \___ \ / /\ \  
    \  /\  /  | |____| |___| |___| |__| | |  | | |____     | | | |__| | | |__| |/ ____ \| | \ \ ____) / ____ \ 
     \/  \/   |______|______\_____\____/|_|  |_|______|    |_|  \____/   \_____/_/    \_\_|  \_\_____/_/    \_\
                                                                                                               
                                                                                                               """))
print(r"""            __
           /(`o
     ,-,  //  \\
    (,,,) ||   V
   (,,,,)\//		
   (,,,/w)-'
   \,,/w)
   `V/uu
     / |
     \ |
\,/  ,\|,.  \,/""")
print(color_text("V.0.1", "yellow"))

arg_parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description= textwrap.dedent( '''This script integrates each analysis of the FOXCONN pipeline
______________________________________________________________
dedup			-- Runs the desduplication analysis, removing duplicated SNPs or multiallelic variants
update_rsID		-- Runs the update of all (possible) rsIDs using hg19 or hg38 references
rename_sample_id	-- Runs an update of samples ID
quality_control		-- Runs the quality control script for SNPs
quality_ind		-- Runs Quality control for individuals with missing data
kinship			-- Runs Kinship analysis and correction for admixture populations
PCA			-- Runs PCA and population analysis
GWAS			-- Runs GWAS analysis using GCTA or BOLT-LMM software
PRS 			-- Runs PRS analysis using LDPred2
download_db		-- Download the database needed for the update_rsID module
''')
)

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
# database_path = os.path.join(primary_script_path, "database")

# setting scripts path (assuming they are in the same directory as the primary script)
script_path = os.path.join(primary_script_path, "modules")

#Pgear toda a linha de comando passada
args = sys.argv[1:]

#print(args)

#O primeiro comando deve sempre ser o script/analise a ser rodado!!

command = args[0]
print(command)

if command == "-h" or command == "--help":
	arg_parser.print_help(sys.stderr)
	sys.exit(0)

if command == "dedup":
	script_to_run = os.path.join(script_path,"deduplication.py")
	arguments = ["python3",script_to_run] + args[1:]
	#print(arguments)
	subprocess.run(arguments)


if command == "update_rsID":
	script_to_run = os.path.join(script_path,"update_rsID.py")
	arguments = ["python3",script_to_run] + args[1:]
	#print(arguments)
	subprocess.run(arguments)

if command == "rename_sample_id":
	script_to_run = os.path.join(script_path,"rename_sample_id.py")
	arguments = ["python3",script_to_run] + args[1:]
	#print(arguments)
	subprocess.run(arguments)

if command == "quality_control":
	script_to_run = os.path.join(script_path,"SNP_QC.py")
	arguments = ["python3",script_to_run] + args[1:]
	#print(arguments)
	subprocess.run(arguments)

if command == "quality_ind":
	script_to_run = os.path.join(script_path,"sample_QC.py")
	arguments = ["python3",script_to_run] + args[1:]
	subprocess.run(arguments)

if command == "kinship":
	script_to_run = os.path.join(script_path,"Kinship_and_correction.py")
	arguments = ["python3",script_to_run] + args[1:]
	subprocess.run(arguments)

if command == "PCA":
	script_to_run = os.path.join(script_path,"PCA_analysis.py")
	# arguments = ["python3",script_to_run] + ["--garsa_path",primary_script_path] + args[1:]
	arguments = ["python3",script_to_run] + args[1:]
	# 
	subprocess.run(arguments)

if command == "GWAS":
	script_to_run = os.path.join(script_path,"GWAS.py")
	arguments = ["python3",script_to_run] + args[1:]
	subprocess.run(arguments)	

if command == "PRS":
	script_to_run = os.path.join(script_path,"LDPred_PRS.py")
	arguments = ["python3",script_to_run] + args[1:]
	subprocess.run(arguments)

if command == "download_db":
	script_to_run = os.path.join(script_path, "download_database.py")
	subprocess.run(["python3", script_to_run])

c = ["dedup", "update_rsID", "rename_sample_id", "quality_control", "quality_ind", "kinship", "PCA", "GWAS", "PRS", "download_db"]

if command not in c:
	print(color_text("Command not found", "red"))
	print(color_text("Valid commands are "+str(c), "yellow"))


