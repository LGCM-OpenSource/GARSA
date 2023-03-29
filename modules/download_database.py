# -- coding: utf-8 --
import os
import sys
import subprocess


## Quick check on the database folder

# getting primary script full path
path = os.path.realpath(__file__)


# getting primary script directory full path
primary_script_path = path.split("/")[:-1]
primary_script_path = "/".join(primary_script_path)


# setting database full path (assuming they are in the same directory as the primary script)
database_path = os.path.join(primary_script_path, "database")


#Try to create folders if it not already exists

hg37_folder = os.path.join(database_path, "dbsnp_hg37p13_b151")

hg37_vcf = os.path.join(hg37_folder, "hg37_b151_dbSNP.vcf.gz")
hg37_tbi = os.path.join(hg37_folder,"hg37_b151_dbSNP.vcf.gz.tbi")

if not os.path.exists(hg37_folder):
	print("Creating folder and checking for SNPdb files")
	os.mkdir(hg37_folder)

	if not os.path.exists(hg37_vcf):
		print("hg37 VCF file not found, starting download")
		subprocess.run(["wget", "-O",hg37_vcf,"https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh37p13/VCF/00-All.vcf.gz"])
		print("Downloading index file")
		subprocess.run(["wget", "-O", hg37_tbi,"https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh37p13/VCF/00-All.vcf.gz.tbi"])
	else:
		print("Files found =)")


hg38_folder = os.path.join(database_path, "dbsnp_hg38p7_b151")

hg38_vcf = os.path.join(hg38_folder, "hg38_b151_dbSNP.vcf.gz")
hg38_tbi = os.path.join(hg38_folder,"hg38_b151_dbSNP.vcf.gz.tbi")

if not os.path.exists(hg38_folder):
	print("Creating folder and checking for SNPdb files")
	os.mkdir(hg38_folder)

	if not os.path.exists(hg38_vcf):
		print("hg38 VCF file not found, starting download")
		subprocess.run(["wget", "-O", hg38_vcf, "https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh38p7/VCF/00-All.vcf.gz"])
		print("Downloading index file")
		subprocess.run(["wget", "-O", hg38_vcf, "https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh38p7/VCF/00-All.vcf.gz.tbi"])

