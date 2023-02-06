library(bigsnpr)
options(bigstatsr.check.parallel.blas = FALSE)
options(default.nproc.blas = NULL)
library(data.table)
library(magrittr)
library(ggplot2)
library(optparse)
library(tidyverse)

option_list = list(make_option(c("--bfile"), type="character", default=NULL, 
                               help="bfile for processing", metavar="character"),
                   make_option(c("--mlma"), type="character", default=NULL, help="GWAS output file -- .mlma",
                               metavar="character"),
                   make_option(c("--pheno"), type="character", default=NULL, help="GWAS Phenotype input file",
                               metavar="character"),
                   make_option(c("--covar"), type="character", default=NULL, help="GWAS Covars input file",
                               metavar="character"),
                   make_option(c("--qcovar"), type="character", default=NULL, help="GWAS qCovars input file",
                               metavar="character"),
                   make_option(c("--database"), type="character", default=NULL, 
                               help="SNPs_database in RDS format", metavar="character"),
                   make_option(c("--outdir"), type="character", default=NULL,
                               help="Output directory", metavar="character"),
                   make_option(c("--PCs"), type="integer", default=NULL,
                               help="Number of PCs to use", metavar="integer"),
                   make_option(c("--threads"), type="integer", default=NULL,
                               help="Number of threads to use", metavar="integer"))

opt_parser = OptionParser(option_list=option_list)

opt = parse_args(opt_parser)

bfile = opt$bfile
mlma = opt$mlma
pheno_file = opt$pheno
covar = opt$covar
qcovar = opt$qcovar
database = opt$database
out_dir = opt$outdir
n_pcs = opt$PCs
threads = opt$threads

##Read in the phenotype, covariate and pcs files

#Phenotype --> mesma tabela de fenotipos do GWAS

phenotype <- read.table(pheno_file,
                        )
colnames(phenotype) = c("FID", "IID", "Pheno")
#Tabela de covars do GWAS

covariate <- read.table(covar,
                        header = TRUE)
#Tabela de qcovars do GWAS -- Aqui como temos apenas 4 PCs, pegamos as colunas de FID e IID + PC1 .. PC4

pcs <- read.table(qcovar,
                  header = TRUE)

##Gerando a tabela necessária para seguir em frente

pheno <- merge(phenotype, covariate) %>%
  merge(., pcs)


##HapMapSNPs

info <- readRDS(database)
info <- drop_na(info)
#Load Summary file do GWAS

sumstats <- bigreadr::fread2(mlma)
sumstats <- drop_na(sumstats)
# Filter out hapmap SNPs
print("Filtering SNPs using Reference -- database")
sumstats <- sumstats[sumstats$rsid%in% info$rsid,]
print(paste0(length(sumstats$rsid), " SNPs will be used for the LDPRED analysis"))

## Calculate the LD matrix ##

NCORES <- threads

# Open a temporary file
tmp <- tempfile(tmpdir = out_dir)
on.exit(file.remove(paste0(tmp, ".sbk")), add = TRUE)

# Initialize variables for storing the LD score and LD matrix
corr <- NULL
ld <- NULL

# We want to know the ordering of samples in the bed file 
fam.order <- NULL

# preprocess the bed file (only need to do once for each data set)
print("processing .bed file")
snp_readBed(bfile)

bfile_rds =gsub(".bed", ".rds", bfile)

# now attach the genotype object
obj.bigSNP <- snp_attach(bfile_rds)

# extract the SNP information from the genotype

map <- obj.bigSNP$map[-3]
names(map) <- c("chr", "rsid", "pos", "a1", "a0")

# perform SNP matching
print("SNP matching against reference")
info_snp <- snp_match(sumstats, map,join_by_pos = FALSE)

# Assign the genotype to a variable for easier downstream analysis
print("Assign the genotype to a variable")
genotype <- obj.bigSNP$genotypes

#imputation in genotype file, because it can't contains NA values
#genotype_2 <- snp_fastImputeSimple(genotype, ncores = NCORES)

# Rename the data structures
CHR <- map$chr
POS <- map$pos
# get the CM information from 1000 Genome

POS2 <- snp_asGeneticPos(CHR, POS, dir = out_dir)


# calculate LD
print("Calculating LD scores")
for (chr in 1:22) {
  print(paste0("Working on chr", chr))
  # Extract SNPs that are included in the chromosome
  ind.chr <- which(info_snp$chr == chr)
  ind.chr2 <- info_snp$`_NUM_ID_`[ind.chr]
  # Calculate the LD
  corr0 <- snp_cor(
    genotype,
    ind.col = ind.chr2,
    ncores = NCORES,
    infos.pos = POS2[ind.chr2],
    size = 3 / 1000
  )
  if (chr == 1) {
    ld <- Matrix::colSums(corr0^2)
    corr <- as_SFBM(corr0, tmp)
  } else {
    ld <- c(ld, Matrix::colSums(corr0^2))
    corr$add_columns(corr0, nrow(corr))
  }
}

# We assume the fam order is the same across different chromosomes
fam.order <- as.data.table(obj.bigSNP$fam)
# Rename fam order
setnames(fam.order,
         c("family.ID", "sample.ID"),
         c("FID", "IID"))

## Perform LD score regression
print("Starting LD score regression")
warnings()
df_beta <- info_snp[,c("beta", "beta_se", "n_eff", "_NUM_ID_")]
ldsc <- snp_ldsc( ld, 
                  length(ld), 
                  # chi2 = (df_beta$beta / df_beta$beta_se)^2,
                  chi2 = `^`((df_beta$beta / df_beta$beta_se),2),
                  sample_size = df_beta$n_eff, 
                  blocks = NULL)
h2_est <- ldsc[["h2"]]

#h2 não pode ser negativo, de acordo com tutoriais caso seja negativo botar o valor de 0.01

if (ldsc[["h2"]] < 0){
  h2_est <- 0.01
} else {
  h2_est <- ldsc[["h2"]]
}

fam.order =
  fam.order %>% as.data.frame()

# Reformat the phenotype file such that y is of the same order as the 
# sample ordering in the genotype file
y <- pheno[match(fam.order[,1],pheno[,1]),]

# Calculate the null R2
# use glm for binary trait 
# (will also need the fmsb package to calculate the pseudo R2)
#Aqui testamos o modelo nulo do fenótipo x Sexo -- Sexo, de acordo com a nossa documentação sempre é a terceira coluna do arquivo covar

sex_column <- names(covariate[3])

null.model <- paste("PC", 1:n_pcs, sep = "", collapse = "+") %>%
  paste0("Pheno~PRS",sex_column, "+", .) %>%
  as.formula %>%
  lm(., data = y) %>%
  summary
null.r2 <- null.model$r.squared

# Prepare data for grid model
# Get adjusted beta from the auto model
print("Starting LDPred auto model")
multi_auto <- snp_ldpred2_auto(
  corr,
  df_beta,
  h2_init = h2_est,
  vec_p_init = seq_log(1e-4, 0.9, length.out = NCORES),
  ncores = NCORES
)
beta_auto <- sapply(multi_auto, function(auto)
  auto$beta_est)
genotype <- obj.bigSNP$genotypes
# calculate PRS for all samples
print("Starting PRS calculations for all samples")
ind.test <- 1:nrow(genotype)
pred_auto <-
  big_prodMat(genotype,
              beta_auto,
              ind.row = ind.test,
              ind.col = info_snp$`_NUM_ID_`)
# scale the PRS generated from AUTO
pred_scaled <- apply(pred_auto, 2, sd)
final_beta_auto <-
  rowMeans(beta_auto[,
                     abs(pred_scaled -
                           median(pred_scaled)) <
                       3 * mad(pred_scaled)])
pred_auto <-
  big_prodVec(genotype,
              final_beta_auto,
              ind.row = ind.test,
              ind.col = info_snp$`_NUM_ID_`)

##Get the final performance of the LDpred models
reg.formula <- paste("PC", 1:n_pcs, sep = "", collapse = "+") %>%
  paste0("Pheno~PRS+",sex_column,"+", .) %>%
  as.formula
reg.dat <- y
reg.dat$PRS <- pred_auto
auto.model <- lm(reg.formula, dat=reg.dat) %>%
  summary
(result <- data.table(
  auto = auto.model$r.squared - null.r2,
  null = null.r2
))

#Gerar dataset final
print("Generating final dataset")

#Recuperar rsID, alelo1 e betas_auto

#rsID e alelo 1 estão em sumstats!

rsID <- sumstats$rsid %>% as.data.frame() %>% rename("rsID" = ".")
allele1 <- sumstats$a0 %>% as.data.frame() %>% rename("A1" = ".")

beta_new <- final_beta_auto %>% as.data.frame() %>% rename("LDPRED2_betas" = ".")

final_dataset <- NULL

final_dataset$rsID <- rsID$rsID
final_dataset$A1 <- allele1$A1
final_dataset$LDPRED2_betas <- beta_new$LDPRED2_betas

file_output <- file.path(out_dir, "weights_LDPRED2.tsv")

write.table(final_dataset, 
            file = file_output,
            sep = "\t", row.names = FALSE,
            col.names = TRUE, quote = FALSE)
