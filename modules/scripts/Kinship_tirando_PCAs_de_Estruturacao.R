library(SeqArray)
library(SNPRelate)
library(ggplot2)
library(GENESIS)
library(dplyr)
library(SeqVarTools)
library(rgl)
library(tidyr)
library(dplyr)
library(reshape)
library(genio)
library(tidyverse)
library(tibble)
library("optparse")

option_list = list(make_option(c("--vcf_file"), type="character", default=NULL, 
                               help="vcf file for processing", metavar="character"),
make_option(c("--output"), type="character", default=NULL, help="Output dir", metavar="character"))

opt_parser = OptionParser(option_list=option_list)

opt = parse_args(opt_parser)

if (is.null(opt$vcf_file)){
  print_help(opt_parser)
  stop("The argument -vcf/--vcf_file must be suplied", call.=FALSE)
}

# Carregar arquivo VCF
vcf_file = opt$vcf_file

out_path = opt$output

# Identificar arquivo para output em gds
gds_file = gsub(".vcf", ".gds",vcf_file)

# Converter
seqVCF2GDS(vcf_file, gds_file, fmt.import="GT", storage.option="LZMA_RA")

# Open GDS file
gds <- seqOpen(gds_file)

#1. Começando gerando uma Genetic Relationship matrix (GRM)
#Essa matriz nada mais é do que o calculo de similaridade 2 a 2 --> o metodo aplicado é o GCTA Yang J.Lee S.H.Goddard M.E.Visscher P.M.
#GCTA: a tool for genome-wide complex trait analysis.
#Am. J. Hum. Genet. 2011; 88: 76-82
#grm <- snpgdsGRM(gds, method="GCTA")

#Para que seja possivel separar as relações de ancestralidade distantes (estrutura populacional) 
#das relações de parentesco recentes (relação familiar)
#será implementada a analise feita por Conomos et al., 2016 --> https://www.cell.com/ajhg/fulltext/S0002-9297(15)00496-6

#Para isso são feitas 3 analises KING, PC-AiR, e PC-Relate:
#1. KING --> calculo robusto para estrutura familiar, mas não robusto para a estrutura da população
king <- snpgdsIBDKING(gds)
kingMat <- king$kinship

#Como são amostras par-a-par, vamos definir como linhas e colunas (index, e col_index) os samples.id
colnames(kingMat) <- rownames(kingMat) <- king$sample.id

#Agora podemos extrair os valores de KINSHIP e IBS0 (a proporção de variantes
#pelo qual o par de indivudos compartilham 0 alelos "identical by state")
kinship <- snpgdsIBDSelection(king)
head(kinship)

# Para visualizarmos esses dados
plot_1=paste0(out_path, "/kinship_estimate.png")
png(plot_1, res=300,width = 1500, height = 1600)
ggplot(kinship, aes(IBS0, kinship)) +
  geom_hline(yintercept=2^(-seq(3,9,2)/2), linetype="dashed", color="grey") +
  geom_hex(bins = 100) +
  ylab("kinship estimate") +
  theme_bw()
dev.off()

# A grande quantidade de valores negativos observados na imagem são indicativos de individuos que possuem ancestralidade
# de diferentes populações

# 2. A proxima etapa é usar PCAiR, uma ferramente poderosa pra inferencia de estruturas populacionais em amostras com estruturas
# de "kinship", ou seja, aparentadas

# Primeiro, o PC-AiR particiona o conjunto de amostras completo em um conjunto de amostras mutuamente não relacionadas 
#que possui o maior valor informativo sobre todos os ancestrais na amostra (conjunto não relacionado) 
#e seus parentes (conjunto relacionado). Usamos um limiar de parentesco de 3º grau (kin.thresh = 2^(-9/2)), que corresponde 
#a primos de primeiro grau – isso define qualquer pessoa menos relacionada que primos de primeiro grau como “não relacionada”. 
#Usamos as estimativas KING-robust negativas como medidas de “divergência de ancestralidade” (divMat) para identificar pares de amostras
#com ancestralidade diferente – selecionamos preferencialmente indivíduos com muitas estimativas negativas para o conjunto
#não relacionado para garantir a representação de ancestralidade. 
#Também usamos as estimativas KING-robust como nossas medidas de parentesco (kinMat)

# Uma vez que os conjuntos não relacionados e relacionados são identificados, o PC-AiR executa uma Análise de Componente Principal 
#(PCA) padrão no conjunto não relacionado de indivíduos e, em seguida, projeta os parentes nesses componentes (PCs).

# Agora vamos fazer a analise de PCA tendo como base o KINSHIP para dividir a população entre aparentados e não aparentados


pca <- pcair(gds, 
             kinobj = kingMat,
             kin.thresh = 0.044194174,
             divobj = kingMat,
             div.thresh = -0.044194174)

# Agora vamos extrair os primeiros 10 componentes do PCA em um dataframe
pcs <- data.frame(pca$vectors[,1:10])

# Vamos adicionar o nome de cada um dos componentes
colnames(pcs) <- paste0('PC', 1:10)

# E vamos ajustar o nome das amostras/individuos
pcs$sample.id <- pca$sample.id

# Para checar quantos PCs são necessários para explicar a variancia no dado fazemos:

#1. Checar a variancia total de cada PC
#Biblioteca para analisar o df coluna a coluna
std_dev <- pcs %>% summarise_if(is.numeric, sd)

# calcular a variancia
pc_var = std_dev^2

# para entender o quanto cada componente contribui para explicar a variancia dos dados nós podemos simplesmente calcular:
# variancia/somatoria(variancia)
prop_varex <- pc_var/sum(pc_var)

# Vamos plotar essas analises para visualizar o quanto cada componente explica
plot_2=paste0(out_path, "/pca_variance.png")
png(plot_2, res=300,width = 1500, height = 1600)
plot(as.numeric(prop_varex), xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type="b")
dev.off()
#Nesse caso, podemos ver que os cinco primeiros componentes explicam a maior parte dos dados
#3. Para o proximo passo vamos usar PC-relate, que faz uma nova analise de KINSHIP usando os PCAs para ajustar/corrigir
#em relação a ancestralidade predita pelo PCA
#Para rodar o PCrelate precisa de um objeto SeqVarIterator
seqResetFilter(gds, verbose=FALSE)
seqData <- SeqVarData(gds)
seqSetFilter(seqData)

# Gerar o iterator
iterator <- SeqVarBlockIterator(seqData, verbose=FALSE)

#Calcular as novas relações com a correção do PCA
#O argumento training.set permite especificar quais amostras usar para “aprender” o ajuste de ancestralidade
#recomendamos o conjunto não relacionado da análise PC-AiR.
pcrel <- pcrelate(iterator, 
                  pcs=pca$vectors[,1:4], 
                  training.set=pca$unrels)

#Plotamos as estimativas de parentesco pareadas (KINSHIP) contra as estimativas IBD0 (k0) 
#(a proporção de variantes para as quais o par de indivíduos compartilha 0 alelos idênticos por descendência (IBD)). 
#Usamos um gráfico hexbin para visualizar o parentesco para todos os pares de amostras.
plot_3=paste0(out_path, "/kinship_corrected.png")
png(plot_3, res=300,width = 1500, height = 1600)
ggplot(pcrel$kinBtwn, aes(k0, kin)) +
  geom_hline(yintercept=2^(-0.044194174), linetype="dashed", color="grey") +
  geom_hex(bins = 100) +
  geom_abline(intercept = 0.25, slope = -0.25) + 
  ylab("kinship estimate") +
  theme_bw()
dev.off()
# Agora para finaliar podemos transformar em uma matriz de Kinhsip (todos contra todos)
# scaleKin=2, se depois for ser lido no GCTA.
pcrelMat <- pcrelateToMatrix(pcrel, scaleKin=2, verbose=FALSE)

# pegar os dados de King e IBD0
KingxIBD = pcrel$kinBtwn

output_file=paste0(out_path, "/Kinship_corrected.tsv")

write.table(KingxIBD,output_file,
          row.names = FALSE, quote = FALSE, sep = "\t")

# Multiplicar por 2 (pra ser compativel com range do GCTA)
# KingxIBD_x2 = KingxIBD
# KingxIBD_x2[3] <- lapply(KingxIBD_x2[3], function(x) 2 * x)



##### ESTER #####################################

# Pivot_wider. Até aqui vcs já tinham chegado
king_active <- KingxIBD %>% 
  select(ID1, ID2, kin) %>% 
  pivot_wider(id_cols = ID1,
              names_from = ID2,
              values_from = kin) %>% 
  column_to_rownames("ID1") 

# Mas a gente precisa organizar as linhas de acordo com as que tem menos NAs (ou seja, mais valores) até as mais esparças
king_active$n_vars <- rowSums(!is.na(king_active)) # cria uma coluna com a contagem de valores diferentes de NA
king_active <- king_active %>% 
  arrange(desc(n_vars)) %>% # Organiza pela coluna criada
  select(- n_vars) # Joga essa coluna fora

# Agora temos que fazer a mesma coisa com as colunas. Primeiro as mais vazias, depois as mais completas
col_order = sort(x = colSums(!is.na(king_active)),
                 decreasing = F) # cria um vetor com a contagem de valores diferentes de NA, onde os nomes são os nomes das colunas e os valores são a contagem
names_order = names(col_order) # Cria um vetor com os nomes já ordenados

# Organizando as colunas da mais esparsa para a menos esparsa
king_active <- king_active %>% 
  select(all_of(names_order)) %>% 
  rownames_to_column("ID1")

#################################################


### Seguindo da solução acima, da Ester.

# Teste Matrix totalmente quadrada:
M <- king_active

# Gerar vetor com tamanho de nrows:
col_NAs <- rep(NA, nrow(M))

# Copy matrix to a new one, pra garantir nao fazer besteira na original.
new_M <- M

# Append column above to Matrix:
new_M <- add_column(new_M, col_NAs, .after = 1)

# Rename column with proper sample ID.
colnames(new_M)[2] <- M[1,1]

# Gerar vetor com tamanho de ncols:
row_NAs <- rep(NA, ncol(new_M))

# Append column above after last row of Matrix:
new_M <- rbind(new_M, row_NAs)

# Renomear última row.
new_M[nrow(new_M), 1] <- colnames(new_M)[ncol(new_M)]

# Colocar 0.5 nas diagonais.
for (i in 1:nrow(new_M))
{
  new_M[i, i+1] <- 0.5
}

# Rebater matriz triangular de cima pra baixo
for (i in 2:ncol(new_M))
  for (j in i:nrow(new_M))
  {
    if(j == nrow(new_M)+1)
    {
      break
    }
    else {
      new_M[j, i] <- new_M[i-1, j+1]
    }
  }

# Set column "ID1" as index.
rownames(new_M) <- new_M$ID1
new_M$ID1 <- NULL


# Transform dataframe to matrix, and replace row names and column names.
M <- as.matrix(new_M)
rownames(M) <- colnames(M) <- colnames(new_M)
M <- pmax(M,0)
# Pro GCTA: Multiply all elements by 2.
M_x2 <- 2 * M

# Write to GCTA format.
output_grm=paste0(out_path, "/RKinship_for_GRM")
write_grm(output_grm, M_x2, shape="triangle")
