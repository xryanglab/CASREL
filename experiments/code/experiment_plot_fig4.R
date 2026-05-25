# ==============================================================================
# Figure 4 A
# ==============================================================================

library(ComplexHeatmap)
library(circlize)
library(ggplot2)
library(ggrepel)
library(VennDetail)

hcccd4 = read.table("lgb/hcc_cd4sum/summary/Regulator_contribution_summary.csv", header = T, sep = ',', stringsAsFactors = F)
hcccd8 = read.table("lgb/hcc_cd8sum/summary/Regulator_contribution_summary.csv", header = T, sep = ',', stringsAsFactors = F)
crccd4 = read.table("lgb/crc_cd4sum/summary/Regulator_contribution_summary.csv", header = T, sep = ',', stringsAsFactors = F)
crccd8 = read.table("lgb/crc_cd8sum/summary/Regulator_contribution_summary.csv", header = T, sep = ',', stringsAsFactors = F)

hcccd4$sum = paste(hcccd4$RBP, hcccd4$AS, hcccd4$Direction, sep = '_')
hcccd8$sum = paste(hcccd8$RBP, hcccd8$AS, hcccd8$Direction, sep = '_')
crccd4$sum = paste(crccd4$RBP, crccd4$AS, crccd4$Direction, sep = '_')
crccd8$sum = paste(crccd8$RBP, crccd8$AS, crccd8$Direction, sep = '_')


sets <- list(
  HCC_CD4 = hcccd4$sum,
  HCC_CD8 = hcccd8$sum,
  CRC_CD4 = crccd4$sum,
  CRC_CD8 = crccd8$sum
)

n <- length(sets)
names_vec <- names(sets)
celltype_consistency <- matrix(NA, nrow = n, ncol = n, dimnames = list(names_vec, names_vec))

for (i in 1:n) {
  for (j in 1:n) {
    inter_size <- length(intersect(sets[[i]], sets[[j]]))
    celltype_consistency[i, j] <- inter_size / length(sets[[i]])  
  }
}


dataset_labels <- c("CD4+T (CRC)", "CD4+T (HCC)", "CD8+T (HCC)", "CD8+T (CRC)")
rownames(celltype_consistency) <- dataset_labels
colnames(celltype_consistency) <- dataset_labels


col_fun_consistency <- colorRamp2(c(75, 82.5, 90), c('#3777b8', 'white', '#b52d2d'))

# Generate Heatmap
ht <- Heatmap(celltype_consistency, 
              name = "Consistency", 
              column_title = "Consistency of predicted RBP-AS circuits", 
              col = col_fun_consistency,
              cluster_rows = FALSE, 
              cluster_columns = FALSE, 
              show_row_names = TRUE, 
              show_column_names = TRUE,  
              row_names_side = "right",
              column_names_rot = 90)
draw(ht)

# ==============================================================================
# Figure 4B
# Schematic diagram: alternative splicing of STAT3 
# ==============================================================================


# ==============================================================================
# Figure 4C-F
# ==============================================================================

color_map <- c("Literature" = "#c82423", "KD RBP" = "#2878b5", "Other" = "grey60")

# Base theme for all scatter plots to match the figure's aesthetic
scatter_theme <- theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.title = element_text(size = 12, face = "bold"),
    panel.background = element_rect(fill = "#e8eef5", color = NA),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = c(0.85, 0.25),
    legend.background = element_rect(fill = "white", color = "black", size = 0.3),
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 9)
  )

# Load scatter plot data
rbp_cd4 <- read.table('plot_stat3/CD4_T_scatter.csv', sep = ',', header = TRUE, stringsAsFactors = FALSE)
rbp_cd8 <- read.table('plot_stat3/CD8_T_scatter.csv', sep = ',', header = TRUE, stringsAsFactors = FALSE)
rbp_crc <- read.table('plot_stat3/CRC_T_scatter.csv', sep = ',', header = TRUE, stringsAsFactors = FALSE)
rbp_hcc <- read.table('plot_stat3/HCC_T_scatter.csv', sep = ',', header = TRUE, stringsAsFactors = FALSE)

# Panel C: STAT3 regulation in CRC dataset
p_crc <- ggplot(rbp_crc, aes(x = crc_cd4, y = crc_cd8)) +
  geom_hline(yintercept = 0, color = "black", size = 0.5) +
  geom_vline(xintercept = 0, color = "black", size = 0.5) +
  geom_point(aes(color = type), size = 3) +
  geom_text_repel(aes(label = rbp), size = 3.5, max.overlaps = 20) +
  scale_color_manual(values = color_map) +
  labs(title = expression(bold("STAT3" * beta * " regulation in CRC dataset")),
       x = "SHAP value (CD4+ T cells)",
       y = "SHAP value (CD8+ T cells)",
       color = "Validation") +
  scatter_theme

print(p_crc)

# Panel C: STAT3 regulation in HCC dataset
p_hcc <- ggplot(rbp_hcc, aes(x = hcc_cd4, y = hcc_cd8)) +
  geom_hline(yintercept = 0, color = "black", size = 0.5) +
  geom_vline(xintercept = 0, color = "black", size = 0.5) +
  geom_point(aes(color = type), size = 3) +
  geom_text_repel(aes(label = rbp), size = 3.5, max.overlaps = 20) +
  scale_color_manual(values = color_map) +
  labs(title = expression(bold("STAT3" * beta * " regulation in HCC dataset")),
       x = "SHAP value (CD4+ T cells)",
       y = "SHAP value (CD8+ T cells)",
       color = "Validation") +
  scatter_theme

print(p_hcc)

# Panel E: STAT3 regulation in CD8+ T cells
p_cd8 <- ggplot(rbp_cd8, aes(x = crc_cd8, y = hcc_cd8)) +
  geom_hline(yintercept = 0, color = "black", size = 0.5) +
  geom_vline(xintercept = 0, color = "black", size = 0.5) +
  geom_point(aes(color = type), size = 3) +
  geom_text_repel(aes(label = rbp), size = 3.5, max.overlaps = 20) +
  scale_color_manual(values = color_map) +
  labs(title = expression(bold("STAT3" * beta * " regulation in CD8+ T cells")),
       x = "SHAP value (CRC)",
       y = "SHAP value (HCC)",
       color = "Validation") +
  scatter_theme

print(p_cd8)

# Panel F: STAT3 regulation in CD4+ T cells
p_cd4 <- ggplot(rbp_cd4, aes(x = crc_cd4, y = hcc_cd4)) +
  geom_hline(yintercept = 0, color = "black", size = 0.5) +
  geom_vline(xintercept = 0, color = "black", size = 0.5) +
  geom_point(aes(color = type), size = 3) +
  geom_text_repel(aes(label = rbp), size = 3.5, max.overlaps = 20) +
  scale_color_manual(values = color_map) +
  labs(title = expression(bold("STAT3" * beta * " regulation in CD4+ T cells")),
       x = "SHAP value (CRC)",
       y = "SHAP value (HCC)",
       color = "Validation") +
  scatter_theme

print(p_cd4)