# =============================================================================
# figure 3A
# =============================================================================

library(data.table)
library(dplyr)


net_files <- c(
  "tnbc1_Regulator_contribution_summary.csv",
  "tnbc2_Regulator_contribution_summary.csv",
  "hccdc_Regulator_contribution_summary.csv",
  "hccmac_Regulator_contribution_summary.csv",
  "hccnk_Regulator_contribution_summary.csv",
  "tcell_hcc_Regulator_contribution_summary.csv",
  "tcell_nsclc_Regulator_contribution_summary.csv",
  "tcell_crc_Regulator_contribution_summary.csv",
  "pdac_Regulator_contribution_summary.csv"
)


use_fdr        <- TRUE  
pval_threshold <- 0.05   
das_threshold  <- 0.1    
das_cache_file <- "das_cache.rds"


build_prob_obj <- function(prob_df, sample_info, cell_name) {
  
  if (!"sample_id" %in% names(sample_info) && "file_accession" %in% names(sample_info)) {
    names(sample_info)[names(sample_info) == "file_accession"] <- "sample_id"
  }
  
  site_col    <- names(prob_df)[1]
  sample_cols <- setdiff(names(prob_df), site_col)
  
  common_samples <- intersect(sample_cols, sample_info$sample_id)
  prob_df        <- prob_df[, c(site_col, common_samples), drop = FALSE]
  
  ctrl_label <- "Non-specific target control"
  ctrl_ids   <- sample_info$sample_id[sample_info$target == ctrl_label]
  ctrl_ids   <- intersect(ctrl_ids, common_samples)
  
  rbp_list <- setdiff(unique(sample_info$target), ctrl_label)
  
  list(
    prop_mat    = prob_df,
    sample_cols = common_samples,
    ctrl_ids    = ctrl_ids,
    rbp_list    = rbp_list,
    cell        = cell_name
  )
}


preprocess_net <- function(net_file) {
  net   <- fread(net_file, data.table = FALSE, colClasses = "character")
  n_raw <- nrow(net)
  
  no_chr <- !grepl("^chr", net$AS_Site)
  if (any(no_chr)) {
    cat(sprintf("    [Preprocessing] Added chr prefix: %d rows\n", sum(no_chr)))
    net$AS_Site[no_chr] <- paste0("chr", net$AS_Site[no_chr])
  }
  
  cat(sprintf("    [Preprocessing] Original %d rows -> retained %d rows\n", n_raw, nrow(net)))
  
  out_path <- gsub("\\.csv$", "_processed.csv", net_file)
  fwrite(net, out_path, row.names = FALSE, quote = FALSE)
  cat(sprintf("    [Preprocessing] Saved: %s\n", out_path))
  
  net
}

net_list        <- list()
all_needed_rbps <- character(0)

for (nf in net_files) {
  cat(sprintf("\nPreprocessing: %s\n", nf))
  net             <- preprocess_net(nf)
  net_list[[nf]]  <- net
  all_needed_rbps <- union(all_needed_rbps, unique(net$RBP))
}
cat(sprintf("\nTotal RBPs involved across all net_files: %d\n", length(all_needed_rbps)))


k562_prob         <- fread("k562_AS_probability_matrix.csv",  data.table = FALSE)
sample_info_k562  <- fread("k526_rbp_file.csv",               data.table = FALSE)
hepg2_prob        <- fread("HepG2_AS_probability_matrix.csv", data.table = FALSE)
sample_info_hepg2 <- fread("hepg2_rbp_file.csv",              data.table = FALSE)

cat("Building K562 prob_obj ...\n")
k562_obj <- build_prob_obj(k562_prob, sample_info_k562, "K562")
cat(sprintf("  K562: %d sites, %d samples, %d control samples, %d available RBPs\n",
            nrow(k562_obj$prop_mat), length(k562_obj$sample_cols),
            length(k562_obj$ctrl_ids), length(k562_obj$rbp_list)))

cat("Building HEPG2 prob_obj ...\n")
hepg2_obj <- build_prob_obj(hepg2_prob, sample_info_hepg2, "HEPG2")
cat(sprintf("  HEPG2: %d sites, %d samples, %d control samples, %d available RBPs\n",
            nrow(hepg2_obj$prop_mat), length(hepg2_obj$sample_cols),
            length(hepg2_obj$ctrl_ids), length(hepg2_obj$rbp_list)))


k562_site_col  <- names(k562_obj$prop_mat)[1]
hepg2_site_col <- names(hepg2_obj$prop_mat)[1]
covered_sites  <- intersect(k562_obj$prop_mat[[k562_site_col]],
                            hepg2_obj$prop_mat[[hepg2_site_col]])
cat(sprintf("K562 ∩ HEPG2 covered sites total: %d\n", length(covered_sites)))


k562_obj$prop_mat  <- k562_obj$prop_mat[
  k562_obj$prop_mat[[k562_site_col]]   %in% covered_sites, , drop = FALSE]
hepg2_obj$prop_mat <- hepg2_obj$prop_mat[
  hepg2_obj$prop_mat[[hepg2_site_col]] %in% covered_sites, , drop = FALSE]


rbp_cache <- new.env(hash = TRUE)

if (file.exists(das_cache_file)) {
  cat("Found cache file:", das_cache_file, "\n")
  cached <- readRDS(das_cache_file)
  for (nm in names(cached)) {
    assign(nm, cached[[nm]], envir = rbp_cache, inherits = FALSE)
  }
  cat(sprintf("Loaded %d RBPs from DAS cache.\n", length(names(cached))))
  rm(cached)
} else {
  cat("No cache file found, will compute and save.\n")
}


compute_das_from_obj <- function(prob_obj, rbp_name, sample_info) {
  if (!"sample_id" %in% names(sample_info) && "file_accession" %in% names(sample_info)) {
    names(sample_info)[names(sample_info) == "file_accession"] <- "sample_id"
  }
  
  kd_ids   <- sample_info$sample_id[sample_info$target == rbp_name]
  kd_ids   <- intersect(kd_ids, prob_obj$sample_cols)
  ctrl_ids <- prob_obj$ctrl_ids
  
  if (length(kd_ids) == 0 || length(ctrl_ids) < 2) return(NULL)
  
  site_col <- names(prob_obj$prop_mat)[1]
  mat      <- prob_obj$prop_mat
  
  kd_m   <- as.matrix(mat[, kd_ids,   drop = FALSE])
  ctrl_m <- as.matrix(mat[, ctrl_ids, drop = FALSE])
  
  mean_kd   <- rowMeans(kd_m,   na.rm = TRUE)
  mean_ctrl <- rowMeans(ctrl_m, na.rm = TRUE)
  das_val   <- mean_kd - mean_ctrl
  
  pvals <- vapply(seq_len(nrow(mat)), function(i) {
    x <- kd_m[i, !is.na(kd_m[i, ])]
    y <- ctrl_m[i, !is.na(ctrl_m[i, ])]
    if (length(x) < 2 || length(y) < 2) return(NA_real_)
    alt <- if (mean_kd[i] >= mean_ctrl[i]) "greater" else "less"
    tryCatch(
      suppressWarnings(wilcox.test(x, y, alternative = alt, exact = FALSE)$p.value),
      error = function(e) NA_real_
    )
  }, numeric(1))
  
  data.frame(
    site      = mat[[site_col]],
    mean_kd   = mean_kd,
    mean_ctrl = mean_ctrl,
    das       = das_val,
    pvalue    = pvals,
    cell      = prob_obj$cell,
    stringsAsFactors = FALSE
  )
}


avail_k      <- k562_obj$rbp_list   
avail_h      <- hepg2_obj$rbp_list
compute_rbps <- intersect(all_needed_rbps, intersect(avail_k, avail_h))

new_rbps <- compute_rbps[
  !sapply(compute_rbps, exists, envir = rbp_cache, inherits = FALSE)
]

cat(sprintf("RBPs in net_files: %d | cached: %d | need compute: %d\n",
            length(compute_rbps),
            length(compute_rbps) - length(new_rbps),
            length(new_rbps)))

if (length(new_rbps) > 0) {
  for (i in seq_along(new_rbps)) {
    rbp <- new_rbps[i]
    
    parts <- Filter(Negate(is.null), list(
      compute_das_from_obj(k562_obj,  rbp, sample_info_k562),
      compute_das_from_obj(hepg2_obj, rbp, sample_info_hepg2)
    ))
    
    if (length(parts) > 0) {
      res     <- do.call(rbind, parts)
      res$rbp <- rbp
      res     <- res[!is.na(res$pvalue), ]
      res$fdr <- p.adjust(res$pvalue, method = "BH")
      assign(rbp, res, envir = rbp_cache, inherits = FALSE)
    } else {
      assign(rbp, NULL, envir = rbp_cache, inherits = FALSE)
    }
    
    if (i %% 10 == 0 || i == length(new_rbps)) {
      cat(sprintf("  Progress: %d / %d (current: %s)\n", i, length(new_rbps), rbp))
    }
  }
  
  cache_to_save <- Filter(Negate(is.null), as.list(rbp_cache))
  saveRDS(cache_to_save, das_cache_file)
  cat(sprintf("Cache updated -> %s (total %d RBPs)\n", das_cache_file, length(cache_to_save)))
} else {
  cat("All RBPs cached, skipping Wilcoxon calculation.\n")
}

all_das <- do.call(rbind, Filter(Negate(is.null),
                                 lapply(compute_rbps, function(r) {
                                   if (exists(r, envir = rbp_cache, inherits = FALSE))
                                     get(r, envir = rbp_cache, inherits = FALSE)
                                   else NULL
                                 })))

if (!is.null(all_das) && nrow(all_das) > 0) {
  all_das$key <- paste(all_das$rbp, all_das$site, sep = "_")
  cat(sprintf("all_das total rows: %d\n", nrow(all_das)))
} else {
  cat("Warning: all_das is empty, please check the data.\n")
}


all_results        <- list()
global_detail_list <- list()

for (nf in net_files) {
  cat(sprintf("\nProcessing: %s\n", nf))
  net         <- net_list[[nf]]
  net$key     <- paste(net$RBP, net$AS_Site, sep = "_")
  target_rbps <- intersect(unique(net$RBP), compute_rbps)
  
  cat(sprintf("  Target RBPs count: %d\n", length(target_rbps)))
  if (length(target_rbps) == 0 || is.null(all_das) || nrow(all_das) == 0) {
    cat("  Skipping (no target RBPs or no DAS results).\n")
    next
  }
  
  das_sub <- all_das[all_das$rbp %in% target_rbps, ]
  
  if (use_fdr) {
    sig <- das_sub %>%
      filter((fdr    < pval_threshold | abs(das) > das_threshold) &
               key %in% net$key)
  } else {
    sig <- das_sub %>%
      filter((pvalue < pval_threshold | abs(das) > das_threshold) &
               key %in% net$key)
  }
  
  
  sig_dedup <- sig %>%
    group_by(rbp, site, cell) %>%
    summarize(
      pvalue  = min(pvalue,   na.rm = TRUE),
      fdr     = min(fdr,      na.rm = TRUE),
      abs_das = max(abs(das), na.rm = TRUE),
      .groups = "drop"
    ) %>%
    as.data.frame()
  
  truth_counts <- if (nrow(sig_dedup) > 0) {
    sig_dedup %>%
      distinct(rbp, site) %>%
      count(rbp, name = "truth_affect")
  } else {
    data.frame(rbp = character(), truth_affect = integer(),
               stringsAsFactors = FALSE)
  }
  
  preds_counts <- net %>%
    filter(RBP %in% target_rbps, AS_Site %in% covered_sites) %>%
    distinct(RBP, AS_Site) %>%
    count(RBP, name = "preds") %>%
    rename(rbp = RBP)
  
  result <- merge(preds_counts, truth_counts, by = "rbp", all.x = TRUE)
  result$truth_affect[is.na(result$truth_affect)] <- 0
  result$support <- result$truth_affect / result$preds
  result <- result[order(result$rbp), c("rbp", "truth_affect", "preds", "support")]
  
  dir_label <- basename(dirname(nf))
  out_file  <- paste0(dir_label, "_support_no_direction.csv")
  fwrite(result, out_file, row.names = FALSE, quote = FALSE)
  cat(sprintf("  Mean support ratio: %.4f | result -> %s\n",
              mean(result$support, na.rm = TRUE), out_file))
  
  all_results[[nf]] <- result
  
  if (nrow(sig_dedup) > 0) {
    for (r in unique(sig_dedup$rbp)) {
      sub <- sig_dedup[sig_dedup$rbp == r,
                       c("site", "cell", "pvalue", "fdr", "abs_das")]
      global_detail_list[[r]] <- rbind(global_detail_list[[r]], sub)
    }
  }
}

if (length(all_results) > 0) {
  plot_df_step6 <- do.call(rbind, lapply(names(all_results), function(nf) {
    df           <- all_results[[nf]]
    df$data_name <- sub("_Regulator_contribution_summary\\.csv$", "", basename(nf))
    df
  }))
  plot_df_step6 <- plot_df_step6[, c("data_name", "rbp", "truth_affect", "preds", "support")]
  fwrite(plot_df_step6, "step6_support_plot.csv", row.names = FALSE, quote = FALSE)
  cat(sprintf("\nPlot data frame saved -> support_plot.csv (%d rows)\n",
              nrow(plot_df_step6)))
}


ggplot(plot, aes(x = reorder(data_name,support,decreasing=F), y = support, fill=data_name))+
  geom_violin(scale = "width") +
  stat_boxplot(geom = "errorbar",width=0.1) +
  geom_boxplot(size=0.4, width=0.15,outlier.fill=NA,outlier.color=NA)+ 
  ggtitle("RBP knockdown support ratio in different datasets")+
  theme_bw()+ 
  theme(axis.text.x=element_text(angle=30,colour="black",size=15), 
        axis.text.y=element_text(size=15,face="plain"), 
        axis.title.y=element_text(size = 15,face="plain"), 
        axis.title.x=element_text(size = 15,face="plain"), 
        plot.title = element_text(size=15,face="bold",hjust = 0.5), 
        panel.border = element_blank(),axis.line = element_line(colour = "black",size=1), 
        legend.text=element_text(face="italic", colour="black", 
                                 size=15),
        legend.title=element_text(face="italic", colour="black", 
                                  size=15),
        panel.grid.major = element_blank(),  
        panel.grid.minor = element_blank())+  
  ylab("Support ratio")+xlab("Datasets")



library(data.table)
library(dplyr)
library(tidyr)
library(purrr)
library(edgeR)
library(ggplot2)
library(ggpubr)


hepg_annot <- fread("lgb/hepg2_rbp_file.csv", data.table = FALSE)
k562_annot <- fread("lgb/k526_rbp_file.csv", data.table = FALSE)

k562_counts <- read.table("comp/trans/k562_trans_count_comp.csv", sep = ",", 
                          header = TRUE, stringsAsFactors = FALSE)
hepg_counts <- read.table("comp/trans/hep_trans_count_comp.csv", sep = ",", 
                          header = TRUE, stringsAsFactors = FALSE)
rownames(k562_counts) <- k562_counts$X; k562_counts <- k562_counts[, -1]
rownames(hepg_counts) <- hepg_counts$X; hepg_counts <- hepg_counts[, -1]

k562_cpm <- cpm(k562_counts, normalized.lib.size = TRUE)
hepg_cpm  <- cpm(hepg_counts,  normalized.lib.size = TRUE)

rownames(k562_cpm) <- sub("\\..*", "", rownames(k562_cpm))
rownames(hepg_cpm)  <- sub("\\..*", "", rownames(hepg_cpm))

deep_shap <- read.table("apply_the_model/output/explain_postar/df_DeepLIFT_knockout_tstat_TxRBPs.csv",
                        sep = ",", header = TRUE, stringsAsFactors = FALSE)
rownames(deep_shap) <- deep_shap$X; deep_shap <- deep_shap[, -1]

high_idx <- which(abs(deep_shap) > 10, arr.ind = TRUE)
deep_pairs <- data.frame(
  TF        = colnames(deep_shap)[high_idx[, "col"]],
  gene      = rownames(deep_shap)[high_idx[, "row"]],
  score     = deep_shap[high_idx],
  direction = ifelse(deep_shap[high_idx] > 0, "+", "-")
)


sfp_preds <- read.table("comp/sfp/exs_predict_reg.csv", sep = ",", 
                        header = TRUE, stringsAsFactors = FALSE)

colnames(sfp_preds)[colnames(sfp_preds) == "Event"] <- "RBP"


SAMAM_k562 <- read.table("comp/canonical/kdrbp_k562_das_num.csv", sep = ",", 
                         header = TRUE, stringsAsFactors = FALSE)
SAMAM_hepg <- read.table("comp/canonical/kdrbp_hepg2_das_num.csv", sep = ",", 
                         header = TRUE, stringsAsFactors = FALSE)


compute_diff_transcripts <- function(expr_matrix, annot_df, cell_label) {
  common_samples <- intersect(colnames(expr_matrix), annot_df$file_accession)
  expr_mat <- expr_matrix[, common_samples, drop = FALSE]
  
  ctrl_ids <- intersect(common_samples, 
                        annot_df$file_accession[annot_df$target == "Non-specific target control"])
  
  rbp_list <- setdiff(unique(annot_df$target), "Non-specific target control")
  
  result_list <- list()
  
  for (i in seq_along(rbp_list)) {
    rbp <- rbp_list[i]
    kd_ids <- intersect(common_samples, 
                        annot_df$file_accession[annot_df$target == rbp])
    if (length(kd_ids) == 0) next
    
    kd_mat   <- expr_mat[, kd_ids, drop = FALSE]
    ctrl_mat <- expr_mat[, ctrl_ids, drop = FALSE]
    
    significant_rows <- c()
    das_values <- c()
    kd_means   <- c()
    ctrl_means <- c()
    pvalues    <- c()
    
    for (j in seq_len(nrow(expr_mat))) {
      x <- as.numeric(kd_mat[j, ])
      y <- as.numeric(ctrl_mat[j, ])
      
      if (sum(!is.na(x)) < 2 || sum(!is.na(y)) < 2) next
      
      test_res <- wilcox.test(x, y, paired = FALSE, alternative = "two.sided", 
                              conf.level = 0.95)
      pv  <- test_res$p.value
      das <- mean(x, na.rm = TRUE) - mean(y, na.rm = TRUE)
      
      if (!is.nan(pv) && pv < 0.05) {
        significant_rows <- c(significant_rows, j)
        das_values <- c(das_values, das)
        kd_means   <- c(kd_means, mean(x, na.rm = TRUE))
        ctrl_means <- c(ctrl_means, mean(y, na.rm = TRUE))
        pvalues    <- c(pvalues, pv)
      }
    }
    
    if (length(significant_rows) > 0) {
      result_list[[rbp]] <- data.frame(
        site  = rownames(expr_mat)[significant_rows],
        das   = das_values,
        kd    = kd_means,
        ctrl  = ctrl_means,
        pvalue = pvalues,
        stringsAsFactors = FALSE
      )
    }
  }
  
  return(result_list)
}


cat("Computing differential transcripts for HEPG2...\n")
hepg_diff_list <- compute_diff_transcripts(hepg_cpm, hepg_annot, "HEPG2")
saveRDS(hepg_diff_list, "comp/trans/kdrbp_support_trans_detrans_hepg2.rds")

cat("Computing differential transcripts for K562...\n")
k562_diff_list <- compute_diff_transcripts(k562_cpm, k562_annot, "K562")
saveRDS(k562_diff_list, "comp/trans/kdrbp_support_trans_detrans_k562.rds")


calc_support <- function(diff_list, pred_df, target_col = "gene") {
  common_rbps <- intersect(names(diff_list), unique(pred_df$RBP))
  res <- lapply(common_rbps, function(r) {
    true_sites <- unique(diff_list[[r]]$site)
    pred_sites <- unique(pred_df[pred_df$RBP == r, target_col])
    if (length(pred_sites) == 0) return(data.frame(RBP = r, support = NA_real_))
    support <- length(intersect(true_sites, pred_sites)) / length(pred_sites)
    data.frame(RBP = r, support = support, stringsAsFactors = FALSE)
  })
  bind_rows(res)
}

deep_support_hepg <- calc_support(hepg_diff_list, deep_pairs, target_col = "gene")
deep_support_k562 <- calc_support(k562_diff_list, deep_pairs, target_col = "gene")

deep_support <- bind_rows(deep_support_hepg, deep_support_k562) %>%
  group_by(RBP) %>%
  summarise(support = mean(support, na.rm = TRUE), .groups = "drop") %>%
  mutate(method = "DeepRBP")


sfp_support_hepg <- calc_support(hepg_diff_list, sfp_preds, target_col = "trans")
sfp_support_k562 <- calc_support(k562_diff_list, sfp_preds, target_col = "trans")
sfp_support <- bind_rows(sfp_support_hepg, sfp_support_k562) %>%
  group_by(RBP) %>%
  summarise(support = mean(support, na.rm = TRUE), .groups = "drop") %>%
  mutate(method = "SFpointer")


SAMAM_hepg$support <- pmin(SAMAM_hepg$support, 1)
SAMAM_k562$support <- pmin(SAMAM_k562$support, 1)
SAMAM_support <- bind_rows(
  SAMAM_hepg %>% select(RBP, support),
  SAMAM_k562 %>% select(RBP, support)
) %>%
  group_by(RBP) %>%
  summarise(support = mean(support, na.rm = TRUE), .groups = "drop") %>%
  mutate(method = "SAMAM")


casrel_all <- read.table("mainfig_kdrbp_support_mean_regall.csv", sep = ",", 
                         header = TRUE, stringsAsFactors = FALSE)
casrel_support <- casrel_all %>%
  filter(method == "casrel") %>%
  select(RBP = rbp, support) %>%
  mutate(method = "casrel")

real_methods <- bind_rows(deep_support, sfp_support, SAMAM_support, casrel_support)


set.seed(123)
transcripts_pool <- union(rownames(k562_cpm), rownames(hepg_cpm))  

library(dplyr)
library(tidyr)
library(purrr)

randomize_and_calc <- function(real_pred_df, diff_list, target_col, background_pool, n_iterations = 100) {
  
  rbp_n <- real_pred_df %>%
    group_by(RBP) %>%
    summarise(n = n_distinct(!!sym(target_col)), .groups = "drop")
  
  single_random_run <- function(iteration_idx) {
    random_preds <- rbp_n %>%
      rowwise() %>%
      mutate(target = list(sample(background_pool, n, replace = FALSE))) %>%
      unnest(target) %>%
      rename(!!target_col := target) %>%
      ungroup()

    calc_support(diff_list, random_preds, target_col) %>%
      mutate(
        method = paste0(unique(real_pred_df$method)[1], "_random"),
        iteration = iteration_idx
      )
  }
  
  random_results_df <- map_dfr(1:n_iterations, single_random_run)
  
  return(random_results_df)
}

deep_random_dist <- randomize_and_calc(
  real_pred_df = deep_pairs %>% mutate(method = "DeepRBP"), 
  diff_list = hepg_diff_list, 
  target_col = "gene",
  background_pool = transcripts_pool,
  n_iterations = 1000
)

sfp_random_dist <- randomize_and_calc(
  real_pred_df = sfp_preds %>% mutate(method = "SFpointer"), 
  diff_list = hepg_diff_list, 
  target_col = "trans",
  background_pool = transcripts_pool,
  n_iterations = 1000
)


SAMAM_preds <- bind_rows(
  SAMAM_k562 %>% select(RBP, target = SE),  
  SAMAM_hepg %>% select(RBP, target = SE)
) %>% distinct()


splice_random_dist <- randomize_and_calc(
  real_pred_df = SAMAM_preds %>% mutate(method = "SFpointer"), 
  diff_list = hepg_diff_list, 
  target_col = "target",
  background_pool = transcripts_pool,
  n_iterations = 1000
)

all_random <- bind_rows(deep_random, sfp_random, splice_random)


final_plot_data <- bind_rows(real_methods, all_random)



ggplot(final_plot_data, aes(x = reorder(method, support, FUN = median, decreasing = TRUE),
                            y = support, fill = method)) +
  geom_violin(scale = "width", trim = FALSE, alpha = 0.8) +
  geom_boxplot(width = 0.15, fill = "white", outlier.shape = NA, size = 0.4) +
  stat_summary(fun = median, geom = "point", shape = 23, size = 2, fill = "red") +
  scale_fill_manual(values = c("#EE7600","#006400","#104E8B","#CD3333","#CDAA7D",
                               "gray60","gray70","gray80")) +
  ggtitle("Performance comparison of different methods: KD RBP") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 12),
        legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  stat_compare_means(comparisons = list(c('DeepRBP','CASREL'),c('SAMAM','CASREL')))+
  ylab("Support ratio") + xlab("Methods")


# =============================================================================
# figure 3B
# =============================================================================

truth <- read.table('revision_v2/human_binding_gene2.csv', sep = ',', header = T, stringsAsFactors = FALSE)
truth[1:5,]

net_files <- c(
  "tnbc1_Regulator_contribution_summary.csv",
  "tnbc2_Regulator_contribution_summary.csv",
  "hccdc_Regulator_contribution_summary.csv",
  "hccmac_Regulator_contribution_summary.csv",
  "hccnk_Regulator_contribution_summary.csv",
  "tcell_hcc_Regulator_contribution_summary.csv",
  "tcell_nsclc_Regulator_contribution_summary.csv",
  "tcell_crc_Regulator_contribution_summary.csv",
  "pdac_Regulator_contribution_summary.csv"
)

truth$sum <- paste(truth$RBP, truth$gene, sep = '_')


global_rbp_gene_list <- list()
bind_result <- list()
for (nf in net_files) {
  data_name <- sub("_Regulator_contribution_summary.*\\.csv$", "", basename(nf))
  cat(sprintf("\n[Analysis 1] processing: %s  (data_name = %s)\n", nf, data_name))
  
  pre <- read.table(nf, sep = ',', header = TRUE, stringsAsFactors = FALSE)
  pre$sum <- paste(pre$RBP, pre$Gene, sep = '_')
  pre <- pre[pre$RBP %in% truth$RBP, ]
  
  if (nrow(pre) == 0) {
    cat("  Warning: No matching RBP, skipping.\n")
    next
  }
  
  meanper <- length(intersect(pre$sum, truth$sum)) / length(unique(pre$sum))
  cat(sprintf("  Overall average support ratio: %.4f\n", meanper))
  cat(sprintf("  Number of unique RBPs: %d\n", length(unique(pre$RBP))))
  
  rbp_vec     <- unique(pre$RBP)
  bindpercent <- numeric(length(rbp_vec))
  
  for (i in seq_along(rbp_vec)) {
    r    <- rbp_vec[i]
    tru  <- truth$sum[truth$RBP == r]
    pred <- pre$sum[pre$RBP == r]
    bindpercent[i] <- length(intersect(pred, tru)) / length(unique(pred))
    
    inter_genes <- intersect(
      unique(pre$Gene[pre$RBP == r]),
      unique(truth$gene[truth$RBP == r])
    )
    if (length(inter_genes) > 0) {
      global_rbp_gene_list[[r]] <- union(global_rbp_gene_list[[r]], inter_genes)
    }
  }
  
  temp     <- data.frame(rbp = rbp_vec, percent = bindpercent, stringsAsFactors = FALSE)
  bind_result[[data_name]] <- temp
  out_file <- file.path(dirname(nf), paste0(data_name, "_binding_perform_percent_gene.csv"))
  write.csv(temp, out_file, row.names = FALSE, quote = FALSE)
  cat(sprintf("  Per-RBP support ratio saved: %s\n", out_file))
}


if (length(bind_result) > 0) {
  plot_df_bind <- do.call(rbind, lapply(names(bind_result), function(nf) {
    df           <- bind_result[[nf]]
    df$data_name <- sub("_Regulator_contribution_summary_acc85\\.csv$", "", basename(nf))
    df
  }))
  plot_df_bind <- plot_df_bind[, c("data_name", "rbp", "percent")]
  fwrite(plot_df_bind, "binding_support_plot.csv", row.names = FALSE, quote = FALSE)
  cat(sprintf("\nStep 6 plot data frame saved -> step6_support_plot.csv (%d rows)\n",
              nrow(plot_df_bind)))
}

plot_df_bind[1:5,]


ggplot(plot_df_bind, aes(x = reorder(data_name,percent,decreasing=F), y = percent, fill=data_name))+
  geom_violin(scale = "width") +
  stat_boxplot(geom = "errorbar",width=0.1) +
  geom_boxplot(size=0.4, width=0.15,outlier.fill=NA,outlier.color=NA)+ 
  ggtitle("RBP binding support ratio in different datasets")+
  theme_bw()+ 
  theme(axis.text.x=element_text(angle=30,colour="black",size=15), 
        axis.text.y=element_text(size=15,face="plain"), 
        axis.title.y=element_text(size = 15,face="plain"), 
        axis.title.x=element_text(size = 15,face="plain"), 
        plot.title = element_text(size=15,face="bold",hjust = 0.5), 
        panel.border = element_blank(),axis.line = element_line(colour = "black",size=1), 
        legend.text=element_text(face="italic", colour="black", 
                                 size=15),
        legend.title=element_text(face="italic", colour="black", 
                                  size=15),
        panel.grid.major = element_blank(),  
        panel.grid.minor = element_blank())+  
  ylab("Binding ratio")+xlab("Datasets") 

write.csv(plot_df_bind, 'preds/preds/binding_plot.csv', row.names = F, quote = F)



# =============================================================================
# figure 3C
# =============================================================================


library(dplyr)
library(ggplot2)

datasets <- list(
  tnbc2   = list(folder = "lgbm/tnbc2",   name = "TNBC-2"),
  tnbc1   = list(folder = "lgbm/tnbc1",   name = "TNBC-1"),
  tc_hcc  = list(folder = "lgbm/tc_hcc",  name = "Tcell(HCC)"),
  tc_crc  = list(folder = "lgbm/tc_crc",  name = "Tcell(CRC)"),
  tc_nsclc= list(folder = "lgbm/tc_nsclc",name = "Tcell(NSCLC)"),
  hccmac  = list(folder = "lgbm/hccmac",  name = "Macrophage(HCC)"),
  hccdc   = list(folder = "lgbm/hccdc",   name = "DC(HCC)"),
  pdac    = list(folder = "lgbm/pdac",    name = "PDAC(10X)"),
  hccnk   = list(folder = "lgbm/hccnk",   name = "NKcell(HCC)")
)


process_dataset <- function(folder, method_name, k_values = 1:5) {
  file_paths <- file.path(folder, paste0("k", k_values, "_lgb_accuracy.csv"))
  exist_files <- file_paths[file.exists(file_paths)]
  if (length(exist_files) == 0) {
    warning("No accuracy files found in ", folder)
    return(NULL)
  }
  
  
  acc_list <- lapply(exist_files, function(f) {
    df <- read.table(f, sep = ",", header = TRUE, stringsAsFactors = FALSE)
    return(df$accuracy)
  })
  acc_mat <- do.call(cbind, acc_list)
  mean_acc <- rowMeans(acc_mat, na.rm = TRUE)
  valid_idx <- !is.na(mean_acc)
  mean_acc <- mean_acc[valid_idx]
  
  data.frame(acc = mean_acc, method = method_name, stringsAsFactors = FALSE)
}

ras_list <- lapply(names(datasets), function(key) {
  process_dataset(datasets[[key]]$folder, datasets[[key]]$name)
})

casrel <- do.call(rbind, ras_list)


deeprbp <- read.table("comp/deeprbp/TCGA_MSE.csv", sep = ",", header = TRUE, stringsAsFactors = FALSE)
deeprbp <- data.frame(
  acc = 1 - deeprbp$MSE,
  method = "DeepRBP",
  stringsAsFactors = FALSE
)


plot_data <- rbind(casrel, deeprbp)


ggplot(plot_data, aes(x = reorder(method, acc, decreasing = FALSE), 
                      y = acc, fill = method)) +
  geom_violin(scale = "width") +
  stat_boxplot(geom = "errorbar", width = 0.1) +
  geom_boxplot(size = 0.4, width = 0.15, outlier.fill = NA, outlier.color = NA) +
  ggtitle("AS event prediction accuracy by RBP expression") +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 30, colour = "black", size = 15),
    axis.text.y = element_text(size = 15, face = "plain"),
    axis.title.y = element_text(size = 15, face = "plain"),
    axis.title.x = element_text(size = 15, face = "plain"),
    plot.title = element_text(size = 15, face = "bold", hjust = 0.5),
    panel.border = element_blank(),
    axis.line = element_line(colour = "black", size = 1),
    legend.text = element_text(face = "italic", colour = "black", size = 15),
    legend.title = element_text(face = "italic", colour = "black", size = 15),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  ylab("Accuracy") + xlab("Methods")