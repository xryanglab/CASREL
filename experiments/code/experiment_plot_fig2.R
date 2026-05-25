# =============================================================================
# figure 2 A
# =============================================================================

library(dplyr)

find_knee_point <- function(importance_sorted_desc) {
  arr <- as.numeric(importance_sorted_desc)
  n <- length(arr)
  if (n <= 1) return(1)
  if (n == 2) return(ifelse(arr[1] >= arr[2], 1, 2))
  
  total <- sum(arr)
  if (total <= 0) return(1)
  
  cum_frac <- cumsum(arr) / total
  x_norm <- seq(0, 1, length.out = n)
  distances <- cum_frac - x_norm
  
  return(which.max(distances))
}


filter_single_column <- function(data, value_col, direction,
                                 method, cumulative_pct,
                                 contribution_filter, verbose) {
  df <- data %>%
    transmute(
      RBP = .data$rbp,
      original = .data[[value_col]],
      abs_val = abs(original)
    ) %>%
    filter(!is.na(abs_val)) %>%
    arrange(desc(abs_val))
  
  if (nrow(df) == 0) {
    if (verbose) message(sprintf("  [%s] No non-missing values, skipped.", direction))
    return(data.frame())
  }
  
  abs_vec <- df$abs_val
  n <- length(abs_vec)
  total_abs <- sum(abs_vec)
  
  if (total_abs <= 0) {
    if (verbose) message(sprintf("  [%s] Sum of absolute values is 0, skipped.", direction))
    return(data.frame())
  }
  
  
  if (method == "knee") {
    cutoff_pos <- find_knee_point(abs_vec)
  } else if (method == "cumulative") {
    cum_frac <- cumsum(abs_vec) / total_abs
    above <- which(cum_frac >= cumulative_pct)
    cutoff_pos <- if (length(above) > 0) above[1] else n
  } else {
    stop(paste0("Unknown method: ", method))
  }
  
  cutoff_pos <- max(1, cutoff_pos)
  selected <- df[1:cutoff_pos, ]
  thresh_abs <- selected$abs_val[cutoff_pos]
  cum_contrib <- sum(selected$abs_val) / total_abs
  
  if (verbose) {
    message(sprintf("  [%s] Single-column filtering on |%s| (%s):",
                    direction, value_col, method))
    message(sprintf("    Total RBPs: %d -> Selected: %d", n, cutoff_pos))
    message(sprintf("    Threshold (absolute value): %.4f", thresh_abs))
    message(sprintf("    Cumulative |%s|: %.1f%%", value_col, cum_contrib * 100))
  }
  
  
  contributions <- selected$abs_val
  if (contribution_filter != "none" && length(contributions) > 0) {
    if (contribution_filter == "knee") {
      knee_idx <- find_knee_point(sort(contributions, decreasing = TRUE))
      min_contrib <- contributions[knee_idx]
      if (verbose) message(sprintf("    Contribution knee filter: %.4f", min_contrib))
    } else if (startsWith(contribution_filter, "quantile:")) {
      q <- as.numeric(sub("quantile:", "", contribution_filter))
      if (is.na(q) || q < 0 || q > 1) {
        warning("Invalid quantile value, skipping contribution filter.")
        min_contrib <- 0
      } else {
        min_contrib <- quantile(contributions, probs = q)
        if (verbose) message(sprintf("    Contribution quantile (%.2f) filter: %.4f", q, min_contrib))
      }
    } else {
      warning(paste0("Unknown contribution_filter: ", contribution_filter, ". Skipping."))
      min_contrib <- 0
    }
    if (min_contrib > 0) {
      before_n <- nrow(selected)
      selected <- selected %>% filter(abs_val >= min_contrib)
      if (verbose) {
        message(sprintf("    Post-filter: %d -> %d rows", before_n, nrow(selected)))
      }
    }
  }
  
  if (nrow(selected) == 0) return(data.frame())
  
  result <- selected %>%
    mutate(
      Direction = direction,
      Contribution = abs_val
    ) %>%
    select(RBP, !!value_col := original, Contribution, Direction) %>%
    arrange(desc(Contribution))
  
  return(result)
}


filter_regulators_shap <- function(
  data,
  rbp_col = "rbp",
  shap_inhibit_col = "aggregated_shap_inhibit",
  shap_exhibit_col = "aggregated_shap_exhibit",
  method = "knee",
  cumulative_pct = 0.80,
  contribution_filter = "knee",
  separate = FALSE,          
  verbose = TRUE
) {
  
  if (!(rbp_col %in% names(data))) stop("rbp column not found.")
  data$rbp <- data[[rbp_col]]
  
  if (!separate) {
    
    data <- data %>%
      mutate(
        importance = abs(.data[[shap_inhibit_col]] - .data[[shap_exhibit_col]])
      ) %>%
      arrange(desc(importance))
    
    total_importance <- sum(data$importance)
    if (total_importance <= 0) {
      if (verbose) message("Total importance is 0, no regulators to select.")
      return(data.frame())
    }
    
    imp_arr <- data$importance
    n <- nrow(data)
    
    if (method == "knee") {
      cutoff_pos <- find_knee_point(imp_arr)
    } else if (method == "cumulative") {
      cum_frac <- cumsum(imp_arr) / total_importance
      above <- which(cum_frac >= cumulative_pct)
      cutoff_pos <- if (length(above) > 0) above[1] else n
    } else {
      stop(paste0("Unknown method: ", method))
    }
    
    cutoff_pos <- max(1, cutoff_pos)
    selected <- data[1:cutoff_pos, ]
    adaptive_thresh <- selected$importance[cutoff_pos]
    cum_contrib <- sum(imp_arr[1:cutoff_pos]) / total_importance
    
    if (verbose) {
      message(sprintf("Adaptive threshold (%s):", method))
      message(sprintf("  Total RBPs: %d -> Selected: %d", n, cutoff_pos))
      message(sprintf("  Threshold value: %.4f", adaptive_thresh))
      message(sprintf("  Cumulative contribution: %.1f%%", cum_contrib * 100))
    }
    
    selected <- selected %>%
      mutate(
        Direction = ifelse(
          .data[[shap_inhibit_col]] > .data[[shap_exhibit_col]], "-", "+"
        ),
        Contribution = importance
      )
    
    
    if (contribution_filter != "none" && nrow(selected) > 0) {
      contributions <- selected$Contribution
      if (contribution_filter == "knee") {
        knee_idx <- find_knee_point(sort(contributions, decreasing = TRUE))
        min_contrib <- contributions[knee_idx]
        if (verbose) message(sprintf("  Contribution knee filter: %.4f", min_contrib))
      } else if (startsWith(contribution_filter, "quantile:")) {
        q <- as.numeric(sub("quantile:", "", contribution_filter))
        if (is.na(q) || q < 0 || q > 1) {
          warning("Invalid quantile value, skipping contribution filter.")
          min_contrib <- 0
        } else {
          min_contrib <- quantile(contributions, probs = q)
          if (verbose) message(sprintf("  Contribution quantile (%.2f) filter: %.4f", q, min_contrib))
        }
      } else {
        warning(paste0("Unknown contribution_filter: ", contribution_filter, ". Skipping."))
        min_contrib <- 0
      }
      if (min_contrib > 0) {
        before_n <- nrow(selected)
        selected <- selected %>% filter(Contribution >= min_contrib)
        if (verbose) message(sprintf("  Contribution filter: %d -> %d rows", before_n, nrow(selected)))
      }
    }
    
    result <- selected %>%
      select(
        RBP = !!sym(rbp_col),
        Contribution,
        Direction,
        shap_inhibit = !!sym(shap_inhibit_col),
        shap_exhibit = !!sym(shap_exhibit_col)
      ) %>%
      arrange(desc(Contribution))
    
    return(result)
    
  } else {
    
    if (verbose) message("=== Separate mode: filtering by absolute value of each column independently ===")
    
    inhibit_res <- filter_single_column(
      data = data,
      value_col = shap_inhibit_col,
      direction = "-",
      method = method,
      cumulative_pct = cumulative_pct,
      contribution_filter = contribution_filter,
      verbose = verbose
    )
    
    promote_res <- filter_single_column(
      data = data,
      value_col = shap_exhibit_col,
      direction = "+",
      method = method,
      cumulative_pct = cumulative_pct,
      contribution_filter = contribution_filter,
      verbose = verbose
    )
    
    
    if (nrow(inhibit_res) > 0) {
      inhibit_res <- inhibit_res %>%
        left_join(data %>% select(rbp, !!shap_exhibit_col := .data[[shap_exhibit_col]]),
                  by = c("RBP" = "rbp"))
    }
    if (nrow(promote_res) > 0) {
      promote_res <- promote_res %>%
        left_join(data %>% select(rbp, !!shap_inhibit_col := .data[[shap_inhibit_col]]),
                  by = c("RBP" = "rbp"))
    }
    
    return(list(inhibit = inhibit_res, promote = promote_res))
  }
}


filter_regulators_reg <- function(
  data,
  rbp_col = "rbp",
  inhibit_col = "Inhibit",
  promote_col = "Promote",
  method = "knee",
  cumulative_pct = 0.80,
  contribution_filter = "knee",
  separate = FALSE,
  verbose = TRUE
) {
  if (!separate) {
    result <- filter_regulators_shap(
      data = data,
      rbp_col = rbp_col,
      shap_inhibit_col = inhibit_col,
      shap_exhibit_col = promote_col,
      method = method,
      cumulative_pct = cumulative_pct,
      contribution_filter = contribution_filter,
      separate = FALSE,
      verbose = verbose
    )
    if (nrow(result) > 0) {
      result <- result %>% rename(Inhibit = shap_inhibit, Promote = shap_exhibit)
    }
    return(result)
  } else {
    result_list <- filter_regulators_shap(
      data = data,
      rbp_col = rbp_col,
      shap_inhibit_col = inhibit_col,
      shap_exhibit_col = promote_col,
      method = method,
      cumulative_pct = cumulative_pct,
      contribution_filter = contribution_filter,
      separate = TRUE,
      verbose = verbose
    )
    
    if (nrow(result_list$inhibit) > 0) {
      result_list$inhibit <- result_list$inhibit %>%
        rename(Inhibit = !!sym(inhibit_col),
               Promote = !!sym(promote_col))
    }
    if (nrow(result_list$promote) > 0) {
      result_list$promote <- result_list$promote %>%
        rename(Inhibit = !!sym(inhibit_col),
               Promote = !!sym(promote_col))
    }
    return(result_list)
  }
}


shap_single_lgbm = read.table('lgbm/shap_analysis_cd44.csv', sep = ',',  header = T, stringsAsFactors = F)
shap_single_fcn = read.table('fcn/shap_analysis_cd44.csv', sep = ',',  header = T, stringsAsFactors = F)
shap_single_lr = read.table('lr/shap_analysis_cd44.csv', sep = ',',  header = T, stringsAsFactors = F)


fcn_separate <- filter_regulators_shap(
  data = shap_single_fcn,
  rbp_col = "rbp",
  shap_inhibit_col = "aggregated_shap_inhibit",
  shap_exhibit_col = "aggregated_shap_exhibit",
  method = "knee",
  contribution_filter = "knee",
  separate = TRUE,
  verbose = TRUE
)

lgbm_separate <- filter_regulators_shap(
  data = shap_single_lgbm,
  rbp_col = "rbp",
  shap_inhibit_col = "aggregated_shap_inhibit",
  shap_exhibit_col = "aggregated_shap_exhibit",
  method = "knee",
  contribution_filter = "knee",
  separate = TRUE,
  verbose = TRUE
)

reg_separate <- filter_regulators_reg(
  data = shap_single_lr,
  rbp_col = "rbp",
  inhibit_col = "Inhibit",
  promote_col = "Promote",
  method = "knee",
  contribution_filter = "knee",
  separate = TRUE,         
  verbose = TRUE
)


fcn1 = data.frame(X = fcn_separate$inhibit$RBP, shap = fcn_separate$inhibit$aggregated_shap_inhibit, type = 'Inhibit', stringsAsFactors = F)
fcn2 = data.frame(X = fcn_separate$inhibit$RBP, shap = fcn_separate$inhibit$aggregated_shap_exhibit, type = 'Promote', stringsAsFactors = F)

fcn = rbind(fcn1, fcn2)

ggplot(fcn, aes(x = shap,y = reorder(X, shap, function(x) sum(x)),fill = type))+
  geom_bar(stat ="identity",width = 0.6,position ="stack")+     
  scale_fill_manual(values = c("RoyalBlue","IndianRed3"))+              
  labs(x = "SHAP",y = "RBP", title = "FCN: Contribution of each RBP to CD44 splicing (inhibit)")+    
  guides(fill = guide_legend(reverse = F))+               
  theme_bw()+ 
  theme(
    axis.text.x=element_text(colour="black",size=10), 
    axis.text.y=element_text(size=15,face="plain"), 
    axis.title.y=element_text(size = 15,face="plain"), 
    axis.title.x=element_text(size = 15,face="plain"), 
    plot.title = element_text(size=15,face="bold",hjust = 0.5), 
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank())

fcn1 = data.frame(X = fcn_separate$promote$RBP, shap = fcn_separate$promote$aggregated_shap_inhibit, type = 'Inhibit', stringsAsFactors = F)
fcn2 = data.frame(X = fcn_separate$promote$RBP, shap = fcn_separate$promote$aggregated_shap_exhibit, type = 'Promote', stringsAsFactors = F)

fcn = rbind(fcn1, fcn2)

ggplot(fcn, aes(x = shap,y = reorder(X, shap, function(x) sum(x)),fill = type))+
  geom_bar(stat ="identity",width = 0.6,position ="stack")+     
  scale_fill_manual(values = c("RoyalBlue","IndianRed3"))+              
  labs(x = "SHAP",y = "RBP", title = "FCN: Contribution of each RBP to CD44 splicing (promote)")+    
  guides(fill = guide_legend(reverse = F))+               
  theme_bw()+ 
  theme(
    axis.text.x=element_text(colour="black",size=10), 
    axis.text.y=element_text(size=15,face="plain"), 
    axis.title.y=element_text(size = 15,face="plain"), 
    axis.title.x=element_text(size = 15,face="plain"), 
    plot.title = element_text(size=15,face="bold",hjust = 0.5), 
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank())


lgbm1 = data.frame(X = lgbm_separate$inhibit$RBP, shap = lgbm_separate$inhibit$aggregated_shap_inhibit, type = 'Inhibit', stringsAsFactors = F)
lgbm2 = data.frame(X = lgbm_separate$inhibit$RBP, shap = lgbm_separate$inhibit$aggregated_shap_exhibit, type = 'Promote', stringsAsFactors = F)

lgbm = rbind(lgbm1, lgbm2)

ggplot(lgbm, aes(x = shap,y = reorder(X, shap, function(x) sum(x)),fill = type))+
  geom_bar(stat ="identity",width = 0.6,position ="stack")+     
  scale_fill_manual(values = c("RoyalBlue","IndianRed3"))+              
  labs(x = "SHAP",y = "RBP", title = "LGBM: Contribution of each RBP to CD44 splicing (inhibit)")+    
  guides(fill = guide_legend(reverse = F))+               
  theme_bw()+ 
  theme(
    axis.text.x=element_text(colour="black",size=10), 
    axis.text.y=element_text(size=15,face="plain"), 
    axis.title.y=element_text(size = 15,face="plain"), 
    axis.title.x=element_text(size = 15,face="plain"), 
    plot.title = element_text(size=15,face="bold",hjust = 0.5), 
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank())

lgbm1 = data.frame(X = lgbm_separate$promote$RBP, shap = lgbm_separate$promote$aggregated_shap_inhibit, type = 'Inhibit', stringsAsFactors = F)
lgbm2 = data.frame(X = lgbm_separate$promote$RBP, shap = lgbm_separate$promote$aggregated_shap_exhibit, type = 'Promote', stringsAsFactors = F)

lgbm = rbind(lgbm1, lgbm2)

ggplot(lgbm, aes(x = shap,y = reorder(X, shap, function(x) sum(x)),fill = type))+
  geom_bar(stat ="identity",width = 0.6,position ="stack")+     
  scale_fill_manual(values = c("RoyalBlue","IndianRed3"))+              
  labs(x = "SHAP",y = "RBP", title = "LGBM: Contribution of each RBP to CD44 splicing (promote)")+    
  guides(fill = guide_legend(reverse = F))+               
  theme_bw()+ 
  theme(
    axis.text.x=element_text(colour="black",size=10), 
    axis.text.y=element_text(size=15,face="plain"), 
    axis.title.y=element_text(size = 15,face="plain"), 
    axis.title.x=element_text(size = 15,face="plain"), 
    plot.title = element_text(size=15,face="bold",hjust = 0.5), 
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank())


lr1 = data.frame(X = reg_separate$inhibit$RBP, shap = reg_separate$inhibit$Inhibit, type = 'Promote', stringsAsFactors = F)
lr2 = data.frame(X = reg_separate$inhibit$RBP, shap = reg_separate$inhibit$Promote, type = 'Inhibit', stringsAsFactors = F)

lr = rbind(lr1, lr2)

ggplot(lr, aes(x = shap,y = reorder(X, shap, function(x) sum(x)),fill = type))+
  geom_bar(stat ="identity",width = 0.6,position ="stack")+     
  scale_fill_manual(values = c("RoyalBlue","IndianRed3"))+              
  labs(x = "SHAP",y = "RBP", title = "LR: Contribution of each RBP to CD44 splicing (inhibit)")+    
  guides(fill = guide_legend(reverse = F))+               
  theme_bw()+ 
  theme(
    axis.text.x=element_text(colour="black",size=10), 
    axis.text.y=element_text(size=15,face="plain"), 
    axis.title.y=element_text(size = 15,face="plain"), 
    axis.title.x=element_text(size = 15,face="plain"), 
    plot.title = element_text(size=15,face="bold",hjust = 0.5), 
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank())


lr1 = data.frame(X = reg_separate$promote$RBP, shap = reg_separate$promote$Inhibit, type = 'Inhibit', stringsAsFactors = F)
lr2 = data.frame(X = reg_separate$promote$RBP, shap = reg_separate$promote$Promote, type = 'Promote', stringsAsFactors = F)

lr = rbind(lr1, lr2)

ggplot(lr, aes(x = shap,y = reorder(X, shap, function(x) sum(x)),fill = type))+
  geom_bar(stat ="identity",width = 0.6,position ="stack")+     
  scale_fill_manual(values = c("RoyalBlue","IndianRed3"))+              
  labs(x = "SHAP",y = "RBP", title = "LR: Contribution of each RBP to CD44 splicing (promote)")+    
  guides(fill = guide_legend(reverse = F))+               
  theme_bw()+ 
  theme(
    axis.text.x=element_text(colour="black",size=10), 
    axis.text.y=element_text(size=15,face="plain"), 
    axis.title.y=element_text(size = 15,face="plain"), 
    axis.title.x=element_text(size = 15,face="plain"), 
    plot.title = element_text(size=15,face="bold",hjust = 0.5), 
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank())



# =============================================================================
# figure 2 B
# =============================================================================


lgb = read.table('tnbc1/lgbm_shap_summary.csv', sep = ',',  header = T, stringsAsFactors = F)
reg = read.table('tnbc1/lr_summary.csv', sep = ',',  header = T, stringsAsFactors = F)
fcn = read.table('tnbc1/fcn_summary.csv', sep = ',',  header = T, stringsAsFactors = F)

fcn$aggregated_shap_inhibit[which(abs(fcn$aggregated_shap_inhibit) < 1)] = 0
fcn$aggregated_shap_exhibit[which(abs(fcn$aggregated_shap_exhibit) < 1)] = 0
fcn = fcn[-which(fcn$aggregated_shap_inhibit == 0 & fcn$aggregated_shap_exhibit == 0),]
lgb$aggregated_shap_inhibit[which(abs(lgb$aggregated_shap_inhibit) < 1)] = 0
lgb$aggregated_shap_exhibit[which(abs(lgb$aggregated_shap_exhibit) < 1)] = 0
lgb = lgb[-which(lgb$aggregated_shap_inhibit == 0 & lgb$aggregated_shap_exhibit == 0),]
reg = reg[-which(reg$Inhibit == 0 & reg$Promote == 0),]

regmono = c()
lgbmono = c()
fcnmono = c()
for (i in 1:length(temp)) {
  site = temp[i]
  reg1 = reg[which(reg$site == site),]
  lgb1 = lgb[which(lgb$site == site),]
  fcn1 = fcn[which(fcn$site == site),]
  
  prolgb1 = length(lgb1$site[which(lgb1$aggregated_shap_inhibit/lgb1$aggregated_shap_exhibit < 0.1 | lgb1$aggregated_shap_inhibit/lgb1$aggregated_shap_exhibit > 10)]) / length(lgb1$site)
  profcn1 = length(fcn1$site[which(fcn1$aggregated_shap_inhibit/fcn1$aggregated_shap_exhibit < 0.1 | fcn1$aggregated_shap_inhibit/fcn1$aggregated_shap_exhibit > 10)]) / length(fcn1$site)
  proreg1 = length(reg1$site[which(abs(reg1$Inhibit/reg1$Promote) < 0.1|abs(reg1$Inhibit/reg1$Promote) > 10)]) / length(reg1$site)
  regmono = c(regmono,proreg1)
  lgbmono = c(lgbmono,prolgb1)
  fcnmono = c(fcnmono,profcn1)
}

mono = data.frame(site = temp, LGB = lgbmono, REG = regmono, FCN = fcnmono, stringsAsFactors = F)

po1 = data.frame(mono = mono$LGB, type = 'LGBM', stringsAsFactors = F)
po2 = data.frame(mono = mono$REG, type = 'REG', stringsAsFactors = F)
po3 = data.frame(mono = mono$FCN, type = 'FCN', stringsAsFactors = F)

po = rbind(po1,po2,po3)

library(ggplot2)
library(ggpubr)
ggplot(po, aes(x = reorder(type,mono,decreasing=T), y = mono, fill=type))+
  stat_boxplot(geom = "errorbar",width = 0.2, size = 0.5)+ 
  geom_boxplot(size=0.5,fill="white",outlier.shape = NA)+ 
  geom_boxplot(size=0.5,fill="white",outlier.fill="#B9D3EE",outlier.color="#B9D3EE", outlier.size = 0.5)+ 
  geom_jitter(aes(fill=type),width =0.2,shape = 21,size=1)+ 
  scale_fill_manual(values = c('FCN'='#B22222','LGBM'='#2E8B57','REG'='#4169E1'))+  
  ggtitle("Monotonic effects in all sites: TNBC-1")+
  theme_bw()+
  theme(legend.position="none", 
        axis.text.x=element_text(colour="black",size=15), 
        axis.text.y=element_text(size=15,face="plain"), 
        axis.title.y=element_text(size = 15,face="plain"), 
        axis.title.x=element_text(size = 15,face="plain"), 
        plot.title = element_text(size=20,face="bold",hjust = 0.5), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())+
  stat_compare_means(comparisons = list(c("FCN","REG"), c("REG","LGBM"),c("FCN","LGBM")))+
  ylab("monotonic RBP percent")+xlab("Model type")



lgb = read.table('tnbc1/lgbm_shap_summary.csv', sep = ',',  header = T, stringsAsFactors = F)
reg = read.table('tnbc1/lr_summary.csv', sep = ',',  header = T, stringsAsFactors = F)
fcn = read.table('tnbc1/fcn_summary.csv', sep = ',',  header = T, stringsAsFactors = F)

fcn$aggregated_shap_inhibit[which(abs(fcn$aggregated_shap_inhibit) < 1)] = 0
fcn$aggregated_shap_exhibit[which(abs(fcn$aggregated_shap_exhibit) < 1)] = 0
fcn = fcn[-which(fcn$aggregated_shap_inhibit == 0 & fcn$aggregated_shap_exhibit == 0),]
lgb$aggregated_shap_inhibit[which(abs(lgb$aggregated_shap_inhibit) < 1)] = 0
lgb$aggregated_shap_exhibit[which(abs(lgb$aggregated_shap_exhibit) < 1)] = 0
lgb = lgb[-which(lgb$aggregated_shap_inhibit == 0 & lgb$aggregated_shap_exhibit == 0),]
reg = reg[-which(reg$Inhibit == 0 & reg$Promote == 0),]

regmono = c()
lgbmono = c()
fcnmono = c()
for (i in 1:length(temp)) {
  site = temp[i]
  reg1 = reg[which(reg$site == site),]
  lgb1 = lgb[which(lgb$site == site),]
  fcn1 = fcn[which(fcn$site == site),]
  
  prolgb1 = length(lgb1$site[which(lgb1$aggregated_shap_inhibit/lgb1$aggregated_shap_exhibit < 0.1 | lgb1$aggregated_shap_inhibit/lgb1$aggregated_shap_exhibit > 10)]) / length(lgb1$site)
  profcn1 = length(fcn1$site[which(fcn1$aggregated_shap_inhibit/fcn1$aggregated_shap_exhibit < 0.1 | fcn1$aggregated_shap_inhibit/fcn1$aggregated_shap_exhibit > 10)]) / length(fcn1$site)
  proreg1 = length(reg1$site[which(abs(reg1$Inhibit/reg1$Promote) < 0.1|abs(reg1$Inhibit/reg1$Promote) > 10)]) / length(reg1$site)
  regmono = c(regmono,proreg1)
  lgbmono = c(lgbmono,prolgb1)
  fcnmono = c(fcnmono,profcn1)
}

mono = data.frame(site = temp, LGB = lgbmono, REG = regmono, FCN = fcnmono, stringsAsFactors = F)

po1 = data.frame(mono = mono$LGB, type = 'LGBM', stringsAsFactors = F)
po2 = data.frame(mono = mono$REG, type = 'REG', stringsAsFactors = F)
po3 = data.frame(mono = mono$FCN, type = 'FCN', stringsAsFactors = F)

po = rbind(po1,po2,po3)

library(ggplot2)
library(ggpubr)
ggplot(po, aes(x = reorder(type,mono,decreasing=T), y = mono, fill=type))+
  stat_boxplot(geom = "errorbar",width = 0.2, size = 0.5)+ 
  geom_boxplot(size=0.5,fill="white",outlier.shape = NA)+ 
  geom_boxplot(size=0.5,fill="white",outlier.fill="#B9D3EE",outlier.color="#B9D3EE", outlier.size = 0.5)+ 
  geom_jitter(aes(fill=type),width =0.2,shape = 21,size=1)+ 
  scale_fill_manual(values = c('FCN'='#B22222','LGBM'='#2E8B57','REG'='#4169E1'))+  
  ggtitle("Monotonic effects in all sites: TNBC-2")+
  theme_bw()+
  theme(legend.position="none", 
        axis.text.x=element_text(colour="black",size=15), 
        axis.text.y=element_text(size=15,face="plain"), 
        axis.title.y=element_text(size = 15,face="plain"), 
        axis.title.x=element_text(size = 15,face="plain"), 
        plot.title = element_text(size=20,face="bold",hjust = 0.5), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())+
  stat_compare_means(comparisons = list(c("FCN","REG"), c("REG","LGBM"),c("FCN","LGBM")))+
  ylab("monotonic RBP percent")+xlab("Model type")



lgb = read.table('tnbc2/lgbm_shap_summary.csv', sep = ',',  header = T, stringsAsFactors = F)
reg = read.table('tnbc2/lr_summary.csv', sep = ',',  header = T, stringsAsFactors = F)
fcn = read.table('tnbc2/fcn_summary.csv', sep = ',',  header = T, stringsAsFactors = F)

fcn = fcn[-which(fcn$aggregated_shap_inhibit == 0 & fcn$aggregated_shap_exhibit == 0),]
fcn$aggregated_shap_inhibit[which(abs(fcn$aggregated_shap_inhibit) < 1)] = 0
fcn$aggregated_shap_exhibit[which(abs(fcn$aggregated_shap_exhibit) < 1)] = 0
lgb = lgb[-which(lgb$aggregated_shap_inhibit == 0 & lgb$aggregated_shap_exhibit == 0),]
lgb$aggregated_shap_inhibit[which(abs(lgb$aggregated_shap_inhibit) < 1)] = 0
lgb$aggregated_shap_exhibit[which(abs(lgb$aggregated_shap_exhibit) < 1)] = 0
reg = reg[-which(reg$Inhibit == 0 & reg$Promote == 0),]

regmono = c()
lgbmono = c()
fcnmono = c()
for (i in 1:length(temp)) {
  site = temp[i]
  reg1 = reg[which(reg$site == site),]
  lgb1 = lgb[which(lgb$site == site),]
  fcn1 = fcn[which(fcn$site == site),]
  
  prolgb1 = length(lgb1$site[which(lgb1$aggregated_shap_inhibit/lgb1$aggregated_shap_exhibit < 0.1 | lgb1$aggregated_shap_inhibit/lgb1$aggregated_shap_exhibit > 10)]) / length(lgb1$site)
  profcn1 = length(fcn1$site[which(fcn1$aggregated_shap_inhibit/fcn1$aggregated_shap_exhibit < 0.1 | fcn1$aggregated_shap_inhibit/fcn1$aggregated_shap_exhibit > 10)]) / length(fcn1$site)
  proreg1 = length(reg1$site[which(abs(reg1$Inhibit/reg1$Promote) < 0.1|abs(reg1$Inhibit/reg1$Promote) > 10)]) / length(reg1$site)
  regmono = c(regmono,proreg1)
  lgbmono = c(lgbmono,prolgb1)
  fcnmono = c(fcnmono,profcn1)
}

mono = data.frame(site = temp, LGB = lgbmono, REG = regmono, FCN = fcnmono, stringsAsFactors = F)

po1 = data.frame(mono = mono$LGB, type = 'LGBM', stringsAsFactors = F)
po2 = data.frame(mono = mono$REG, type = 'REG', stringsAsFactors = F)
po3 = data.frame(mono = mono$FCN, type = 'FCN', stringsAsFactors = F)

po = rbind(po1,po2,po3)

library(ggplot2)
library(ggpubr)
ggplot(po, aes(x = reorder(type,mono,decreasing=T), y = mono, fill=type))+
  stat_boxplot(geom = "errorbar",width = 0.2, size = 0.5)+ 
  geom_boxplot(size=0.5,fill="white",outlier.shape = NA)+ 
  geom_boxplot(size=0.5,fill="white",outlier.fill="#B9D3EE",outlier.color="#B9D3EE", outlier.size = 0.5)+ 
  geom_jitter(aes(fill=type),width =0.2,shape = 21,size=1)+ 
  scale_fill_manual(values = c('FCN'='#B22222','LGBM'='#2E8B57','REG'='#4169E1'))+  
  ggtitle("Monotonic effects in all sites: Tcell (HCC)")+
  theme_bw()+
  theme(legend.position="none", 
        axis.text.x=element_text(colour="black",size=15), 
        axis.text.y=element_text(size=15,face="plain"), 
        axis.title.y=element_text(size = 15,face="plain"), 
        axis.title.x=element_text(size = 15,face="plain"), 
        plot.title = element_text(size=20,face="bold",hjust = 0.5), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())+
  stat_compare_means(comparisons = list(c("FCN","REG"), c("REG","LGBM"),c("FCN","LGBM")))+
  ylab("monotonic RBP percent")+xlab("Model type")
