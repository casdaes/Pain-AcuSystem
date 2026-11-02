
suppressPackageStartupMessages({library(lme4); library(lmerTest); library(ordinal)})
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1], fileEncoding = "UTF-8", stringsAsFactors = FALSE)
arms <- if (length(args) >= 2) strsplit(args[2], ",")[[1]] else c("A1", "A2", "A3", "A4", "A5")
d <- droplevels(subset(d, arm %in% arms))
d$case_id <- factor(d$case_id); d$rater <- factor(d[["评分者"]]); d$arm <- factor(d$arm, levels = arms)
d$task <- factor(d$task); d$cat7 <- factor(d$cat7); d$len_z <- as.numeric(scale(d$n_chars))
d$overall <- rowMeans(d[, c("safety", "accuracy", "usability")])
ctl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

pairwise <- function(m) {
  lsm <- as.data.frame(ls_means(m, which = "arm", pairwise = TRUE))
  lsm$p_holm <- p.adjust(lsm[["Pr(>|t|)"]], method = "holm")
  print(round(lsm[, c("Estimate", "Std. Error", "df", "lower", "upper", "Pr(>|t|)", "p_holm")], 4))
}

for (s in c("overall", "safety", "accuracy", "usability")) {
  cat("\n=====", s, ": ", s, "~ arm + (1|case) + (1|rater) =====\n")
  m <- lmer(as.formula(paste(s, "~ arm + (1 | case_id) + (1 | rater)")), data = d)
  print(anova(m))
  pairwise(m)
  cat("--- adjusted for task, pain category and output length ---\n")
  ma <- lmer(as.formula(paste(s, "~ arm + task + cat7 + len_z + (1 | case_id) + (1 | rater)")), data = d)
  print(anova(ma)["arm", ])
  pairwise(ma)
}

cat("\n===== critical safety error: se_any ~ arm + (1|case) + (1|rater) (GLMM, logit) =====\n")
if (sum(d$se_any) > 0 && sum(d$se_any) < nrow(d)) {
  g <- tryCatch(glmer(se_any ~ arm + (1 | case_id) + (1 | rater), data = d, family = binomial, control = ctl), error = function(e) NULL)
  if (!is.null(g)) { print(summary(g)$coefficients); print(anova(update(g, . ~ . - arm), g)) }
} else cat("no variation in the rater-level events: GLMM not estimable (rely on Cochran's Q / McNemar)\n")


cat("\n===== ORDINAL mixed models (0 = no, 1 = partial, 2 = yes) =====\n")
for (v in c("ord_诊断一致", "ord_证型一致", "ord_选穴策略一致", "ord_操作合规", "ord_信息缺失识别")) {
  if (!(v %in% names(d))) next
  o <- subset(d, !is.na(d[[v]])); o$y <- factor(o[[v]], ordered = TRUE); o <- droplevels(o)
  if (nlevels(o$y) < 2 || nrow(o) < 30) next
  cat("\n---", v, "---\n")
  fit <- tryCatch(clmm(y ~ arm + (1 | case_id) + (1 | rater), data = o), error = function(e) NULL)
  if (!is.null(fit)) print(summary(fit)$coefficients[grep("^arm", rownames(summary(fit)$coefficients)), , drop = FALSE])
}
