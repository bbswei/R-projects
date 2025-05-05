library(rjags)
library(coda)
library(ggplot2)
library(dplyr)
library(ggmcmc)
library(caret)
library(viridis)
library(showtext)
library(patchwork)

dt2 <- read.csv("BAYESIAN2.csv")
showtext_auto()

#set.seed(123)  
#dt_sample <- dt[sample(nrow(dt), 5000, replace = FALSE), ]

features <- c("similarity", "word_count", "readability", "sentiment", "bullet_points")
dt2[features] <- scale(dt2[features])

dt2$readability <- -dt2$readability

dt_sample2 <- dt2 %>%
  mutate(rating_binary = ifelse(rating >= 4.5, 1, 0))  

data_list <- list(
  N = nrow(dt_sample2),
  y = dt_sample2$rating_binary, 
  similarity = dt_sample2$similarity,
  word_count = dt_sample2$word_count,
  readability = dt_sample2$readability,
  sentiment = dt_sample2$sentiment,
  bullet_points = dt_sample2$bullet_points
)

model5_1 <- "
model {
  sigma ~ dunif(0.1, 10)

  for (i in 1:N) {
    z_high[i] ~ dnorm(mu_high[i], sigma^(-2))
    z_low[i] ~ dnorm(mu_low[i], sigma^(-2))

    mu_high[i] <- beta0_high + 
                  beta1_high * similarity[i] + beta2_high * readability[i] +
                  beta3_high * word_count[i] + beta4_high * sentiment[i] +
                  beta5_high * bullet_points[i] +
                  beta6_high * word_count[i] * readability[i] +  
                  beta7_high * similarity[i] * readability[i] +  
                  beta8_high * word_count[i] * bullet_points[i] +  
                  beta9_high * word_count[i] * similarity[i] 

    mu_low[i] <- beta0_low + 
                 beta1_low * similarity[i] + beta2_low * readability[i] +
                 beta3_low * word_count[i] + beta4_low * sentiment[i] +
                 beta5_low * bullet_points[i] +
                 beta6_low * word_count[i] * readability[i] +  
                 beta7_low * similarity[i] * readability[i] +  
                 beta8_low * word_count[i] * bullet_points[i] +  
                 beta9_low * word_count[i] * similarity[i]  

    prob_high[i] <- phi(z_high[i] - z_low[i])
    y[i] ~ dbern(prob_high[i])  
  }

  beta0_high ~ dnorm(0, 0.5)  
  beta1_high ~ dnorm(0, 0.5)
  beta2_high ~ dnorm(0, 0.5)
  beta3_high ~ dnorm(0, 0.5)
  beta4_high ~ dnorm(0, 0.5)
  beta5_high ~ dnorm(0, 0.5)
  beta6_high ~ dnorm(0, 0.5)
  beta7_high ~ dnorm(0, 0.5)
  beta8_high ~ dnorm(0, 0.5)
  beta9_high ~ dnorm(0, 0.5)  

  beta0_low ~ dnorm(0, 0.5)  
  beta1_low ~ dnorm(0, 0.5)
  beta2_low ~ dnorm(0, 0.5)
  beta3_low ~ dnorm(0, 0.5)
  beta4_low ~ dnorm(0, 0.5)
  beta5_low ~ dnorm(0, 0.5)
  beta6_low ~ dnorm(0, 0.5)
  beta7_low ~ dnorm(0, 0.5)
  beta8_low ~ dnorm(0, 0.5)
  beta9_low ~ dnorm(0, 0.5)  
}
"

writeLines(model5_1, con = "model5_1.txt")

jmodel5_1 <- jags.model("model5_1.txt", data = data_list, n.chains = 2)
update(jmodel5_1, n.iter = 1000)

samples5_1 <- coda.samples(jmodel5_1, 
                           c("beta0_high", "beta1_high", "beta2_high", 
                             "beta3_high", "beta4_high", "beta5_high", 
                             "beta6_high", "beta7_high", "beta8_high",
                             "beta9_high", 
                             "beta0_low", "beta1_low", "beta2_low", 
                             "beta3_low", "beta4_low", "beta5_low",
                             "beta6_low", "beta7_low", "beta8_low",
                             "beta9_low"), 
                           n.iter = 1000)

summary(samples5_1)

post5_1 <- ggs(samples5_1)
ggs_density(post5_1) + facet_wrap(~ Parameter, scales = "free")

params5_1 <- c("beta0_high", "beta1_high", "beta2_high", 
               "beta3_high", "beta4_high", "beta5_high",
               "beta6_high", "beta7_high", "beta8_high",
               "beta9_high",
               "beta0_low", "beta1_low", "beta2_low", 
               "beta3_low", "beta4_low", "beta5_low",
               "beta6_low", "beta7_low", "beta8_low",
               "beta9_low")

extract_MAP <- function(param_name, post_data) {
  samples_filtered <- post_data %>% filter(Parameter == param_name) %>% pull(value)
  d <- density(samples_filtered)  
  return(d$x[which.max(d$y)])  
}

MAP5_1 <- sapply(params5_1, function(b) extract_MAP(b, post5_1))
MAP5_1

dt_sample2$z_high <- pnorm(
  MAP5_1["beta0_high"] + 
    MAP5_1["beta1_high"] * dt_sample2$similarity +
    MAP5_1["beta2_high"] * dt_sample2$readability +
    MAP5_1["beta3_high"] * dt_sample2$word_count +
    MAP5_1["beta4_high"] * dt_sample2$sentiment +
    MAP5_1["beta5_high"] * dt_sample2$bullet_points +
    MAP5_1["beta6_high"] * dt_sample2$word_count * dt_sample2$readability +
    MAP5_1["beta7_high"] * dt_sample2$similarity * dt_sample2$readability +
    MAP5_1["beta8_high"] * dt_sample2$word_count * dt_sample2$bullet_points +
    MAP5_1["beta9_high"] * dt_sample2$word_count * dt_sample2$similarity
)

dt_sample2$z_low <- pnorm(
  MAP5_1["beta0_low"] + 
    MAP5_1["beta1_low"] * dt_sample2$similarity +
    MAP5_1["beta2_low"] * dt_sample2$readability +
    MAP5_1["beta3_low"] * dt_sample2$word_count +
    MAP5_1["beta4_low"] * dt_sample2$sentiment +
    MAP5_1["beta5_low"] * dt_sample2$bullet_points +
    MAP5_1["beta6_low"] * dt_sample2$word_count * dt_sample2$readability +
    MAP5_1["beta7_low"] * dt_sample2$similarity * dt_sample2$readability +
    MAP5_1["beta8_low"] * dt_sample2$word_count * dt_sample2$bullet_points +
    MAP5_1["beta9_low"] * dt_sample2$word_count * dt_sample2$similarity
)

df_z_scores <- data.frame(
  z_value = c(dt_sample2$z_high, dt_sample2$z_low),
  group = rep(c("High Score Group", "Low Score Group"), each = nrow(dt_sample2))
)

p_z_density <- ggplot(df_z_scores, aes(x = z_value, fill = group)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("blue", "red")) +
  labs(title = "Density Plot of z-scores for High and Low Score Groups",
       x = "z-score",
       y = "Density",
       fill = "Score Group") +
  theme_minimal()

p_z_density


df_z_diff <- data.frame(
  z_value = c(dt_sample2$z_high, dt_sample2$z_low),
  true_label = rep(dt_sample2$rating_binary, 2),
  group = rep(c("High z-score (z_high)", "Low z-score (z_low)"), each = nrow(dt_sample2))
)

p_z_diff <- ggplot(df_z_diff, aes(x = z_value, fill = as.factor(true_label))) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ group, scales = "free") +
  scale_fill_manual(values = c("#F8766D", "#00BFC4"), labels = c("Low Score (0)", "High Score (1)")) +
  labs(
    title = "Density Plot of z-scores with True Labels",
    x = "z-score",
    y = "Density",
    fill = "True Label"
  ) +
  theme_minimal()

p_z_diff
################################################################################
plot_heatmap_diff <- function(var1, var2, label1, label2, map_params, dt_sample2, resolution = 50) {
  grid_data <- expand.grid(
    var1 = seq(min(dt_sample2[[var1]], na.rm = TRUE), max(dt_sample2[[var1]], na.rm = TRUE), length.out = resolution),
    var2 = seq(min(dt_sample2[[var2]], na.rm = TRUE), max(dt_sample2[[var2]], na.rm = TRUE), length.out = resolution)
  )
  
  similarity_mean <- mean(dt_sample2$similarity, na.rm = TRUE)
  readability_mean <- mean(dt_sample2$readability, na.rm = TRUE)
  word_count_mean <- mean(dt_sample2$word_count, na.rm = TRUE)
  sentiment_mean <- mean(dt_sample2$sentiment, na.rm = TRUE)
  bullet_points_mean <- mean(dt_sample2$bullet_points, na.rm = TRUE)
  
  estimate_mu <- function(v1, v2, high_or_low) {
    map_params[paste0("beta0_", high_or_low)] +
      map_params[paste0("beta1_", high_or_low)] * (ifelse(var1 == "similarity", v1, similarity_mean)) +
      map_params[paste0("beta2_", high_or_low)] * (ifelse(var2 == "readability", v2, readability_mean)) +
      map_params[paste0("beta3_", high_or_low)] * (ifelse(var1 == "word_count", v1, word_count_mean)) +
      map_params[paste0("beta4_", high_or_low)] * (ifelse(var2 == "sentiment", v2, sentiment_mean)) +
      map_params[paste0("beta5_", high_or_low)] * (ifelse(var1 == "bullet_points", v1, bullet_points_mean)) +
      map_params[paste0("beta6_", high_or_low)] * (ifelse((var1 == "word_count" && var2 == "readability") || 
                                                            (var1 == "readability" && var2 == "word_count"), 
                                                          v1 * v2, word_count_mean * readability_mean)) +
      map_params[paste0("beta7_", high_or_low)] * (ifelse((var1 == "similarity" && var2 == "readability") || 
                                                            (var1 == "readability" && var2 == "similarity"), 
                                                          v1 * v2, similarity_mean * readability_mean)) +
      map_params[paste0("beta8_", high_or_low)] * (ifelse((var1 == "word_count" && var2 == "bullet_points") || 
                                                            (var1 == "bullet_points" && var2 == "word_count"), 
                                                          v1 * v2, word_count_mean * bullet_points_mean)) +
      map_params[paste0("beta9_", high_or_low)] * (ifelse((var1 == "word_count" && var2 == "similarity") || 
                                                            (var1 == "similarity" && var2 == "word_count"), 
                                                          v1 * v2, word_count_mean * similarity_mean))
  }
  
  grid_data$z_high <- mapply(estimate_mu, grid_data$var1, grid_data$var2, MoreArgs = list(high_or_low = "high"))
  grid_data$z_low <- mapply(estimate_mu, grid_data$var1, grid_data$var2, MoreArgs = list(high_or_low = "low"))
  
  grid_data$z_diff <- grid_data$z_high - grid_data$z_low
  
  p <- ggplot(grid_data, aes(x = var1, y = var2, fill = z_diff)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
    labs(title = paste("Difference in mu:", label1, "vs.", label2), 
         x = label1, 
         y = label2, 
         fill = "mu High - mu Low") +
    theme_minimal()
  
  return(p)
}

map_params <- MAP5_1

p_diff1 <- plot_heatmap_diff("similarity", "readability", "Similarity", "Readability", map_params, dt_sample2)
p_diff2 <- plot_heatmap_diff("word_count", "readability", "Word Count", "Readability", map_params, dt_sample2)
p_diff3 <- plot_heatmap_diff("word_count", "similarity", "Word Count", "Similarity", map_params, dt_sample2)
p_diff4 <- plot_heatmap_diff("bullet_points", "word_count", "Bullet Points", "Word Count", map_params, dt_sample2)

library(patchwork)

combined_diff_plot <- (p_diff1 | p_diff2) / (p_diff3 | p_diff4) +
  plot_annotation(title = "Differences in mu Across Feature Combinations") &
  theme(plot.title = element_text(hjust = 0.5, size = 23))

combined_diff_plot
################################################################################

plot_heatmap_5_1 <- function(var1, var2, label1, label2, map_params, dt_sample2, type = "all", resolution = 50) {
  grid_data <- expand.grid(
    var1 = seq(min(dt_sample2[[var1]], na.rm = TRUE), max(dt_sample2[[var1]], na.rm = TRUE), length.out = resolution),
    var2 = seq(min(dt_sample2[[var2]], na.rm = TRUE), max(dt_sample2[[var2]], na.rm = TRUE), length.out = resolution)
  )
  
  similarity_mean <- mean(dt_sample2$similarity, na.rm = TRUE)
  readability_mean <- mean(dt_sample2$readability, na.rm = TRUE)
  word_count_mean <- mean(dt_sample2$word_count, na.rm = TRUE)
  sentiment_mean <- mean(dt_sample2$sentiment, na.rm = TRUE)
  bullet_points_mean <- mean(dt_sample2$bullet_points, na.rm = TRUE)
  
  estimate_mu <- function(v1, v2, high_or_low) {
    map_params[paste0("beta0_", high_or_low)] +
      map_params[paste0("beta1_", high_or_low)] * (ifelse(var1 == "similarity", v1, similarity_mean)) +
      map_params[paste0("beta2_", high_or_low)] * (ifelse(var2 == "readability", v2, readability_mean)) +
      map_params[paste0("beta3_", high_or_low)] * (ifelse(var1 == "word_count", v1, word_count_mean)) +
      map_params[paste0("beta4_", high_or_low)] * (ifelse(var2 == "sentiment", v2, sentiment_mean)) +
      map_params[paste0("beta5_", high_or_low)] * (ifelse(var1 == "bullet_points", v1, bullet_points_mean)) +
      map_params[paste0("beta6_", high_or_low)] * (ifelse((var1 == "word_count" && var2 == "readability") || 
                                                            (var1 == "readability" && var2 == "word_count"), 
                                                          v1 * v2, word_count_mean * readability_mean)) +
      map_params[paste0("beta7_", high_or_low)] * (ifelse((var1 == "similarity" && var2 == "readability") || 
                                                            (var1 == "readability" && var2 == "similarity"), 
                                                          v1 * v2, similarity_mean * readability_mean)) +
      map_params[paste0("beta8_", high_or_low)] * (ifelse((var1 == "word_count" && var2 == "bullet_points") || 
                                                            (var1 == "bullet_points" && var2 == "word_count"), 
                                                          v1 * v2, word_count_mean * bullet_points_mean)) +
      map_params[paste0("beta9_", high_or_low)] * (ifelse((var1 == "word_count" && var2 == "similarity") || 
                                                            (var1 == "similarity" && var2 == "word_count"), 
                                                          v1 * v2, word_count_mean * similarity_mean))
  }
  
  grid_data$z_high <- mapply(estimate_mu, grid_data$var1, grid_data$var2, MoreArgs = list(high_or_low = "high"))
  grid_data$z_low <- mapply(estimate_mu, grid_data$var1, grid_data$var2, MoreArgs = list(high_or_low = "low"))
  
  grid_data$z_diff <- grid_data$z_high - grid_data$z_low
  
  plot_single_heatmap <- function(data, z_column, title, fill_label) {
    ggplot(data, aes(x = var1, y = var2, fill = !!sym(z_column))) +
      geom_tile() +
      scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
      labs(title = title, x = label1, y = label2, fill = fill_label) +
      theme_minimal()
  }
  
  plots <- list()
  
  if (type == "high" || type == "all") {
    plots$high <- plot_single_heatmap(grid_data, "z_high", 
                                      paste("High Score Group"),
                                      "mu High")
  }
  
  if (type == "low" || type == "all") {
    plots$low <- plot_single_heatmap(grid_data, "z_low", 
                                     paste("Low Score Group"),
                                     "mu Low")
  }
  
  if (type == "diff" || type == "all") {
    plots$diff <- plot_single_heatmap(grid_data, "z_diff", 
                                      paste("Difference in mu"),
                                      "mu High - mu Low")
  }
  
  return(plots)
}

map_params <- MAP5_1 

plots1 <- plot_heatmap_5_1("similarity", "readability", "Similarity", "Readability", map_params, dt_sample2, type = "all")
plots2 <- plot_heatmap_5_1("word_count", "readability", "Word Count", "Readability", map_params, dt_sample2, type = "all")
plots3 <- plot_heatmap_5_1("word_count", "similarity", "Word Count", "Similarity", map_params, dt_sample2, type = "all")
plots4 <- plot_heatmap_5_1("sentiment", "readability", "Sentiment", "Readability", map_params, dt_sample2, type = "all")
plots5 <- plot_heatmap_5_1("bullet_points", "word_count", "Bullet Points", "Word Count", map_params, dt_sample2, type = "all")

library(patchwork)

combine_plots <- function(plot_list, title) {
  (plot_list$high | plot_list$low | plot_list$diff) + 
    plot_annotation(title = title) &
    theme(plot.title = element_text(hjust = 0.5, size = 23))
}

p1_combined <- combine_plots(plots1, "Similarity vs. Readability")
p2_combined <- combine_plots(plots2, "Word Count vs. Readability")
p3_combined <- combine_plots(plots3, "Word Count vs. Similarity")
p4_combined <- combine_plots(plots4, "Sentiment vs. Readability")
p5_combined <- combine_plots(plots5, "Bullet Points vs. Word Count")

p1_combined
p2_combined
p3_combined
p4_combined
p5_combined


dt_sample2$prob_high <- pnorm(dt_sample2$z_high - dt_sample2$z_low)

ggplot(dt_sample2, aes(x = prob_high, fill = as.factor(rating_binary))) +
  geom_density(alpha = 0.5) +
  labs(
    title = "Probability Distribution of High Score Group",
    x = "Probability of High Score Group (z_high - z_low)",
    y = "Density",
    fill = "True Label"
  ) +
  theme_minimal()

conf_matrix <- confusionMatrix(
  as.factor(ifelse(dt_sample2$prob_high > 0.5, 1, 0)),
  as.factor(dt_sample2$rating_binary)
)

conf_mat <- as.data.frame(as.table(conf_matrix$table))
colnames(conf_mat) <- c("Predicted", "Actual", "Count")

ggplot(conf_mat, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), color = "white", size = 5) +  
  scale_fill_gradientn(colors = c("lightblue", "blue", "darkblue"), name = "Count") +
  labs(
    title = "Confusion Matrix",
    x = "Predicted Label",
    y = "Actual Label"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16),  
    axis.title = element_text(size = 12),  
    axis.text = element_text(size = 10)  
  )
