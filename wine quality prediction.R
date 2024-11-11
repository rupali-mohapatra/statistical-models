####---------STATISTICAL LEARNING------------#### 
getwd()
setwd("Desktop/Statistical learning/Home assignment 2")

#Load necessary libraries
#install.packages("caret")
#install.packages("rpart.plot")
#install.packages("gt")
#install.packages("reshape2")

library(readxl)
library(tree)
library(rpart)
library(ggplot2)
library(randomForest)
library(caret)
library(e1071)
library(rpart.plot)
library(gt)
library(dplyr)
library(tidyr)

#Read data files
red_wine <- read_excel("winequality-red.xlsx")
white_wine <- read_excel("winequality-white.xlsx")

#Add label for red and white wine
red_wine$wine_type <- "red"
white_wine$wine_type <- "white"

#combine both the data sets
all_wines <- rbind(red_wine, white_wine)
summary(all_wines)

# Compute the summary statistics
summary_stats <- all_wines[, -ncol(all_wines)] %>%
  summarise(across(everything(), list(
    mean = ~ mean(.x, na.rm = TRUE),
    sd = ~ sd(.x, na.rm = TRUE),
    min = ~ min(.x, na.rm = TRUE),
    max = ~ max(.x, na.rm = TRUE),
    median = ~ median(.x, na.rm = TRUE)
  ), .names = "{col}_{fn}"))

# Convert the summary statistics to a long format
summary_stats_long <- summary_stats %>%
  pivot_longer(everything(), names_to = c("variable", "stat"), names_sep = "_") %>%
  pivot_wider(names_from = "stat", values_from = "value")

# Create and print a nice table using gt
summary_table <- summary_stats_long %>%
  gt() %>%
  tab_header(
    title = "Summary Statistics for All Wines"
  ) %>%
  cols_label(
    variable = "Variable",
    min = "Min",
    max = "Max",
    median = "Median",
    mean = "Mean",
    sd = "SD"
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels(everything()
  )) %>%
  fmt_number(
    columns = c(mean, sd, min, max, median),
    decimals = 2
  )

print(summary_table) #the mean and variance (SD) varies a lot and hence I will scale all_wines features when I do SVM, PCA and clustering. Scaling is not required in decision trees and random forests
#As quality is a rating here and discrete data, I chose barplot to show the frequency of each rating
barplot(table(all_wines$quality), 
        xlab = "Quality",
        ylab = "Frequency",
        main = "Wine Quality Distribution",
        col = "#c1121f")
#correlation matrix
# Exclude non-numeric variables from the dataset
numeric_data <- all_wines[, sapply(all_wines, is.numeric)]
correlation_matrix <- cor(numeric_data)
par(mfrow=c(1, 1))
ggplot(data = melt(correlation_matrix), aes(Var1, Var2, fill = value)) +
  geom_tile(color="black") +
  scale_fill_gradient2(low = "blue", 
                       high = "red", 
                       mid = "white", 
                       midpoint = 0, 
                       limit = c(-1, 1), 
                       space = "Lab",
                       name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 1, size = 10),
        axis.text.y = element_text(size = 10),
        plot.title = element_text(size = 16),
        plot.background = element_rect(color = "black", size = 1)) +
  coord_fixed() +
  labs(title = "Correlation Heatmap of Numerical Variables")+
  geom_text(aes(label = round(value, 2)), color = "black", size = 3)

# Get the column names
col_names <- colnames(all_wines)
# Replace spaces with underscores
new_col_names <- gsub(" ", "_", col_names)
# Assign the new column names to the dataframe
colnames(all_wines) <- new_col_names
# Convert wine_type to factor
all_wines$wine_type <- as.factor(all_wines$wine_type)
sapply(all_wines, class)

#*---------------------ENSEMBLE MODELS-------------------------*#
set.seed(1)
train_indices <- sample(1:nrow(all_wines), 0.8*nrow(all_wines))
train_data <- all_wines[train_indices, ]
test_data <- all_wines[-train_indices, ]
#### *** MODEL 1- DECISION TREES *** ####
#Fit a decision tree to the train data
dtree <-rpart(quality ~ ., data = train_data, control = rpart.control(cp = 0))
summary(dtree)
plot(dtree, main="Decision Tree For Wine Quality (Unpruned)")

# Predict on test data
dt_test_predictions <- predict(dtree, test_data)
# Calculate Test MSE
dt_test_mse <- mean((test_data$quality - dt_test_predictions)^2)
# Calculate Test RMSE
dt_test_rmse <- sqrt(dt_test_mse)

cat("Test MSE:", dt_test_mse, "\n") #0.583403
cat("Test RMSE:", dt_test_rmse, "\n")#0.7638082

#------- 10-Fold cross-validation in decision tree ---------#
set.seed(2)

# 10-Fold CROSS VALIDATION for decision tree model
cp_values <- seq(0.01, 0.1, by=0.01)
# Initialize vectors to store RMSE and MSE values
train_rmse_values <- numeric(length(cp_values))
test_rmse_values <- numeric(length(cp_values))
train_mse_values <- numeric(length(cp_values))
test_mse_values <- numeric(length(cp_values))
# Set the control parameters for cross-validation
ctrl <- trainControl(method = "cv", number = 10)

# Perform 10-fold cross-validation for each cp value on the training data
for (i in seq_along(cp_values)) {
  
  # Train the decision tree model with the current cp value
  dtree_cv <- train(quality ~ ., data = train_data, method = "rpart",
                    trControl = ctrl, tuneGrid = data.frame(cp = cp_values[i]))
  
  # Get the cross-validated RMSE
  train_rmse_values[i] <- dtree_cv$results$RMSE
  train_mse_values[i] <- train_rmse_values[i]^2
}
# Find the cp value that minimizes MSE
optimal_cp <- cp_values[which.min(train_mse_values)]
min_train_mse <- min(train_mse_values)
min_train_rmse <- min(train_rmse_values)

# Print the optimal cp value and corresponding minimum training MSE
cat("Optimal CP value:", optimal_cp, "\n") #0.01
cat("Minimum Cross-Validated Training MSE:", min_train_mse, "\n")#0.5812148
cat("Minimum Cross-Validated Training RMSE:", min_train_rmse, "\n")#0.7623744

# Train the final decision tree model with the optimal cp value on the entire training set
final_tree <- rpart(quality ~ ., data = train_data, control = rpart.control(cp = optimal_cp))
summary(final_tree)

rpart.plot(final_tree,
           type=3,
           extra=101,
           fallen.leaves = TRUE,
           shadow.col = "gray",
           box.palette = "RdBu",
           cex = 0.8,
           main="Decision Tree For Wine Quality (Pruned)")
# Make predictions on the training set using the final pruned tree
train_predictions <- predict(final_tree, newdata = train_data)

# Calculate Training MSE and RMSE
final_train_mse <- mean((train_data$quality - train_predictions)^2)
final_train_rmse <- sqrt(final_train_mse)

# Make predictions on the test set using the final pruned tree
test_predictions <- predict(final_tree, newdata = test_data)

# Calculate Test MSE and RMSE
final_test_mse <- mean((test_data$quality - test_predictions)^2)
final_test_rmse <- sqrt(final_test_mse)

# Print the Training and Test MSE and RMSE
cat("Final Model Training MSE:", final_train_mse, "\n")#0.56974
cat("Final Model Training RMSE:", final_train_rmse, "\n")#0.75480
cat("Final Model Test MSE:", final_test_mse, "\n") #0.58534
cat("Final Model Test RMSE:", final_test_rmse, "\n") #0.76507

 #Important attributes to consider wine quality
 final_tree$variable.importance
 
 #Variable importance
 par(mar = c(5, 10, 4, 2))
 barplot(final_tree$variable.importance, 
         main = "Variable Importance (pruned tree)", 
         xlab = "Importance",
         horiz = TRUE,
         las=1,
         cex.names = 0.9,
         col = "cornflowerblue", 
         xlim = c(0, 800),
         )
 

 #boxplot
 ggplot(all_wines, aes(x = cut(alcohol, breaks = 10), y = quality, fill = cut(alcohol, breaks = 10))) +
   geom_boxplot() +
   xlab("Alcohol Content") +
   ylab("Wine Quality") +
   labs(title = "Boxplot of Wine Quality by Alcohol Content")+
   theme_minimal()+
   theme(plot.background = element_rect(color = "black", fill = NA),
         plot.title = element_text(hjust = 0.5),
         panel.grid = element_blank(),
         axis.text.x = element_text(angle = 90, hjust = 1)
         )
 
 #### *** MODEL 2-RANDOM FOREST *** ####
set.seed(3)
rf_model <- randomForest(quality ~., data = train_data, importance=TRUE)
rf_model
# Predictions on test data
predictions_rf <- predict(rf_model, newdata = test_data)
# Calculate MSE
mse_rf <- mean((test_data$quality - predictions_rf)^2)
# Calculate RMSE
rmse_rf <- sqrt(mse_rf)
# Print MSE and RMSE
cat("Random Forest Ensemble Model:\n")
cat("MSE:", mse_rf, "\n") #0.3500495
cat("RMSE:", rmse_rf, "\n") #0.5916498

importance(rf_model)
varImpPlot(rf_model, main = "Variable Importance plot (Random Forest Model)")

#CROSS VALIDATION for Random Forest (10 fold)
set.seed(4)
# Define the parameter grid for tuning
# Use p / 3 for regression problems
#tunegrid <- expand.grid(.mtry = floor((ncol(train_data) - 1) / 3))

# Train the Random Forest model with cross-validation
rf_cv_model <- train(quality ~ ., data = train_data, method = "rf",
                     trControl = ctrl,
                     importance = TRUE)
rf_cv_model
rf_cv_model$bestTune #mtry2
# Print the cross-validated RMSE and MSE
cv_rmse_rf <- rf_cv_model$results$RMSE
cv_mse_rf <- cv_rmse_rf^2
cat("Cross-Validated Random Forest Model:\n")
cat("Cross-Validated MSE:", cv_mse_rf, "\n")#0.3737376 
cat("Cross-Validated RMSE:", cv_rmse_rf, "\n")#0.6113408

# Predictions on test data
test_predictions_rf <- predict(rf_model, newdata = test_data)
# Calculate test MSE
test_mse_rf <- mean((test_data$quality - test_predictions_rf)^2)
# Calculate test RMSE
test_rmse_rf <- sqrt(test_mse_rf)

# Print training and test MSE and RMSE
cat("Random Forest Ensemble Model:\n")
cat("Test MSE:", test_mse_rf, "\n")#0.3509024
cat("Test RMSE:", test_rmse_rf, "\n")#0.5923701

#Parameter optimization using tune function
tune_rf_model <- tune(randomForest,
                 quality ~., 
                 data = all_wines,
                 ranges=list(mtry=c(2, 3, 4, 5, 6, 7), ntree=c(100, 200, 300, 500)))
# Summary of tuning results
summary(tune_rf_model)

# Extract the best-tuned model
tune_rf_model$best.model
tune_rf_model$best.parameters

# ****MODEL 3- SVM ****
set.seed(895)
# Fit an SVM model(Radial)
svr_model_radial <- svm(quality ~ ., 
                        data = train_data,
                        type="eps-regression",
                        kernel="radial", 
                        gamma=1, 
                        cost=1)
summary(svr_model_radial)

# Make predictions on the test data
predictions_svm_radial <- predict(svr_model_radial, newdata = test_data)
# Calculate MSE
mse_svm_radial <- mean((test_data$quality - predictions_svm_radial)^2)
# Calculate RMSE
rmse_svm_radial <- sqrt(mse_svm_radial)
# Print MSE and RMSE
cat("SVM Model with RBF kernel gamma=1 and cost=1:\n")
cat("MSE:", mse_svm_radial, "\n") #0.413417
cat("RMSE:", rmse_svm_radial, "\n") #0.6429865

#Parameter optimization SVM Radial model
tune_SVR_model_radial <- tune(svm, quality ~., 
                        data=all_wines, 
                        kernel = "radial",
                        ranges = list(cost=c(0.1, 1, 10, 100), 
                                      gamma= c(0.001, 0.01, 0.1, 1,5,10,100)))
summary(tune_SVM_radial)
tune_SVM_radial$best.parameters
tune
#Predict test data based on the best model
tuned_predictions <- predict(best)

# Fit an SVM model(polynomial)
svr_model_poly <- svm(quality ~ ., 
                      data = train_data,
                      type="eps-regression",
                      kernel="polynomial", 
                      degree=2, 
                      cost=1)
summary(svm_model_poly)
# Make predictions on the test data
predictions_poly <- predict(svm_model_poly, newdata = test_data)

# Calculate MSE
mse_svm_poly <- mean((test_data$quality - predictions_poly)^2)

# Calculate RMSE
rmse_svm_poly <- sqrt(mse_svm_poly)

# Print MSE and RMSE
cat("SVM Model with polynomial degree 2 and cost=1:\n")
cat("MSE:", mse_svm_poly, "\n")#0.5239485
cat("RMSE:", rmse_svm_poly, "\n")#0.7238428

#parameter tuning for SVM polynomial model
tune_SVM_polynomial <- tune(svm, quality ~ . ,
                            data=all_wines ,
                            kernel = "polynomial",
                            ranges = list(cost=c(0.001, 0.01, 0.1, 1, 5, 1),
                                          degree=c(2, 3, 4)))
summary(tune_SVM_polynomial)
tune_SVM_polynomial$best.model
tune_SVM_polynomial$best.parameters

# ---------***** MODEL 4- K-means clustering ****---------
#Using k-means and hierarchical clustering perform clustering on the full dataset. How many clusters are optimal based on the results? If you use k=2, can you validate the clusters based on the labels you have assigned? Does k-means and hierarchical clustering provide the same results? 
library(cluster)
#install.packages("factoextra")
#install.packages("dendextend")
#library(factoextra)
library(dendextend)

#STEP 1:PCA 
scaled_data <- scale(all_wines[, -ncol(all_wines)])
summary(scaled_data)
scaled_summary_table <- data.frame(
  Variable = colnames(all_wines[, -ncol(all_wines)]),
  Min = apply(scaled_data, 2, min),
  Max = apply(scaled_data, 2, max),
  Mean = apply(scaled_data, 2, mean),
  SD = apply(scaled_data, 2, sd)
)
print(scaled_summary_table)
summary_gt <- gt(scaled_summary_table) %>%
  tab_header(
    title = "Summary Statistics of Scaled Wine Data"
  ) %>%
  fmt_number(
    columns = c(Min, Max, Mean, SD),
    decimals = 2
  ) %>%
  cols_label(
    Variable = "Variable",
    Min = "Min",
    Max = "Max",
    Mean = "Mean",
    SD = "Standard Deviation"
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels(everything())
  ) %>%
  tab_options(
    table.font.size = "small",
    heading.align = "center"
  )

# Print the gt table
print(summary_gt)
PCA.out <- prcomp(scaled_data)
names(PCA.out)
#The center and scale components correspond to the means and standard deviations of the variables that were used for scaling prior to implementing PCA.
PCA.out$center
PCA.out$scale
#The rotation matrix provides the principal component loadings
PCA.out$rotation
dim(PCA.out$x)
# We can plot the first two principal components as follows:
biplot(PCA.out, scale=0)
PCA.out$sdev #Standard deviation of each principal component
#the variance explained by each principal component is obtained by squaring the sdev
variance_explained <- PCA.out$sdev^2
variance_explained
#To compute the proportion of total variance explained by each principal component:
PVE <- variance_explained/sum(variance_explained)
PVE
#plot PVE explained by each component as well as cumulative PVE
par(mfrow=c(1, 2))
plot(PVE, 
     xlab = "Principal Component",
     ylab="Proportion of Variance explained",
     ylim = c(0, 1),
     type="b"
)
plot(cumsum(PVE), 
     xlab="Principal Component",
     ylab="Cumulative Proportion of Variance explained",
     ylim=c(0, 1), 
     type = "b")
# ****______K-MEANS CLUSTERING______****
# Select the first 2 principal components
pca_data <- PCA.out$x[, 1:2]
pca_data_5 <- PCA.out$x[, 1:5]
# Perform K-means clustering with k=5 and k=2
set.seed(123)  # For reproducibility
kmeans_result_5 <- kmeans(pca_data_5, centers = 5, nstart = 25)
kmeans_result_5$centers
kmeans_result_5$tot.withinss
pca_data_df_5 <- as.data.frame(pca_data_5)
pca_data_df_5$cluster <- as.factor(kmeans_result_5$cluster)
table(pca_data_df_5$cluster)
ggplot(pca_data_df_5, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point() +
  labs(title = "K-Means Clustering Results with k = 5") +
  theme_minimal()+
  theme(
        panel.border = element_rect(color = "black", fill = NA),
        plot.title = element_text(hjust = 0.5),
        panel.grid = element_blank()
  )

kmeans_result_2 <- kmeans(pca_data, centers = 2, nstart = 25)
kmeans_result_2$centers
kmeans_result_2$tot.withinss

pca_data_df <- as.data.frame(pca_data)
pca_data_df$cluster <- as.factor(kmeans_result_2$cluster)
table(pca_data_df$cluster)
ggplot(pca_data_df, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point() +
  labs(title = "K-Means Clustering Results with k = 2") +
  theme_minimal()+
  theme(
        panel.border = element_rect(color = "black", fill = NA),
        plot.title = element_text(hjust = 0.5),
        panel.grid = element_blank()
  )

# ****---- MODEL 5 - Hierarchial clustering----****
# using euclidean distance
HC.complete <- hclust(dist(pca_data), method = "ward.D")
dend <- as.dendrogram(HC.complete) %>%
  set("labels_cex", 0.5) %>%
  color_branches(k = 2) %>%
  color_labels(k = 2)
# Plot the dendrogram
par(mfrow=c(1, 1))
plot(dend, main= "Complete linkage using euclidean distance", xlab="", sub="")
abline(h = 4000, col = "red", lty = 2)

#Cut the Dendrogram to Obtain Clusters:
cutree_result <- cutree(HC.complete, k = 2)
table(kmeans_result_2$cluster, cutree_result)
#--------------------------THE END-----------------------------------#