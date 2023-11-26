# IMPORTING LIBRARIES

library(dplyr)
library(ggplot2)
library(DataExplorer)
library(summarytools)
library(reshape2)  
library(corrplot)
library(knitr)
library(caret)
library(MASS)
library(randomForest)
library(xgboost)


# DATA IMPORT
train_df <- read.csv("D:/ALTUNI/PROJECTS/R/Media Campaign/train.csv")
test_df <- read.csv("D:/ALTUNI/PROJECTS/R/Media Campaign/test.csv")

# To View the data type of the columns
str(train_df)
str(test_df)

# DATA SUMMARY
summary(train_df)
summary(test_df)

# DATA PREPROCESSING

# Check for missing values of train_data
print(colSums(is.na(train_df)))

# Observation-1: There are no missing values in train_data

# Checking for outliers
num_cols <- train_df[,-1]
boxplot(num_cols)

# Calculate IQR for numerical variables
Q1 <- apply(num_cols, 2, quantile, probs = 0.25)
Q3 <- apply(num_cols, 2, quantile, probs = 0.75)
IQR <- Q3 - Q1

# Identify outliers based on IQR method
outliers_iqr <- apply(num_cols, 2, function(x) (x < (Q1 - 1.5 * IQR) | x > (Q3 + 1.5 * IQR)))
outliers_iqr_data <- num_cols[apply(outliers_iqr, 1, any), ]

# Observation-2: There are no outliers.

# Check for duplicated rows
duplicated_rows <- train_df[duplicated(train_df), ]
duplicated_rows

# Observation-3: There are no duplicated rows.

# Extract numerical columns (excluding 'id')
numerical_columns <- train_df[, sapply(train_df, is.numeric)]
correlation_matrix <- cor(numerical_columns)

# Combine the numerical columns into a long format
data_long <- reshape2::melt(numerical_columns)

# Create histograms using ggplot2
ggplot(data_long, aes(x = value)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 30) +
  facet_wrap(~variable, scales = "free") +
  theme_minimal()

# Plot the correlation matrix
corrplot(correlation_matrix, method = "circle", type = "upper", tl.cex = 0.8, tl.col = "black")

# LINEAR REGRESSION MODEL

# Extract features and target variable for training
X_train <- train_df[, -c(1, which(names(train_df) == "cost"))]
y_train <- train_df$cost

# Train-Validation Split
set.seed(42)  # for reproducibility
train_index <- createDataPartition(y_train, p = 0.8, list = FALSE)
train_set <- train_df[train_index, ]
validation_set <- train_df[-train_index, ]

# Model Training
model <- lm(cost ~ ., data = train_set)

# Step 4: Model Evaluation

# Predict on the validation set
predictions <- predict(model, newdata = validation_set)
rmsle <- sqrt(mean((log1p(predictions) - log1p(validation_set$cost))^2))

cat("RMSLE on Validation Set:", rmsle, "\n")


# Model Training (XGBoost)
xgb_model <- xgboost(data = as.matrix(X_train), label = y_train, objective = "reg:squarederror",nrounds=100)

# Step 4: Model Evaluation (XGBoost)
# Predict on the validation set
xgb_predictions <- predict(xgb_model, as.matrix(validation_set[, -c(1, 17)]))
xgb_rmsle <- sqrt(mean((log1p(xgb_predictions) - log1p(validation_set$cost))^2))

cat("RMSLE on Validation Set (XGBoost):", xgb_rmsle, "\n")

# Built two models (1) Linear Regression Model and (2) XG Boost Model
# to predict the cost and finding RMSLE

# Observation-4: From the two models, RMSLE is lower in XG Boost Model than Linear Regression Model.

# So, calculated the predictions on test data using XG Boost Model.
# Predictions on test data using XGBoost model
X_test <- as.matrix(test_df[2:16])
test_df$cost <-predict(xgb_model, newdata=X_test)

# Create Submission File (XGBoost)
xgb_submission <- data.frame(id = test_df$id, cost = test_df$cost)
write.csv(xgb_submission, "D:/ALTUNI/PROJECTS/R/Media Campaign/sample_submission.csv", row.names = FALSE)
head(xgb_submission)
