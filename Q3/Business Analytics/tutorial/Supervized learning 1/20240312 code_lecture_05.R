### Model validation
set.seed(123)

# Simulate a data set
n <- 25 # number of observations
x <- runif(n, -2, 2) # predictor variable
y <- 2 * x + rnorm(n, sd = 2) # response variable with noise

# Create a highly complex model
x_matrix <- model.matrix(~ poly(x, degree = 10, raw = TRUE))
complex_model <- glmnet::glmnet(x_matrix, y, alpha = 0, lambda = 0)

# Create a simple linear model
simple_model <- lm(y ~ x)

# Predict using both models
x_test <- seq(-2, 2, length.out = 100)
x_test_matrix <- model.matrix(~ poly(x_test, degree = 10, raw = TRUE))
complex_predictions <- predict(complex_model, newx = x_test_matrix)
simple_predictions <- predict(simple_model, newdata = data.frame(x = x_test))

# Plot
library(ggplot2)

ggplot() +
    geom_point(aes(x, y), colour = "blue", size = 4) +
    geom_line(aes(x_test, complex_predictions), colour = "red", size = 1) +
    geom_line(aes(x_test, simple_predictions), colour = "green", size = 1) +
    xlab("Predictor (x)") +
    ylab("Response (y)") +
    theme_minimal()

### Classification: k-Nearest Neighbors
## Reading the data
dta_cla <- mlba::RidingMowers

## Displaying the data frame
dta_cla

## Plot the data
# Adding labels
dta_cla$Label <- rownames(dta_cla)

## Creating a new data point
dta_cla_new <- data.frame(Income = 60,
                          Lot_Size = 20,
                          Ownership = "New",
                          Label = "")

# Creating the plot
ggplot(rbind(dta_cla, dta_cla_new), 
       aes(x = Income, y = Lot_Size, color = Ownership)) +
    geom_point(size = 3) +
    scale_color_manual(values = c("Green", "Black", "Red")) +
    geom_text(aes(label = Label), vjust = -1) +
    theme_minimal() +
    labs(x = "Income",
         y = "Lot Size",
         color = "Ownership")

## Training model on dta_cla
# Create training and testing set
set.seed(35)
proportion <- 0.6
split <- rsample::initial_split(dta_cla, prop = proportion)
training <- rsample::training(split)
testing <- rsample::testing(split)

# Assign labels as row names training, remove Label column
rownames(training) <- training$Label
training <- training[, which(!names(training) %in% "Label")]
testing <- testing[, which(!names(testing) %in% "Label")]

# Train model on training data
knn_model <- caret::train(Ownership ~ .,
                          data = training,
                          method = "knn",
                          preProcess = c("center", "scale"),
                          tuneGrid = expand.grid(k = 3))

## Make prediction for new data point
# Creating new data point
dta_cla_new <- data.frame(Income = 60,
                          Lot_Size = 20)

# Predict class for new data point
predict(knn_model, dta_cla_new)

# Determine nearest neighbors to new data point
pred_train <- predict(knn_model$preProcess, training)
pred_new <- predict(knn_model$preProcess, dta_cla_new)
distances <- apply(pred_train[, 1:2], 
                   1,
                   function(d){ sqrt(sum((d - pred_new)^2)) })
rownames(training)[order(distances)][1:3]

# Make predictions for testing data
predict(knn_model, testing)

## Choosing the k with the best classification performance
# Use leave-one-out cross-validation for small data set
model <- caret::train(Ownership ~ ., data = training,
                      method = "knn",
                      preProcess = c("center", "scale"),
                      tuneGrid = expand.grid(k = c(1:13)),
                      trControl = trainControl(method = "loocv", 
                                               number = 5,
                                               allowParallel = TRUE))
model

### Classification: Naive Bayes
## Creating the fraud data frame
dta <- data.frame(charges = c("y", "n", "n", "n", "n", "n", "y", "y", "n", "y"), 
                  size = c(rep("small", 2), rep("large", 2), rep("small", 3), rep("large", 3)),
                  outcome = c(rep("truthful", 6), rep("fraud", 4)))

## Displaying the data frame
dta

## Reading the Flight Delays data
dta <- mlba::FlightDelays

## Checking the structure of the data frame
str(dta)

# Change numerical variables to categorical
dta$DAY_WEEK <- as.character(dta$DAY_WEEK)
dta$ORIGIN <- as.character(dta$ORIGIN)
dta$DEST <- as.character(dta$DEST)
dta$CARRIER <- as.character(dta$CARRIER)
dta$Flight.Status <- as.character(dta$Flight.Status)
dta$CRS_DEP_TIME <- as.character(round(dta$CRS_DEP_TIME / 100))

# Select variables to be included in the model
dta <- dta[, c("DAY_WEEK", "CRS_DEP_TIME", "ORIGIN", "DEST", "CARRIER", "Flight.Status")]
str(dta)

# Create training and testing set
set.seed(1)
proportion <- 0.6
split <- rsample::initial_split(dta, prop = proportion)
training <- rsample::training(split)
testing <- rsample::testing(split)

# Train Naive Bayes model
model <- e1071::naiveBayes(Flight.Status ~ ., data = training)
model$table$DAY_WEEK

# Predict class membership
predictions <- predict(model, testing)

# Create confusion matrix
gmodels::CrossTable(testing$Flight.Status, as.character(predictions),
                    prop.chisq = FALSE,
                    prop.c = FALSE,
                    prop.r = FALSE,
                    prop.t = FALSE,
                    dnn = c("Actual flight status", "Predicted flight status"))

### Classification: Discriminant analysis
## Reading the data
dta_cla <- mlba::RidingMowers

## Calculate Mahalonobis distance for each ownership group
# Separate data set
owner_data <- subset(dta_cla, Ownership == "Owner")
nonowner_data <- subset(dta_cla, Ownership == "Nonowner")

# Calculate Mahalonobis for owners
cov_owner <- cov(owner_data[,1:2])
cent_owner <- colMeans(owner_data[,1:2])
owner_data$Mahalanobis <- mahalanobis(owner_data[,1:2], center = cent_owner, cov = cov_owner)

# Calculate Mahalonobis for non-owners
cov_nonowner <- cov(nonowner_data[,1:2])
cent_nonowner <- colMeans(nonowner_data[,1:2])
nonowner_data$Mahalanobis <- mahalanobis(nonowner_data[,1:2], center = cent_nonowner, cov = cov_nonowner)

## Putting it back together and adding new data point
dta_cla <- rbind(owner_data, nonowner_data)
dta_cla <- rbind(dta_cla, data.frame(Income = 60,
                                     Lot_Size = 20,
                                     Ownership = "New",
                                     Mahalanobis = 0.5))

## Create the centroid data (for plotting purposes)
dta_cnt <- rbind(data.frame(Income = cent_owner['Income'], 
                            Lot_Size = cent_owner['Lot_Size'], 
                            Ownership = "Owner",
                            Mahalanobis = mean(dta_cla[ which(dta_cla$Ownership %in% "Owner"), ]$Mahalanobis)),
                 data.frame(Income = cent_nonowner['Income'], 
                            Lot_Size = cent_nonowner['Lot_Size'], 
                            Ownership = "Nonowner",
                            Mahalanobis = mean(dta_cla[ which(dta_cla$Ownership %in% "Nonowner"), ]$Mahalanobis)))

## Plot the data
ggplot() +
    geom_point(data = dta_cla, aes(x = Income, y = Lot_Size, color = Ownership, size = Mahalanobis)) +
    geom_point(data = dta_cnt, aes(x = Income, y = Lot_Size, color = Ownership), shape = 13, size = 4) +
    scale_color_manual(values = c("Owner" = "Red", "Nonowner" = "Black", "New" = "Green")) +
    scale_size_continuous(range = c(5, 1)) +
    theme_minimal() +
    labs(x = "Income",
         y = "Lot Size",
         color = "Ownership",
         size = "Mahalanobis Distance") +
    guides(size = "none", color = guide_legend(override.aes = list(size = 4, shape = 16)))

## Determining classification functions
# Performing Linear Discriminant Analysis
dta_cla <- mlba::RidingMowers
model <- MASS::lda(Ownership ~ Income + Lot_Size, data = dta_cla)

# Showing model output 
# Note: classification functions are NOT provided directly
model

# Use model for prediction
predict(model, data.frame(Income = 60,
                          Lot_Size = 20))

# Adjust prior probabilities
table(predict(model, dta_cla, prior = c(0.50, 0.50))$class)
table(predict(model, dta_cla, prior = c(0.85, 0.15))$class)

### Classification: Deep learning
library(h2o) # requires a Java installation to work!
h2o.init()

dta_cla <- mlba::RidingMowers
dta_cla$Ownership <- as.factor(ifelse(dta_cla$Ownership == "Owner", 1, 0))

set.seed(1)
proportion <- 0.7
split <- rsample::initial_split(dta_cla, prop = proportion)
training <- rsample::training(split)
testing <- rsample::testing(split)
training <- h2o::as.h2o(training, destination_frame = "dta_cla")
testing <- h2o::as.h2o(testing, destination_frame = "dta_cla")
features <- colnames(training[, -which(colnames(training) %in% "Ownership") ])

model <- h2o::h2o.deeplearning(x = features,
                               y = "Ownership",
                               training_frame = training,
                               activation = "Rectifier",
                               hidden = c(rep(3, 1)), # 3 neurons, 1 layer
                               epochs = 10) # 10 passes through the network

print("Confusion matrix based on testing data")
pred.test <- h2o::h2o.predict(model, testing)
gmodels::CrossTable(as.numeric(as.vector(testing$Ownership)), 
                    as.numeric(as.vector(pred.test$predict)),
                    prop.chisq = FALSE,
                    prop.c = FALSE,
                    prop.r = FALSE,
                    prop.t = FALSE,
                    dnn = c("Actual class", "Predicted class"))
