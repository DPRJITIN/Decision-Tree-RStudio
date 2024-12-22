#Step 1: Import Necessary Libraries
library(rpart)
library(rpart.plot)
library(caret)
library(e1071)
library(ggplot2)

#Step 2: Load and Prepare the Dataset
#Load the Iris dataset
data(iris)
#Split the dataset into training and testing sets
set.seed(42)
index <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
train_data <- iris[index, ]
test_data <- iris[-index, ]

#Step 3: Create and Train the Decision Tree Model, Visualise the Model
tree_model <- rpart(Species ~ ., data = train_data, method = "class")
#Visualise Decision Tree Model
rpart.plot(tree_model, type = 3, extra = 102, fallen.leaves = TRUE, main = "Decision Tree Visualisation")

#Step 4: Make Predictions and Evaluate the Model
#Make predictions
predictions <- predict(tree_model, test_data, type = "class")
#Evaluate the model
confusion_matrix <- confusionMatrix(predictions, test_data$Species)
print(confusion_matrix)
#Visualise the confusion matrix
conf_matrix <- as.data.frame(confusion_matrix$table)
#Calculate accuracy
accuracy <- sum(diag(confusion_matrix$table)) / sum(confusion_matrix$table)
print(paste("Accuracy:", round(accuracy, 2)))
#Plot Confusion Matrix with Accuracy
ggplot(conf_matrix, aes(Prediction, Reference)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = paste("Confusion Matrix, Accuracy:", round(accuracy, 2)), x = "Predicted", y = "Actual")

#Step 5: Prune the Decision Tree and Plot
pruned_tree <- prune(tree_model, cp = tree_model$cptable[which.min(tree_model$cptable[,"xerror"]), "CP"])
# Visualise the pruned decision tree
rpart.plot(pruned_tree, type = 3, extra = 102, fallen.leaves = TRUE, main = "Pruned Decision Tree Visualisation")