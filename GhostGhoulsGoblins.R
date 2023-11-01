library(tidyverse)
library(vroom)
library(tidymodels)

missing_values <- vroom("./GhostGhoulsandGoblins/trainWithMissingValues.csv")
missing_values

train <- vroom("./GhostGhoulsandGoblins/train.csv")

missing_values_recipe <- recipe(type ~ ., data = missing_values) %>%
  step_impute_median(all_numeric_predictors())
prep <- prep(missing_values_recipe)
baked_train <- bake(prep, new_data = missing_values)

rmse_vec(train[is.na(missing_values)], baked_train[is.na(missing_values)])
