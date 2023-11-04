library(tidyverse)
library(vroom)
library(tidymodels)

train <- vroom("./GhostGhoulsandGoblins/train.csv")
# train <- train %>%
#   select(-c("id"))
test <- vroom("./GhostGhoulsandGoblins/test.csv")

ggg_recipe <- recipe(type ~ ., data=train) %>%
  step_lencode_glm(color, outcome = vars(type)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())
  # step_mutate_at(all_numeric_predictors(), fn=factor)
  
prep <- prep(ggg_recipe)
baked_train <- bake(prep, new_data = train)

# rmse_vec(train[is.na(missing_values)], baked_train[is.na(missing_values)])

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(svmRadial)

# Fit or Tune Model HERE
tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(train, v = 10, repeats=1)

CV_results <- svm_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))

# Predict
bestTune <- CV_results %>%
  select_best("accuracy")

# Finalize workflow and predict
final_wf <- svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

ggg_predictions <- predict(final_wf, new_data = test, type="class")

ggg_predictions$type <- ggg_predictions$.pred_class
ggg_predictions$id <- test$id
ggg_final <- ggg_predictions %>%
  select(id, type)

write.csv(ggg_final, "./GhostGhoulsandGoblins/svmradial.csv", row.names = F)

