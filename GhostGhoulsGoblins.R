library(tidyverse)
library(vroom)
library(tidymodels)
library(embed)
library(discrim)
library(naivebayes)
library(kknn)
library(kernlab)
library(themis)
library(nnet)
library(bonsai)
library(lightgbm)
library(dbarts)

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

## neural networks
nn_recipe <- recipe(type ~ ., data=train) %>%
  update_role(id, new_role="id") %>%
  step_lencode_glm(color, outcome = vars(type)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

prep <- prep(nn_recipe)
baked_train <- bake(prep, new_data = train)

nn_model <- mlp(hidden_units = tune(),
                epochs = 50,
                activation = "relu") %>%
  set_engine("keras", verbose=0) %>% #verbose = 0 prints off less
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 10)),
                            levels=5)

folds <- vfold_cv(train, v = 5, repeats=1)

tuned_nn <- nn_wf %>%
  tune_grid(resamples=folds,
            grid=nn_tuneGrid,
            metrics=metric_set(accuracy))

tuned_nn %>% collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

# Predict
bestTune <- CV_results %>%
  select_best("accuracy")

# Finalize workflow and predict
final_wf <- nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

ggg_predictions <- predict(final_wf, new_data = test, type="class")

ggg_predictions$type <- ggg_predictions$.pred_class
ggg_predictions$id <- test$id
ggg_final <- ggg_predictions %>%
  select(id, type)

write.csv(ggg_final, "./GhostGhoulsandGoblins/nn.csv", row.names = F)

## boosting
ggg_recipe <- recipe(type ~ ., data=train) %>%
  step_lencode_glm(color, outcome = vars(type)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())
# step_mutate_at(all_numeric_predictors(), fn=factor)

prep <- prep(ggg_recipe)
baked_train <- bake(prep, new_data = train)

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(boost_model)

# Fit or Tune Model HERE
tuning_grid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(train, v = 5, repeats=1)

CV_results <- boost_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))

# Predict
bestTune <- CV_results %>%
  select_best("accuracy")

# Finalize workflow and predict
final_wf <- boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

ggg_predictions <- predict(final_wf, new_data = test, type="class")

ggg_predictions$type <- ggg_predictions$.pred_class
ggg_predictions$id <- test$id
ggg_final <- ggg_predictions %>%
  select(id, type)

write.csv(ggg_final, "./GhostGhoulsandGoblins/boosting.csv", row.names = F)

## bart
bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(bart_model)

# Fit or Tune Model HERE
tuning_grid <- grid_regular(trees(),
                            levels = 3) ## L^2 total tuning possibilities

folds <- vfold_cv(train, v = 5, repeats=1)

CV_results <- bart_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))

# Predict
bestTune <- CV_results %>%
  select_best("accuracy")

# Finalize workflow and predict
final_wf <- bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

ggg_predictions <- predict(final_wf, new_data = test, type="class")

ggg_predictions$type <- ggg_predictions$.pred_class
ggg_predictions$id <- test$id
ggg_final <- ggg_predictions %>%
  select(id, type)

write.csv(ggg_final, "./GhostGhoulsandGoblins/bart.csv", row.names = F)

## naive bayes
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

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

nb_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(nb_model)

# Tune smoothness and Laplace here
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities

# Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats=1)

CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Predict
bestTune <- CV_results %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

ggg_predictions <- predict(final_wf, new_data = test, type="class")

ggg_predictions$type <- ggg_predictions$.pred_class
ggg_predictions$id <- test$id
ggg_final <- ggg_predictions %>%
  select(id, type)

write.csv(ggg_final, "./GhostGhoulsandGoblins/naivebayes.csv", row.names = F)
