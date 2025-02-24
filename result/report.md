# Genta AutoML Comprehensive Report

## Overview

This report provides an overview of the model evaluation, configuration, SHAP analysis, and additional session state details gathered during the AutoML process.

## Model Evaluation Summary
- **Best Model**: lgbm
- **Best Test Score**: 1.0
- **Metric**: accuracy

## SHAP Analysis
SHAP values were computed. A summary plot is shown in the UI, but is not embedded here.

## Additional Session State Details
### X_selected
```json
null
```

### model_config
```json
"ModelConfig(target_column=' species', task_type=<TaskType.CLASSIFICATION: 'Classification'>, models=['lgbm', 'rf'], time_budget=60, n_clusters=3, random_seed=42, verbosity=1, show_shap=False, metric_name='accuracy', target_score=0.8, hyperparameter_configs={}, auto_model_selection=False)"
```

### y_pd
```json
"[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2\n 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2\n 2 2]"
```

### evaluation_report
```json
{
  "Best Model": "lgbm",
  "Best Test Score": 1.0,
  "Metric": "accuracy",
  "Evaluation Table": [
    {
      "Model": "lgbm",
      "Train Score": 0.975,
      "Test Score": 1.0
    },
    {
      "Model": "rf",
      "Train Score": 1.0,
      "Test Score": 1.0
    }
  ]
}
```

### shap_values
```json
"[[ 0.00394181  0.18520769 -0.15008867 ... -0.08530968  0.03509342\n  -0.06505759]\n [ 0.02295859 -0.02493626  0.12401382 ...  0.34996297  0.05817891\n   0.0145159 ]\n [-0.21609119  0.05665589  0.02383159 ...  0.27270329  0.11208774\n   0.        ]\n ...\n [-0.12300151  0.         -0.30705049 ...  0.10270074  0.15986431\n   0.        ]\n [ 0.63113347  0.07782094 -0.24976514 ...  0.05732456  0.00110787\n   0.00796106]\n [-0.15824419  0.          0.83064563 ...  0.23487562 -0.15255232\n   0.        ]]"
```

### X_pd
```json
"     sepal length   sepal width   petal length   petal width\n0             5.1           3.5            1.4           0.2\n1             4.9           3.0            1.4           0.2\n2             4.7           3.2            1.3           0.2\n3             4.6           3.1            1.5           0.2\n4             5.0           3.6            1.4           0.2\n..            ...           ...            ...           ...\n145           6.7           3.0            5.2           2.3\n146           6.3           2.5            5.0           1.9\n147           6.5           3.0            5.2           2.0\n148           6.2           3.4            5.4           2.3\n149           5.9           3.0            5.1           1.8\n\n[150 rows x 4 columns]"
```

### trained_model
```json
{
  "model_name": "fake_model_object"
}
```

### target_metrics
```json
null
```

### model
```json
null
```
