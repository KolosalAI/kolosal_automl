# ML Training Engine Benchmark Report

Generated on: 2025-03-11 13:55:27

## System Information

- **Platform:** Windows-10-10.0.22631-SP0
- **Processor:** 12th Gen Intel(R) Core(TM) i7-12700H
- **CPU Cores:** 14 physical, 20 logical
- **Memory:** 63.67 GB total, 44.51 GB available
- **Python Version:** 3.10.11

## Performance Summary

| Dataset | Task | Samples | Features | Best Strategy | Best Model | Training Time (s) | Throughput (samples/s) | Peak Memory (MB) |
|---------|------|---------|----------|---------------|------------|-------------------|------------------------|------------------|
| Adult Census Income | classification | 48842 | 14 | random_search | logistic_regression | 11.26 | 3471.52 | 21.92 |
| Blood Transfusion | classification | 748 | 4 | random_search | logistic_regression | 0.14 | 4214.76 | 2.15 |
| German Credit | classification | 1000 | 20 | grid_search | logistic_regression | 0.45 | 1787.05 | 4.14 |
| Diabetes | classification | 768 | 8 | random_search | logistic_regression | 0.21 | 2922.25 | 2.21 |
| Breast Cancer Wisconsin | classification | 286 | 9 | random_search | logistic_regression | 0.21 | 1107.42 | 1.08 |
| Boston Housing | regression | 506 | 13 | random_search | linear_regression | 0.14 | 2861.43 | 1.92 |
| California Housing | regression | 20640 | 9 | random_search | linear_regression | 2.03 | 8139.25 | 138.92 |
| Medical Charges | regression | 50000 | 11 | grid_search | linear_regression | 6.63 | 6036.69 | 145.63 |

## Detailed Results

### Adult Census Income (classification)

- **Samples:** 48842
- **Features:** 14

#### Optimization Strategy Comparison

| Strategy | Total Time (s) | Peak Memory (MB) | Avg CPU (%) | Max Memory (MB) |
|----------|---------------|------------------|-------------|----------------|
| grid_search | 70.53 | 22.62 | 17.89 | 235.59 |
| random_search | 41.12 | 22.02 | 23.46 | 237.72 |
| adaptive_surrogate_assisted_hyperparameter_tuning | 125.62 | 22.17 | 4.40 | 241.79 |

#### grid_search Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 21.93 | 0.41 | 1782.09 | 24062.33 | 21.90 |
| logistic_regression | 23.14 | 0.02 | 1688.29 | 624901.34 | 22.50 |
| gradient_boosting | 24.99 | 0.05 | 1563.33 | 207851.37 | 22.62 |

**Performance Metrics:**

| Model | accuracy | precision | recall | f1 | roc_auc |
|-------|---|---|---|---|---|
| random_forest | 0.8319 | 0.8220 | 0.8319 | 0.8225 | 0.8798 |
| logistic_regression | 0.7996 | 0.7809 | 0.7996 | 0.7786 | 0.7872 |
| gradient_boosting | 0.8381 | 0.8291 | 0.8381 | 0.8294 | 0.8871 |

#### random_search Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 16.51 | 0.22 | 2367.16 | 44072.16 | 21.55 |
| logistic_regression | 11.26 | 0.02 | 3471.52 | 625216.00 | 21.92 |
| gradient_boosting | 13.06 | 0.06 | 2991.99 | 166489.87 | 22.02 |

**Performance Metrics:**

| Model | accuracy | precision | recall | f1 | roc_auc |
|-------|---|---|---|---|---|
| random_forest | 0.8313 | 0.8213 | 0.8313 | 0.8220 | 0.8795 |
| logistic_regression | 0.7996 | 0.7809 | 0.7996 | 0.7786 | 0.7872 |
| gradient_boosting | 0.8381 | 0.8291 | 0.8381 | 0.8294 | 0.8871 |

#### adaptive_surrogate_assisted_hyperparameter_tuning Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 53.27 | 0.24 | 733.51 | 41392.18 | 21.74 |
| logistic_regression | 32.55 | 0.02 | 1200.43 | 626305.46 | 22.16 |
| gradient_boosting | 39.48 | 0.05 | 989.66 | 198122.72 | 22.17 |

**Performance Metrics:**

| Model | accuracy | precision | recall | f1 | roc_auc |
|-------|---|---|---|---|---|
| random_forest | 0.8323 | 0.8224 | 0.8323 | 0.8229 | 0.8802 |
| logistic_regression | 0.7996 | 0.7809 | 0.7996 | 0.7786 | 0.7872 |
| gradient_boosting | 0.8381 | 0.8291 | 0.8381 | 0.8294 | 0.8871 |
### Blood Transfusion (classification)

- **Samples:** 748
- **Features:** 4

#### Optimization Strategy Comparison

| Strategy | Total Time (s) | Peak Memory (MB) | Avg CPU (%) | Max Memory (MB) |
|----------|---------------|------------------|-------------|----------------|
| grid_search | 1.60 | 2.04 | 16.23 | 200.61 |
| random_search | 1.58 | 2.15 | 15.24 | 201.36 |
| adaptive_surrogate_assisted_hyperparameter_tuning | 8.51 | 1.71 | 3.19 | 201.73 |

#### grid_search Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 0.86 | 0.16 | 694.72 | 951.14 | 0.50 |
| logistic_regression | 0.17 | 0.02 | 3470.83 | 9365.91 | 2.04 |
| gradient_boosting | 0.36 | 0.03 | 1650.69 | 4439.92 | 2.04 |

**Performance Metrics:**

| Model | accuracy | precision | recall | f1 | roc_auc |
|-------|---|---|---|---|---|
| random_forest | 0.7667 | 0.7522 | 0.7667 | 0.7576 | 0.7671 |
| logistic_regression | 0.7733 | 0.7441 | 0.7733 | 0.7136 | 0.7880 |
| gradient_boosting | 0.7867 | 0.7653 | 0.7867 | 0.7671 | 0.7574 |

#### random_search Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 0.86 | 0.18 | 698.89 | 842.74 | 0.50 |
| logistic_regression | 0.14 | 0.02 | 4214.76 | 9974.72 | 2.15 |
| gradient_boosting | 0.36 | 0.03 | 1656.60 | 4800.37 | 2.15 |

**Performance Metrics:**

| Model | accuracy | precision | recall | f1 | roc_auc |
|-------|---|---|---|---|---|
| random_forest | 0.7667 | 0.7522 | 0.7667 | 0.7576 | 0.7671 |
| logistic_regression | 0.7733 | 0.7441 | 0.7733 | 0.7136 | 0.7880 |
| gradient_boosting | 0.7867 | 0.7653 | 0.7867 | 0.7671 | 0.7574 |

#### adaptive_surrogate_assisted_hyperparameter_tuning Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 3.89 | 0.07 | 153.59 | 2133.72 | 0.51 |
| logistic_regression | Error: float division by zero | - | - | - | - |
| gradient_boosting | 2.56 | 0.03 | 233.36 | 4800.88 | 1.71 |

**Performance Metrics:**

| Model | accuracy | precision | recall | f1 | roc_auc |
|-------|---|---|---|---|---|
| random_forest | 0.7400 | 0.7279 | 0.7400 | 0.7331 | 0.7304 |
| gradient_boosting | 0.7867 | 0.7653 | 0.7867 | 0.7671 | 0.7574 |
### German Credit (classification)

- **Samples:** 1000
- **Features:** 20

#### Optimization Strategy Comparison

| Strategy | Total Time (s) | Peak Memory (MB) | Avg CPU (%) | Max Memory (MB) |
|----------|---------------|------------------|-------------|----------------|
| grid_search | 2.66 | 4.14 | 18.00 | 209.24 |
| random_search | 2.52 | 4.18 | 16.20 | 210.83 |
| adaptive_surrogate_assisted_hyperparameter_tuning | 10.75 | 2.19 | 3.44 | 206.39 |

#### grid_search Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 1.38 | 0.16 | 580.89 | 1250.05 | 1.00 |
| logistic_regression | 0.45 | 0.01 | 1787.05 | 19998.11 | 4.14 |
| gradient_boosting | 0.63 | 0.03 | 1262.84 | 6639.66 | 4.14 |

**Performance Metrics:**

| Model | accuracy | precision | recall | f1 | roc_auc |
|-------|---|---|---|---|---|
| random_forest | 0.7350 | 0.7214 | 0.7350 | 0.7248 | 0.7628 |
| logistic_regression | 0.7400 | 0.7247 | 0.7400 | 0.7270 | 0.7852 |
| gradient_boosting | 0.7500 | 0.7372 | 0.7500 | 0.7394 | 0.7561 |

#### random_search Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 1.23 | 0.16 | 649.08 | 1271.39 | 1.01 |
| logistic_regression | 0.45 | 0.01 | 1768.11 | 19039.49 | 4.18 |
| gradient_boosting | 0.64 | 0.03 | 1258.15 | 6678.88 | 4.18 |

**Performance Metrics:**

| Model | accuracy | precision | recall | f1 | roc_auc |
|-------|---|---|---|---|---|
| random_forest | 0.7600 | 0.7522 | 0.7600 | 0.7548 | 0.7746 |
| logistic_regression | 0.7650 | 0.7527 | 0.7650 | 0.7523 | 0.7805 |
| gradient_boosting | 0.7600 | 0.7467 | 0.7600 | 0.7461 | 0.7801 |

#### adaptive_surrogate_assisted_hyperparameter_tuning Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 4.41 | 0.08 | 181.26 | 2496.70 | 1.01 |
| logistic_regression | 2.75 | 0.01 | 291.16 | 20001.93 | 2.12 |
| gradient_boosting | 3.46 | 0.02 | 230.92 | 10145.87 | 2.19 |

**Performance Metrics:**

| Model | accuracy | precision | recall | f1 | roc_auc |
|-------|---|---|---|---|---|
| random_forest | 0.7300 | 0.7150 | 0.7300 | 0.7186 | 0.7277 |
| logistic_regression | 0.7200 | 0.6929 | 0.7200 | 0.6869 | 0.7562 |
| gradient_boosting | 0.7300 | 0.7095 | 0.7300 | 0.7095 | 0.7497 |
### Diabetes (classification)

- **Samples:** 768
- **Features:** 8

#### Optimization Strategy Comparison

| Strategy | Total Time (s) | Peak Memory (MB) | Avg CPU (%) | Max Memory (MB) |
|----------|---------------|------------------|-------------|----------------|
| grid_search | 1.87 | 2.20 | 15.87 | 207.61 |
| random_search | 1.86 | 2.21 | 17.56 | 207.48 |
| adaptive_surrogate_assisted_hyperparameter_tuning | 9.16 | 1.51 | 3.08 | 207.58 |

#### grid_search Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 0.99 | 0.15 | 618.26 | 1025.60 | 0.57 |
| logistic_regression | 0.21 | 0.01 | 2866.04 | 15360.46 | 2.20 |
| gradient_boosting | 0.47 | 0.03 | 1313.41 | 5131.95 | 2.20 |

**Performance Metrics:**

| Model | accuracy | precision | recall | f1 | roc_auc |
|-------|---|---|---|---|---|
| random_forest | 0.7532 | 0.7497 | 0.7532 | 0.7509 | 0.8100 |
| logistic_regression | 0.7078 | 0.7005 | 0.7078 | 0.7026 | 0.8233 |
| gradient_boosting | 0.7662 | 0.7610 | 0.7662 | 0.7614 | 0.8317 |

#### random_search Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 0.97 | 0.16 | 635.94 | 985.00 | 0.59 |
| logistic_regression | 0.21 | 0.01 | 2922.25 | 15400.75 | 2.21 |
| gradient_boosting | 0.48 | 0.03 | 1266.80 | 5057.61 | 2.21 |

**Performance Metrics:**

| Model | accuracy | precision | recall | f1 | roc_auc |
|-------|---|---|---|---|---|
| random_forest | 0.7532 | 0.7497 | 0.7532 | 0.7509 | 0.8100 |
| logistic_regression | 0.7078 | 0.7005 | 0.7078 | 0.7026 | 0.8233 |
| gradient_boosting | 0.7662 | 0.7610 | 0.7662 | 0.7614 | 0.8317 |

#### adaptive_surrogate_assisted_hyperparameter_tuning Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 3.80 | 0.08 | 161.62 | 1920.73 | 0.60 |
| logistic_regression | 2.16 | 0.01 | 283.86 | 15562.53 | 1.35 |
| gradient_boosting | 3.08 | 0.03 | 199.18 | 5133.42 | 1.35 |

**Performance Metrics:**

| Model | accuracy | precision | recall | f1 | roc_auc |
|-------|---|---|---|---|---|
| random_forest | 0.7273 | 0.7231 | 0.7273 | 0.7247 | 0.8057 |
| logistic_regression | 0.7078 | 0.7005 | 0.7078 | 0.7026 | 0.8233 |
| gradient_boosting | 0.7662 | 0.7610 | 0.7662 | 0.7614 | 0.8317 |
### Breast Cancer Wisconsin (classification)

- **Samples:** 286
- **Features:** 9

#### Optimization Strategy Comparison

| Strategy | Total Time (s) | Peak Memory (MB) | Avg CPU (%) | Max Memory (MB) |
|----------|---------------|------------------|-------------|----------------|
| grid_search | 1.43 | 1.57 | 17.26 | 207.96 |
| random_search | 1.42 | 1.42 | 16.05 | 208.29 |
| adaptive_surrogate_assisted_hyperparameter_tuning | 10.32 | 1.94 | 2.83 | 209.29 |

#### grid_search Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 0.68 | 0.08 | 334.05 | 737.89 | 0.48 |
| logistic_regression | 0.21 | 0.01 | 1089.74 | 5782.08 | 1.07 |
| gradient_boosting | 0.42 | 0.03 | 541.59 | 1901.78 | 1.07 |

**Performance Metrics:**

| Model | accuracy | precision | recall | f1 | roc_auc |
|-------|---|---|---|---|---|
| random_forest | 0.7069 | 0.6697 | 0.7069 | 0.6686 | 0.6155 |
| logistic_regression | 0.7414 | 0.7435 | 0.7414 | 0.6790 | 0.6141 |
| gradient_boosting | 0.7241 | 0.8016 | 0.7241 | 0.6241 | 0.5997 |

#### random_search Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 0.66 | 0.08 | 346.96 | 724.29 | 0.50 |
| logistic_regression | 0.21 | 0.01 | 1107.42 | 5799.45 | 1.08 |
| gradient_boosting | 0.44 | 0.02 | 513.13 | 2425.32 | 1.08 |

**Performance Metrics:**

| Model | accuracy | precision | recall | f1 | roc_auc |
|-------|---|---|---|---|---|
| random_forest | 0.7069 | 0.6697 | 0.7069 | 0.6686 | 0.6155 |
| logistic_regression | 0.7414 | 0.7435 | 0.7414 | 0.6790 | 0.6141 |
| gradient_boosting | 0.7241 | 0.8016 | 0.7241 | 0.6241 | 0.5997 |

#### adaptive_surrogate_assisted_hyperparameter_tuning Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 4.10 | 0.17 | 55.56 | 332.09 | 0.55 |
| logistic_regression | 2.19 | 0.02 | 104.11 | 3713.36 | 1.94 |
| gradient_boosting | 3.79 | 0.04 | 60.12 | 1442.77 | 1.94 |

**Performance Metrics:**

| Model | accuracy | precision | recall | f1 | roc_auc |
|-------|---|---|---|---|---|
| random_forest | 0.7586 | 0.7499 | 0.7586 | 0.7215 | 0.6643 |
| logistic_regression | 0.7586 | 0.8201 | 0.7586 | 0.6917 | 0.6198 |
| gradient_boosting | 0.7069 | 0.6697 | 0.7069 | 0.6686 | 0.6133 |
### Boston Housing (regression)

- **Samples:** 506
- **Features:** 13

#### Optimization Strategy Comparison

| Strategy | Total Time (s) | Peak Memory (MB) | Avg CPU (%) | Max Memory (MB) |
|----------|---------------|------------------|-------------|----------------|
| grid_search | 1.46 | 1.38 | 21.51 | 210.46 |
| random_search | 1.44 | 1.92 | 23.82 | 211.36 |
| adaptive_surrogate_assisted_hyperparameter_tuning | 12.82 | 3.20 | 7.23 | 215.54 |

#### grid_search Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 0.95 | 0.08 | 425.99 | 1274.92 | 0.55 |
| linear_regression | 0.16 | 0.01 | 2523.44 | 10200.25 | 1.38 |
| svr | 0.25 | 0.02 | 1642.45 | 5088.12 | 1.38 |

**Performance Metrics:**

| Model | mse | rmse | mae | r2 |
|-------|---|---|---|---|
| random_forest | 8.0252 | 2.8329 | 2.0469 | 0.8906 |
| linear_regression | 23.1671 | 4.8132 | 3.2034 | 0.6841 |
| svr | 12.1583 | 3.4869 | 1.9398 | 0.8342 |

#### random_search Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 0.95 | 0.07 | 427.11 | 1448.90 | 0.57 |
| linear_regression | 0.14 | 0.01 | 2861.43 | 10198.79 | 1.92 |
| svr | 0.25 | 0.02 | 1642.08 | 5116.96 | 1.92 |

**Performance Metrics:**

| Model | mse | rmse | mae | r2 |
|-------|---|---|---|---|
| random_forest | 8.1415 | 2.8533 | 2.0385 | 0.8890 |
| linear_regression | 23.1671 | 4.8132 | 3.2034 | 0.6841 |
| svr | 12.1583 | 3.4869 | 1.9398 | 0.8342 |

#### adaptive_surrogate_assisted_hyperparameter_tuning Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 7.95 | 0.27 | 50.81 | 378.66 | 0.64 |
| linear_regression | 0.39 | 0.02 | 1046.05 | 6034.80 | 3.20 |
| svr | 4.16 | 0.03 | 97.13 | 3596.36 | 3.20 |

**Performance Metrics:**

| Model | mse | rmse | mae | r2 |
|-------|---|---|---|---|
| random_forest | 8.4215 | 2.9020 | 2.0296 | 0.8852 |
| linear_regression | 23.1671 | 4.8132 | 3.2034 | 0.6841 |
| svr | 12.1583 | 3.4869 | 1.9398 | 0.8342 |
### California Housing (regression)

- **Samples:** 20640
- **Features:** 9

#### Optimization Strategy Comparison

| Strategy | Total Time (s) | Peak Memory (MB) | Avg CPU (%) | Max Memory (MB) |
|----------|---------------|------------------|-------------|----------------|
| grid_search | 82.99 | 138.98 | 20.30 | 601.00 |
| random_search | 58.10 | 138.92 | 30.41 | 597.54 |
| adaptive_surrogate_assisted_hyperparameter_tuning | 115.26 | 138.97 | 4.06 | 605.55 |

#### grid_search Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 51.16 | 0.71 | 322.73 | 5800.28 | 6.19 |
| linear_regression | 2.62 | 0.02 | 6290.97 | 262354.53 | 138.98 |
| svr | 27.81 | 0.66 | 593.78 | 6266.53 | 138.98 |

**Performance Metrics:**

| Model | mse | rmse | mae | r2 |
|-------|---|---|---|---|
| random_forest | 2571999923.6521 | 50714.8886 | 32659.1779 | 0.8037 |
| linear_regression | 5046614056.0457 | 71039.5246 | 52239.8095 | 0.6149 |
| svr | 7904893578.9908 | 88909.4684 | 62540.4782 | 0.3968 |

#### random_search Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 27.04 | 0.52 | 610.57 | 7922.40 | 6.15 |
| linear_regression | 2.03 | 0.02 | 8139.25 | 172879.82 | 138.92 |
| svr | 27.86 | 0.62 | 592.65 | 6607.61 | 138.92 |

**Performance Metrics:**

| Model | mse | rmse | mae | r2 |
|-------|---|---|---|---|
| random_forest | 2571999923.6521 | 50714.8886 | 32659.1779 | 0.8037 |
| linear_regression | 5046614056.0457 | 71039.5246 | 52239.8095 | 0.6149 |
| svr | 7904893578.9908 | 88909.4684 | 62540.4782 | 0.3968 |

#### adaptive_surrogate_assisted_hyperparameter_tuning Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 59.00 | 0.54 | 279.87 | 7575.99 | 6.17 |
| linear_regression | 3.28 | 0.02 | 5032.48 | 206115.17 | 138.97 |
| svr | 51.80 | 0.59 | 318.75 | 6944.96 | 138.97 |

**Performance Metrics:**

| Model | mse | rmse | mae | r2 |
|-------|---|---|---|---|
| random_forest | 2571999923.6521 | 50714.8886 | 32659.1779 | 0.8037 |
| linear_regression | 5046614056.0457 | 71039.5246 | 52239.8095 | 0.6149 |
| svr | 7904893578.9908 | 88909.4684 | 62540.4782 | 0.3968 |
### Medical Charges (regression)

- **Samples:** 50000
- **Features:** 11

#### Optimization Strategy Comparison

| Strategy | Total Time (s) | Peak Memory (MB) | Avg CPU (%) | Max Memory (MB) |
|----------|---------------|------------------|-------------|----------------|
| grid_search | 370.89 | 145.63 | 27.10 | 648.96 |
| random_search | 559.81 | 145.75 | 16.67 | 651.59 |
| adaptive_surrogate_assisted_hyperparameter_tuning | 762.04 | 17.91 | 7.78 | 469.71 |

#### grid_search Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 89.28 | 0.64 | 448.01 | 15628.01 | 17.28 |
| linear_regression | 6.63 | 0.02 | 6036.69 | 499197.11 | 145.63 |
| svr | 267.61 | 6.70 | 149.47 | 1491.69 | 145.63 |

**Performance Metrics:**

| Model | mse | rmse | mae | r2 |
|-------|---|---|---|---|
| random_forest | 4098155.3411 | 2024.3901 | 718.8199 | 0.9319 |
| linear_regression | 14494013.3841 | 3807.1004 | 1803.0982 | 0.7592 |
| svr | 19540669.9370 | 4420.4830 | 1203.5738 | 0.6753 |

#### random_search Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 191.76 | 0.97 | 208.59 | 10305.90 | 17.37 |
| linear_regression | 10.07 | 0.03 | 3972.52 | 327651.84 | 145.75 |
| svr | 350.77 | 6.20 | 114.03 | 1612.20 | 145.75 |

**Performance Metrics:**

| Model | mse | rmse | mae | r2 |
|-------|---|---|---|---|
| random_forest | 4098155.3411 | 2024.3901 | 718.8199 | 0.9319 |
| linear_regression | 14494013.3841 | 3807.1004 | 1803.0982 | 0.7592 |
| svr | 19540669.9370 | 4420.4830 | 1203.5738 | 0.6753 |

#### adaptive_surrogate_assisted_hyperparameter_tuning Models

| Model | Training Time (s) | Inference Time (s) | Training Throughput | Inference Throughput | Peak Memory (MB) |
|-------|------------------|-------------------|-------------------|---------------------|----------------|
| random_forest | 201.89 | 0.21 | 198.12 | 47447.58 | 17.33 |
| linear_regression | 12.53 | 0.02 | 3192.83 | 639463.34 | 17.69 |
| svr | 540.04 | 7.33 | 74.07 | 1364.41 | 17.91 |

**Performance Metrics:**

| Model | mse | rmse | mae | r2 |
|-------|---|---|---|---|
| random_forest | 4112147.4112 | 2027.8430 | 716.5378 | 0.9317 |
| linear_regression | 14494013.3841 | 3807.1004 | 1803.0982 | 0.7592 |
| svr | 19540669.9370 | 4420.4830 | 1203.5738 | 0.6753 |

## Conclusion

This benchmark report provides a comprehensive analysis of the ML Training Engine's performance across various datasets and optimization strategies. The results demonstrate the trade-offs between training time, memory usage, and model performance for different configurations.

