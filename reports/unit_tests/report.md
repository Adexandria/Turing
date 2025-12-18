
# Turing Test Execution Report



---



## Environment Information


| Parameter      | Value                      |
|:---------------|:---------------------------|
| Timestamp      | 2025-12-04 18:14:18        |
| Context        | TURING                     |
| Python Version | 3.12.12                    |
| Platform       | macOS-15.6-arm64-arm-64bit |
| Architecture   | arm64                      |


---


## Executive Summary


**Overall Status:** MOSTLY PASSED


**Success Rate:** 91.2%


| Metric       |   Count |
|:-------------|--------:|
| Total Tests  |      34 |
| Passed       |      31 |
| Failed       |       3 |
| Success Rate |   91.2% |


**Visual Progress:**


```
Progress: [█████████████████████████████████████████████░░░░░] 91.2%
Passed: 31/34 tests
```


---


## UNIT Tests


### Statistics


| Status   |      Count |
|:---------|-----------:|
| Total    |         34 |
| Passed   | 31 (91.2%) |
| Failed   |   3 (8.8%) |


### Test Results


| Module          | Test Case                                          | Result   | Time   | Message                                              |
|:----------------|:---------------------------------------------------|:---------|:-------|:-----------------------------------------------------|
| test_api.py     | test_health_check_returns_ok                       | PASS     | 0.01s  |                                                      |
| test_api.py     | test_predict_success_java                          | PASS     | 0.02s  |                                                      |
| test_api.py     | test_predict_success_python                        | PASS     | 0.00s  |                                                      |
| test_api.py     | test_predict_success_pharo                         | PASS     | 0.00s  |                                                      |
| test_api.py     | test_predict_missing_texts                         | PASS     | 0.00s  |                                                      |
| test_api.py     | test_predict_missing_language                      | PASS     | 0.00s  |                                                      |
| test_api.py     | test_predict_empty_texts                           | PASS     | 0.00s  |                                                      |
| test_api.py     | test_predict_error_handling                        | PASS     | 0.00s  |                                                      |
| test_api.py     | test_predict_invalid_language                      | PASS     | 0.00s  |                                                      |
| test_api.py     | test_prediction_request_valid                      | PASS     | 0.00s  |                                                      |
| test_api.py     | test_prediction_response_valid                     | PASS     | 0.00s  |                                                      |
| test_config.py  | test_proj_root_is_correctly_identified             | PASS     | 0.00s  |                                                      |
| test_config.py  | test_directory_paths_are_correctly_structured      | PASS     | 0.00s  |                                                      |
| test_config.py  | test_dataset_constants_are_valid                   | PASS     | 0.00s  |                                                      |
| test_config.py  | test_labels_map_and_total_categories_are_correct   | PASS     | 0.00s  |                                                      |
| test_config.py  | test_numeric_parameters_are_positive               | PASS     | 0.00s  |                                                      |
| test_config.py  | test_load_dotenv_is_called_on_module_load          | PASS     | 0.00s  |                                                      |
| test_dataset.py | test_initialization_paths_are_correct              | FAIL     | 0.00s  | turing/tests/unit/test_dataset.py:25: AssertionError |
| test_dataset.py | test_format_labels_for_csv[input_labels0-[1, 0,... | PASS     | 0.00s  |                                                      |
| test_dataset.py | test_format_labels_for_csv[[1, 0, 1]-[1, 0, 1]]    | PASS     | 0.00s  |                                                      |
| test_dataset.py | test_format_labels_for_csv[input_labels2-[]]       | PASS     | 0.00s  |                                                      |
| test_dataset.py | test_format_labels_for_csv[None-None]              | PASS     | 0.00s  |                                                      |
| test_dataset.py | test_get_dataset_raises_file_not_found             | PASS     | 0.00s  |                                                      |
| test_dataset.py | test_get_dataset_success_and_label_parsing         | FAIL     | 0.00s  | turing/dataset.py:128: FileNotFoundError             |
| test_model.py   | test_model_initialization[randomForestTfIdf]       | PASS     | 0.00s  |                                                      |
| test_model.py   | test_model_initialization[codeBerta]               | PASS     | 0.00s  |                                                      |
| test_model.py   | test_model_setup[randomForestTfIdf]                | PASS     | 0.00s  |                                                      |
| test_model.py   | test_model_setup[codeBerta]                        | PASS     | 0.93s  |                                                      |
| test_model.py   | test_model_train[randomForestTfIdf]                | PASS     | 2.66s  |                                                      |
| test_model.py   | test_model_train[codeBerta]                        | PASS     | 7.22s  |                                                      |
| test_model.py   | test_model_evaluate[randomForestTfIdf]             | PASS     | 1.31s  |                                                      |
| test_model.py   | test_model_evaluate[codeBerta]                     | FAIL     | 8.83s  | turing/tests/unit/test_model.py:101: AssertionError  |
| test_model.py   | test_model_predict[randomForestTfIdf]              | PASS     | 1.21s  |                                                      |
| test_model.py   | test_model_predict[codeBerta]                      | PASS     | 5.98s  |                                                      |


---


> **ERROR**: 3 test(s) failed. Please review the error messages above.



---



*Report generated on 2025-12-04 at 18:14:18*


*Powered by Turing Test Suite*
