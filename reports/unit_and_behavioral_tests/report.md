
# Test Execution Report


### Environment


```text
     Parameter                     Value
     Timestamp       2025-11-27 15:44:47
       Context                    turing
Python Version                   3.12.12
      Platform Windows-11-10.0.26100-SP0
```


### Executive Summary


```text
 Total  Passed  Failed Success Rate
    66      35      31        53.0%
```


Detailed Breakdown:


### BEHAVIORAL Tests


```text
                       Module                                          Test Case     Result  Time                                                      Message
          test_directional.py              test_java_directional_add_deprecation [ FAILED ] 0.30s turing\tests\behavioral\test_directional.py:16: Assertion...
          test_directional.py                test_python_directional_remove_todo [ FAILED ] 0.15s turing\tests\behavioral\test_directional.py:31: Assertion...
          test_directional.py          test_pharo_directional_add_responsibility [ FAILED ] 0.13s turing\tests\behavioral\test_directional.py:49: Assertion...
          test_directional.py            test_java_directional_contrast_rational [ FAILED ] 0.12s turing\tests\behavioral\test_directional.py:70: Assertion...
          test_directional.py              test_python_directional_contrast_todo [ FAILED ] 0.12s turing\tests\behavioral\test_directional.py:87: Assertion...
          test_directional.py      test_pharo_directional_contrast_collaborators [ FAILED ] 0.13s turing\tests\behavioral\test_directional.py:112: Assertio...
          test_directional.py      test_java_directional_shift_summary_to_expand [ FAILED ] 0.12s turing\tests\behavioral\test_directional.py:132: Assertio...
          test_directional.py  test_python_directional_shift_summary_to_devnotes [ FAILED ] 0.12s turing\tests\behavioral\test_directional.py:152: Assertio...
          test_directional.py            test_pharo_directional_shift_to_example [ FAILED ] 0.12s turing\tests\behavioral\test_directional.py:173: Assertio...
           test_invariance.py test_python_invariance_parameters[:param user_i... [ FAILED ] 0.22s turing\tests\behavioral\test_invariance.py:15: AssertionE...
           test_invariance.py test_python_invariance_parameters[:PARAM USER_I... [ FAILED ] 0.07s turing\tests\behavioral\test_invariance.py:15: AssertionE...
           test_invariance.py test_python_invariance_parameters[  :param user... [ FAILED ] 0.06s turing\tests\behavioral\test_invariance.py:15: AssertionE...
           test_invariance.py test_python_invariance_parameters[:param user_i... [ FAILED ] 0.06s turing\tests\behavioral\test_invariance.py:15: AssertionE...
           test_invariance.py                   test_java_invariance_deprecation [ FAILED ] 0.13s turing\tests\behavioral\test_invariance.py:26: AssertionE...
           test_invariance.py                     test_python_invariance_summary [ FAILED ] 0.13s turing\tests\behavioral\test_invariance.py:45: AssertionE...
           test_invariance.py                       test_pharo_invariance_intent [ FAILED ] 0.13s turing\tests\behavioral\test_invariance.py:64: AssertionE...
           test_invariance.py            test_python_invariance_typos_parameters [ FAILED ] 0.07s turing\tests\behavioral\test_invariance.py:85: AssertionE...
           test_invariance.py              test_java_invariance_semantic_summary   [ PASS ] 0.32s                                                             
test_minimum_functionality.py test_java_mft[test getfilestatus and related li...   [ PASS ] 0.06s                                                             
test_minimum_functionality.py test_java_mft[/* @deprecated Use something else... [ FAILED ] 0.06s turing\tests\behavioral\test_minimum_functionality.py:17:...
test_minimum_functionality.py test_java_mft[code source of this file http gre... [ FAILED ] 0.06s turing\tests\behavioral\test_minimum_functionality.py:17:...
test_minimum_functionality.py test_java_mft[this is balanced if each pool is ... [ FAILED ] 0.06s turing\tests\behavioral\test_minimum_functionality.py:17:...
test_minimum_functionality.py test_java_mft[// For internal use only.-expecte... [ FAILED ] 0.06s turing\tests\behavioral\test_minimum_functionality.py:17:...
test_minimum_functionality.py test_java_mft[this impl delegates to the old fi... [ FAILED ] 0.07s turing\tests\behavioral\test_minimum_functionality.py:17:...
test_minimum_functionality.py test_java_mft[/** Usage: new MyClass(arg1). */-... [ FAILED ] 0.07s turing\tests\behavioral\test_minimum_functionality.py:17:...
test_minimum_functionality.py test_python_mft[a service specific account of t...   [ PASS ] 0.06s                                                             
test_minimum_functionality.py test_python_mft[:param user_id: The ID of the u... [ FAILED ] 0.07s turing\tests\behavioral\test_minimum_functionality.py:29:...
test_minimum_functionality.py test_python_mft[# TODO: Refactor this entire bl... [ FAILED ] 0.07s turing\tests\behavioral\test_minimum_functionality.py:29:...
test_minimum_functionality.py test_python_mft[use this class if you want acce...   [ PASS ] 0.06s                                                             
test_minimum_functionality.py test_python_mft[# create a new list by filterin... [ FAILED ] 0.08s turing\tests\behavioral\test_minimum_functionality.py:29:...
test_minimum_functionality.py test_pharo_mft[i am a simple arrow like arrowhe...   [ PASS ] 0.07s                                                             
test_minimum_functionality.py test_pharo_mft[the example below shows how to c...   [ PASS ] 0.07s                                                             
test_minimum_functionality.py test_pharo_mft[i provide a data structure indep... [ FAILED ] 0.06s turing\tests\behavioral\test_minimum_functionality.py:43:...
test_minimum_functionality.py test_pharo_mft[the cache is cleared after each ... [ FAILED ] 0.07s turing\tests\behavioral\test_minimum_functionality.py:43:...
test_minimum_functionality.py test_pharo_mft[it is possible hovewer to custom...   [ PASS ] 0.07s                                                             
test_minimum_functionality.py test_pharo_mft[collaborators: BlElement, BlSpac... [ FAILED ] 0.07s turing\tests\behavioral\test_minimum_functionality.py:43:...
```


### UNIT Tests


```text
          Module                                          Test Case     Result  Time                                              Message
  test_config.py             test_proj_root_is_correctly_identified   [ PASS ] 0.00s                                                     
  test_config.py      test_directory_paths_are_correctly_structured   [ PASS ] 0.00s                                                     
  test_config.py                   test_dataset_constants_are_valid   [ PASS ] 0.00s                                                     
  test_config.py   test_labels_map_and_total_categories_are_correct   [ PASS ] 0.00s                                                     
  test_config.py               test_numeric_parameters_are_positive   [ PASS ] 0.00s                                                     
  test_config.py          test_load_dotenv_is_called_on_module_load   [ PASS ] 0.00s                                                     
 test_dataset.py              test_initialization_paths_are_correct [ FAILED ] 0.00s turing\tests\unit\test_dataset.py:24: AssertionError
 test_dataset.py test_format_labels_for_csv[input_labels0-[1, 0,...   [ PASS ] 0.00s                                                     
 test_dataset.py    test_format_labels_for_csv[[1, 0, 1]-[1, 0, 1]]   [ PASS ] 0.00s                                                     
 test_dataset.py       test_format_labels_for_csv[input_labels2-[]]   [ PASS ] 0.00s                                                     
 test_dataset.py              test_format_labels_for_csv[None-None]   [ PASS ] 0.00s                                                     
 test_dataset.py             test_get_dataset_raises_file_not_found   [ PASS ] 0.00s                                                     
 test_dataset.py         test_get_dataset_success_and_label_parsing   [ PASS ] 0.48s                                                     
test_features.py                          test_config_id_generation   [ PASS ] 0.00s                                                     
test_features.py                             test_config_attributes   [ PASS ] 0.00s                                                     
test_features.py                              test_clean_text_basic   [ PASS ] 0.00s                                                     
test_features.py                          test_clean_text_stopwords   [ PASS ] 2.39s                                                     
test_features.py                      test_clean_text_lemmatization   [ PASS ] 0.00s                                                     
test_features.py                       test_clean_text_handles_none   [ PASS ] 0.00s                                                     
test_features.py                      test_extract_numeric_features   [ PASS ] 0.00s                                                     
   test_model.py       test_model_initialization[randomForestTfIdf]   [ PASS ] 0.00s                                                     
   test_model.py               test_model_initialization[codeBerta]   [ PASS ] 0.00s                                                     
   test_model.py                test_model_setup[randomForestTfIdf]   [ PASS ] 0.00s                                                     
   test_model.py                        test_model_setup[codeBerta]   [ PASS ] 1.39s                                                     
   test_model.py                test_model_train[randomForestTfIdf]   [ PASS ] 3.06s                                                     
   test_model.py                        test_model_train[codeBerta]   [ PASS ] 4.90s                                                     
   test_model.py             test_model_evaluate[randomForestTfIdf]   [ PASS ] 1.39s                                                     
   test_model.py                     test_model_evaluate[codeBerta] [ FAILED ] 6.36s  turing\tests\unit\test_model.py:101: AssertionError
   test_model.py              test_model_predict[randomForestTfIdf]   [ PASS ] 1.36s                                                     
   test_model.py                      test_model_predict[codeBerta]   [ PASS ] 5.26s                                                     
```
