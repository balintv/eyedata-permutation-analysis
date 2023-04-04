# eyedata-permutation-analysis

This script implements a nonparametric statistical test for analyzing eye-tracking data based on the method described by Maris and Oostenveld (2007). The core of this script is based on a MATLAB script written by Kinga Anna Bohus. I implemented it in Python with extra preprocessing and plotting functions. It can be used to compare two groups and determine if there are any significant differences between them. This can be done either with paired or unpaired data and using parametric (t-test) or non-parametric (Wilcoxon) tests.

The script defines a `PermutationTest` class with various methods for data preprocessing, filtering, interpolation, and statistical testing. Given a set of input parameters, it loads the data, preprocesses it, and runs the permutation tests based on the input parameters.

- `eyedata_reformat()` method: reads the input data, extracts relevant information, and groups it by subject and trial.
- `hampel_filter()` method: used for outlier detection and correction using the Hampel filter.
- `filter_and_interpolate()` method applies the Hampel filter and/or linear interpolation to the data if specified.
- `get_clusters()` method: calculates clusters of significant differences between the two groups using the specified test (parametric or non-parametric) and paired or unpaired data.
- `permute_once()` method: creates one permutation of the data, creating new groups for comparison.
- `run_permutation_test()` method: carries out the permutation test by generating multiple permutations, calculating clusters for each permutation, and comparing the original clusters with the permuted data.

Check the Jupyter notebook for details and use.
