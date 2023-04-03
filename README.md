# eyedata-permutation-analysis

This script implements a nonparametric statistical test for analyzing eye-tracking data based on the method described by Maris and Oostenveld (2007). The main steps of this method:

1. Combine the trials from two experimental conditions into a single set.
2. Randomly draw a partition of trials and calculate the test statistic.
3. Repeat the previous step multiple times to construct a histogram of test statistics.
4. Calculate the p-value based on the observed test statistic and the histogram.
5. Compare the p-value to a critical alpha-level (typically, 0.05) to determine if the data in the two experimental conditions are significantly different.

The method is adapted for time-series data by incorporating the cluster mass test, which includes these steps:

1. Calculate a t-value for each sample comparing the eye-tracking data between the two experimental conditions.
2. Select all samples with t-values above a certain threshold.
3. Cluster the selected samples into connected sets based on temporal adjacency.
4. Calculate cluster-level statistics by taking the sum of the t-values within a cluster.
5. Take the largest of the cluster-level statistics.

The core of this script is based on a MATLAB script written by Kinga Anna Bohus. Its purpose is to compare two groups and determine if there are any significant differences between them. This can be done either with paired or unpaired data and using parametric (t-test) or non-parametric (Wilcoxon) tests.

The script defines a `PermutationTest` class with various methods for data preprocessing, filtering, interpolation, and statistical testing. Given a set of input parameters, it loads the data, preprocesses it, and runs the permutation tests based on the input parameters.

- `eyedata_reformat()` method :reads the input data, extracts relevant information, and groups it by subject and trial.
- `hampel_filter()` method: used for outlier detection and correction using the Hampel filter.
- `filter_and_interpolate() method applies the Hampel filter and/or linear interpolation to the data if specified.
- `get_clusters()` method: calculates clusters of significant differences between the two groups using the specified test (parametric or non-parametric) and paired or unpaired data.
- `permute_once()` method: creates one permutation of the data, creating new groups for comparison.
- `run_permutation_test()` method: carries out the permutation test by generating multiple permutations, calculating clusters for each permutation, and comparing the original clusters with the permuted data.

Check the Jupyter notebook for details and use.
