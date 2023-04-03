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

The core of the script is based on a MATLAB script written by Kinga Anna Bohus.
