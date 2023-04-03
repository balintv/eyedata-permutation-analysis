# eyedata-permutation-analysis

This script implements a nonparametric statistical test for analyzing eye-tracking data based on the method described by Maris and Oostenveld (2007). The main steps of this method:

    Combine the trials from two experimental conditions into a single set.
    Randomly draw a partition of trials and calculate the test statistic.
    Repeat the previous step multiple times to construct a histogram of test statistics.
    Calculate the p-value based on the observed test statistic and the histogram.
    Compare the p-value to a critical alpha-level (typically, 0.05) to determine if the data in the two experimental conditions are significantly different.

The method is adapted for time-series data by incorporating the cluster mass test, which includes these steps:

    Calculate a t-value for each sample comparing the eye-tracking data between the two experimental conditions.
    Select all samples with t-values above a certain threshold.
    Cluster the selected samples into connected sets based on temporal adjacency.
    Calculate cluster-level statistics by taking the sum of the t-values within a cluster.
    Take the largest of the cluster-level statistics.
