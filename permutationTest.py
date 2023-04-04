import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from scipy.stats import t, shapiro, ttest_rel, ttest_ind, wilcoxon, ranksums, norm
from scipy.interpolate import interp1d
from statsmodels.stats.multitest import multipletests

class PermutationTest:
    """
    A class to perform permutation tests for comparing two groups of eye-tracking data.
    
    Attributes:
        exp_name (str): The name of the experiment.
        datafile_0 (str): The path to the data file for group 0.
        datafile_1 (str): The path to the data file for group 1.
        label_0 (str): The label for group 0.
        label_1 (str): The label for group 1.
        eye_variable (str): The eye-tracking variable to analyze.
        freq (float): The sampling frequency of the eye-tracker.
        sample_range (list): A list of two integers specifying the start and end points of the sample range you want to read.
        filtering (bool): Whether to apply Hampel filtering to the data.
        sliding_win_size (int): The sliding window size for the Hampel filter.
        outlier_crit (float): The outlier criterion (number of standard deviations) for the Hampel filter.
        linear_interpolation (bool): Whether to apply linear interpolation to the data.
        max_gap (int): The maximum gap size for linear interpolation.
        permutations (int): The number of permutations for the permutation test.
        analysis_range (list): A list of two integers specifying the start and end points of the analysis range.
        t_test (bool): Whether to use the t-test as the test statistic.
        wilcoxon_test (bool): Whether to use the Wilcoxon test as the test statistic.
        tails (int): The number of tails for the test (1 or 2).
        p_threshold (float): The p-value threshold for significance.
        t_threshold (float): The t-value threshold for significance.
    
    Methods:
        eyedata_reformat(filename, ids=None)
            Load and reformat eye-tracking data from the given file.
        
        hampel_filter(data, win_size, devs)
            Apply the Hampel filter to smooth the data and remove outliers.
        
        filter_and_interpolate()
            Apply filtering and/or linear interpolation to the data.
        
        get_clusters(means_0, means_1, paired, parametric)
            Identify clusters and calculate their masses in the data.
        
        permute_once(means_0, means_1, paired, grouped_0, grouped_1)
            Perform a single permutation of the data.
        
        run_permutation_test(paired, parametric)
            Perform the permutation test and calculate statistics.
    """
    def __init__(
        self, exp_name, datafile_0, datafile_1, label_0, label_1, eye_variable, freq,
        sample_range, filtering, sliding_win_size, outlier_crit, linear_interpolation, max_gap,
        permutations, analysis_range, t_test, wilcoxon_test, tails, p_threshold, t_threshold
    ):
        self.exp_name = exp_name
        self.datafile_0 = datafile_0
        self.datafile_1 = datafile_1
        self.label_0 = label_0
        self.label_1 = label_1
        self.eye_variable = eye_variable
        self.freq = freq

        self.filtering = filtering
        self.sliding_win_size = sliding_win_size
        self.outlier_crit = outlier_crit
        self.linear_interpolation = linear_interpolation
        self.max_gap = max_gap
        self.permutations = permutations

        self.t_test = t_test
        self.wilcoxon_test = wilcoxon_test
        self.tails = tails
        self.p_threshold = p_threshold
        self.t_threshold = t_threshold

        self.analysis_range = np.arange(analysis_range[0], analysis_range[1])
        self.sample_range = np.arange(sample_range[0], sample_range[1])

        # Since for some data, (-) values might be given in sample and analysis range, do this conversion        
        self.an_start = abs(min(self.sample_range) - min(self.analysis_range))
        self.an_end = self.an_start + max(self.analysis_range)

        # Reformat data
        self.group_0 = self.eyedata_reformat(self.datafile_0)
        self.group_1 = self.eyedata_reformat(self.datafile_1)
        print('Data loaded.')

        # Filter, interpolate, if needed
        if self.filtering or self.linear_interpolation:
            self.filter_and_interpolate()
            print('Filtering and/or linear interpolation done.')

        # # Run the permutation test(s)
        if self.t_test:
            self.t_test_results = self.run_permutation_test(paired=True, parametric=True)
            print("Permutation analysis with t-tests done.")
        if self.wilcoxon_test:
            self.wilcoxon_test_results = self.run_permutation_test(paired=True, parametric=False)
            print("Permutation analysis with Wilcoxons done.")

    def eyedata_reformat(self, filename, ids=None):
        """
        Load and reformat eye-tracking data from the given file.
        
        Args:
            filename (str): Path to the file containing eye-tracking data.
            ids (list, optional): List of specific subject IDs to include. If None, include all subjects. Defaults to None.
        
        Returns:
            list: A list of tuples containing the reformatted dataframs and subject IDs corresponding to each subject.
        """
        condition = pd.read_csv(filename, delimiter=',', header=0)

        condition['id'] = condition['id'].astype('category')
        condition['trial'] = condition['trial'].astype('category')

        if ids is not None:
            subjects = sorted(set(ids).intersection(condition['id'].unique()))
        else:
            subjects = sorted(condition['id'].unique())

        grouped_data = []

        # Loop through subjects
        for subject in subjects:
            subject_table = condition[condition['id'] == subject] # Work only with current subject's data

            trials = sorted(subject_table['trial'].unique()) # % Extract unique trial types within subject
            m = pd.DataFrame(index=trials, columns=range(min(self.sample_range), max(self.sample_range) + 1)) # Preallocate timepoint*l matrix for the subject

            # Loop through trial types
            for trial in trials:
                trial_table = subject_table[subject_table['trial'] == trial] # Work only with current trial's data
                idx = (trial_table['time'] >= min(self.sample_range)) & (trial_table['time'] <= max(self.sample_range))
                filtered_trial_table = trial_table.loc[idx, self.eye_variable]

                # Preallocate a row with zeros, then assign the filtered eyedata values
                row = np.zeros(len(self.sample_range))
                row[:len(filtered_trial_table)] = filtered_trial_table.values
                m.loc[trial] = row

            grouped_data.append((m, subject))

        return grouped_data

    def hampel_filter(self, data, win_size, devs):
        """
        Apply the Hampel filter to smooth the data and remove outliers.
        
        Args:
            data (array): The input data.
            win_size (int): The window size for the Hampel filter.
            devs (int): The number of standard deviations from the median for outlier detection.
        
        Returns:
            array: The filtered data.
        """

        filtered = np.array(data)
        size = int(win_size / 2)

        for i in range(size, len(data) - size - 1):
            window = filtered[i - (size - 1) : i + size]
            median = np.median(window)
            madev = np.median(np.abs(window - median))
            if (filtered[i] > median + (devs * madev)) or (filtered[i] < median - (devs * madev)):
                filtered[i] = median
        
        return filtered

    def filter_and_interpolate(self):
        """
        Apply filtering and/or linear interpolation to the data.
        """
        timepoints = np.arange(min(self.sample_range), max(self.sample_range) + 1)

        def preprocess_group(group):
            for s in range(len(group)):
                if group[s] is not None:
                    n_trials = group[s][0].shape[0]
                    for t in range(n_trials):
                        if self.filtering:
                            unfiltered_values = group[s][0].iloc[t].to_numpy(copy=True)
                            filtered_values = self.hampel_filter(unfiltered_values, self.sliding_win_size, self.outlier_crit)
                            group[s][0].iloc[t] = pd.Series(filtered_values, index=group[s][0].columns)
                        if self.linear_interpolation:
                            interpolated_values = interp1d(timepoints, group[s][0].iloc[t].to_numpy(), bounds_error=False, fill_value=np.nan)
                            group[s][0].iloc[t] = pd.Series(filtered_values, index=group[s][0].columns)

        preprocess_group(self.group_0)
        preprocess_group(self.group_1)

    def get_clusters(self, means_0, means_1, paired, parametric):
        """
        Identify clusters and calculate their masses in the data.
        
        Args:
            means_0 (array): Mean values of the first group.
            means_1 (array): Mean values of the second group.
            paired (bool): Whether the permutation test is paired or not.
            parametric (bool): Whether the test statistic is parametric or not.
        
        Returns:
            tuple: A tuple containing clusters, masses, t-values per timepoint, and p-values per timepoint.
        """
        l = means_0.shape[1]
        p_per_timepoint = np.ones(l)
        t_per_timepoint = np.zeros(l)

        for timepoint in range(self.an_start, self.an_end + 1):
            if np.sum(~np.isnan(means_1[:, timepoint])) > 0 and np.sum(~np.isnan(means_0[:, timepoint])) > 0:
                if parametric:
                    if paired:
                        if self.tails == 1:
                            t_stat, p = ttest_rel(means_0[:, timepoint], means_1[:, timepoint], nan_policy='omit', alternative='less')
                        else:
                            t_stat, p = ttest_rel(means_0[:, timepoint], means_1[:, timepoint], nan_policy='omit')
                    else:
                        if self.tails == 1:
                            t_stat, p = ttest_ind(means_0[:, timepoint], means_1[:, timepoint], nan_policy='omit', alternative='less')
                        else:
                            t_stat, p = ttest_ind(means_0[:, timepoint], means_1[:, timepoint], nan_policy='omit')
                else:
                    if paired:
                        if self.tails == 1:
                            t_stat, p = wilcoxon(means_0[:, timepoint], means_1[:, timepoint], nan_policy='omit', alternative='less', mode='approx')
                        else:
                            t_stat, p = wilcoxon(means_0[:, timepoint], means_1[:, timepoint], nan_policy='omit', mode='exact')
                    else:
                        if self.tails == 1:
                            t_stat, p = ranksums(means_0[:, timepoint], means_1[:, timepoint], nan_policy='omit', alternative='less')
                        else:
                            t_stat, p = ranksums(means_0[:, timepoint], means_1[:, timepoint], nan_policy='omit')
                p_per_timepoint[timepoint] = p
                t_per_timepoint[timepoint] = t_stat
            else:
                t_per_timepoint[timepoint] = np.nan
                p_per_timepoint[timepoint] = np.nan

        if self.p_threshold == 0:
            significant_timepoints = (np.abs(t_per_timepoint) > self.t_threshold) | np.isnan(t_per_timepoint)
        else:
            significant_timepoints = (p_per_timepoint <= self.p_threshold) | np.isnan(p_per_timepoint)

        clusters = []
        cluster = []
        timepoint = 0
        while timepoint < l:
            if significant_timepoints[timepoint]:
                cluster.append(timepoint)
            else:
                if len(cluster) > 1:
                    clusters.append((cluster[0], cluster[-1]))
                cluster = []
            timepoint += 1
        if len(cluster) > 1:
            clusters.append((cluster[0], cluster[-1]))

        masses = []
        for cluster_start, cluster_end in clusters:
            mass = np.nansum(t_per_timepoint[cluster_start:cluster_end+1])
            if self.tails == 1:
                masses.append(mass)
            else:
                masses.append(np.abs(mass))

        return clusters, masses, t_per_timepoint, p_per_timepoint

    def permute_once(self, means_0, means_1, paired, grouped_0, grouped_1):
        """
        Perform a single permutation of the data.
        
        Args:
            means_0 (array): Mean values of the first group.
            means_1 (array): Mean values of the second group.
            paired (bool): Whether the permutation test is paired or not.
            grouped_0 (list): List of tuples containing the data and subject ID for the first group.
            grouped_1 (list): List of tuples containing the data and subject ID for the second group.
        
        Returns:
            tuple: A tuple containing the new mean values for both groups after permutation.
        """

        # Preallocate matrices for the new partitions
        new_mean_0 = np.empty_like(means_0)
        new_mean_0[:] = np.nan
        new_mean_1 = np.empty_like(means_1)
        new_mean_1[:] = np.nan

        if paired:
            n = len(grouped_0)
            for i in range(n):
                n_trials_0 = grouped_0[i][0].shape[0] # Number of trials for the subject in condition 0
                n_trials_1 = grouped_1[i][0].shape[0] # Number of trials for the subject in condition 1
                trials = np.vstack((grouped_0[i][0], grouped_1[i][0])) # All datapoints corresponding to the subject in a trialn*sample matrix
                new_order = np.random.permutation(trials.shape[0]) # Random permutations of integers from 1 to n = cumulative number of trials

                if n_trials_0 > 0:
                    selected_trials_0 = trials[new_order[:n_trials_0]].astype(float)
                    all_nan_columns_0 = np.all(np.isnan(selected_trials_0), axis=0)
                    
                    # Calculate the mean for columns with valid data points
                    new_mean_0[i, ~all_nan_columns_0] = np.nanmean(selected_trials_0[:, ~all_nan_columns_0], axis=0)
                    
                    # Fill all-NaN columns with NaN values
                    new_mean_0[i, all_nan_columns_0] = np.nan

                if n_trials_1 > 0:
                    selected_trials_1 = trials[new_order[n_trials_0:(n_trials_0 + n_trials_1)]].astype(float)
                    all_nan_columns_1 = np.all(np.isnan(selected_trials_1), axis=0)
                    
                    # Calculate the mean for columns with valid data points
                    new_mean_1[i, ~all_nan_columns_1] = np.nanmean(selected_trials_1[:, ~all_nan_columns_1], axis=0)
                    
                    # Fill all-NaN columns with NaN values
                    new_mean_1[i, all_nan_columns_1] = np.nan

        else:
            # In case of non-paired test: permute the subjects between onditions,
            # note that if two trials belonged to the same subject they remain with the same subject
            n_0 = len(grouped_0) # Sample size in condition 0
            n_1 = len(grouped_1) # Sample size in condition 1
            trials = grouped_0 + grouped_1 # All matrices corresponding to all subjects in a single set
            new_trials = [trials[i] for i in np.random.permutation(len(trials))] # Mix subjects between conditions using a random permutation of integers from 1 to n = cumulative number of subjects

            # Fill the preallocated matrix with the means of as many subjects
            # from the mixed-up data set as there were subject in condition 0
            for i in range(n_0):
                new_mean_0[i, :] = np.nanmean(new_trials[i][0], axis=0)

            # Fill the preallocated matrix with the means of the rest of the subjects
            for j in range(n_1):
                new_mean_1[j, :] = np.nanmean(new_trials[n_0 + j][0], axis=0)

        return new_mean_0, new_mean_1


    def run_permutation_test(self, paired, parametric):
        """
        Perform the permutation test and calculate statistics.
        
        Args:
            paired (bool): Whether the permutation test is paired or not.
            parametric (bool): Whether the test statistic is parametric or not.

        Returns:
            dict: A dictionary containing the test results and statistics.
        """

        # Check number of subjects in both conditions as well as the length of the whole dataset
        n_subj_0 = len(self.group_0)
        n_subj_1 = len(self.group_1)
        l = self.group_0[0][0].shape[1]

        # Preallocate matrices for within-subject means for condition A and B: nsubj*windowlength
        means_0 = np.empty((n_subj_0, l))
        means_0[:] = np.nan
        means_1 = np.empty((n_subj_1, l))
        means_1[:] = np.nan

        # Fill matrices with means of timepoints across all trials for each subject in both conditions
        for i in range(n_subj_0):
            if not self.group_0[i][0].empty:
                means_0[i, :] = self.group_0[i][0].mean(axis=0)
        for i in range(n_subj_1):
            if not self.group_1[i][0].empty:
                means_1[i, :] = self.group_1[i][0].mean(axis=0)

        # Calculate original clusters
        clusters, masses, t_per_timepoint, p_per_timepoint = self.get_clusters(means_0, means_1, paired, parametric)

        # Iteratively generate permutations, carry out clustering and extract largest test statistics for each permutation
        mass_dist = np.zeros(self.permutations)
        for new_perm in range(self.permutations):
            new_0, new_1 = self.permute_once(means_0, means_1, paired, self.group_0, self.group_1)
            _, masses_n, _, _ = self.get_clusters(new_0, new_1, paired, parametric)

            if len(masses_n) > 0:
                mass_dist[new_perm] = np.max(masses_n)
            else:
                mass_dist[new_perm] = 0

        # For each original cluster, calculate the proportion of permutations that resulted in a larger test statistic than the observed one
        probs = np.zeros(len(masses))
        for i in range(len(masses)):
            probs[i] = len(np.where(mass_dist > masses[i])[0]) / self.permutations

        return {
            "clusters": clusters,
            "probs": probs,
            "masses": masses,
            "means_0": means_0,
            "means_1": means_1,
            "t_per_timepoint": t_per_timepoint,
            "p_per_timepoint": p_per_timepoint
        }

    def plot_results(self, results, y_label, error_bars, baseline_offset):
        """
        Plots the eye-tracking data analysis results, including mean values and error bars for both conditions,
        test statistics, and optionally a normality plot.

        Args:
            results (dict): Dictionary containing the analysis results, including mean values, clusters, and probabilities.
            y_label (str): String to use as the y label on the main plot.
            error_bars (str): Type of error bars to display; either "SE" for standard error or "CI" for confidence interval.
            baseline_offset (float): End time of the baseline period (in seconds).

        Returns:
            None: Displays the plot, does not return any values.
        """
        means_0 = results['means_0']
        means_1 = results['means_1']
        clusters = results['clusters']
        cluster_probs = results['probs']

        def se_ci95(means):
            l = means.shape[1]
            se = np.empty(l) * np.nan
            ci95 = np.empty(l) * np.nan

            for i in range(l):
                if not np.isnan(means[:, i]).all():
                    s_len = len(means[:, i]) - np.isnan(means[:, i]).sum()

                    sem = np.nanstd(means[:, i]) / np.sqrt(s_len)
                    se[i] = sem

                    ci95_interval = t.ppf([0.025, 0.975], s_len - 1)
                    y_CI95 = sem * ci95_interval
                    ci95[i] = y_CI95[1]

                    _, pValue = shapiro(means[:, i][~np.isnan(means[:, i])])

            return se, ci95

        # Extract means for plots
        sample_means_0 = np.nanmean(means_0, axis=0)
        sample_means_1 = np.nanmean(means_1, axis=0)

        # Compute SEs and CI95 for both groups/conditions
        se_0, ci95_0 = se_ci95(means_0)
        se_1, ci95_1 = se_ci95(means_1)

        if error_bars == "SE":
            error_bar_0 = se_0
            error_bar_1 = se_1
        elif error_bars == "CI":
            error_bar_0 = ci95_0
            error_bar_1 = ci95_1

        # Set ranges and limits
        x_range = np.arange(self.an_start, self.an_end + 1)
        xrange_secs = np.arange(np.min(self.analysis_range), np.max(self.analysis_range) + 1) / self.freq
        xrange_secs_plot = np.arange(np.min(self.sample_range), np.max(self.sample_range) + 1) / self.freq
        xlim_secs_plot = [np.min(self.sample_range) / self.freq, np.max(self.sample_range) / self.freq]
        time_ticks_plot = np.arange(np.min(self.sample_range) / self.freq, np.max(self.sample_range) / self.freq, 0.4)
        diff = abs(np.min(self.sample_range) - np.min(self.analysis_range))
        idxs = xrange_secs_plot >= baseline_offset
        idxs_baseline = xrange_secs_plot <= baseline_offset

        # Colors
        color_0 = 'dodgerblue'
        color_1 = 'orangered'
        color_0_baseline = 'powderblue'
        color_1_baseline = 'peachpuff'

        # Plotting
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        # Subplot for main time-series data with shaded error bars
        def plot_main(ax, sample_means, error_bar, color, color_baseline, idxs, idxs_baseline):
            ax.fill_between(xrange_secs_plot[idxs], sample_means[idxs] - error_bar[idxs], sample_means[idxs] + error_bar[idxs], color=color, alpha=0.15)
            ax.fill_between(xrange_secs_plot[idxs_baseline], sample_means[idxs_baseline] - error_bar[idxs_baseline], sample_means[idxs_baseline] + error_bar[idxs_baseline], color=color_baseline, alpha=0.4)
            main_line, = ax.plot(xrange_secs_plot[idxs], sample_means[idxs], color=color)  # Save the line object
            ax.plot(xrange_secs_plot[idxs_baseline], sample_means[idxs_baseline], color=color, alpha=0.4)
            return main_line

        ax_main = axs[0]
        line_0 = plot_main(ax_main, sample_means_0, error_bar_0, color_0, color_0_baseline, idxs, idxs_baseline)
        line_1 = plot_main(ax_main, sample_means_1, error_bar_1, color_1, color_1_baseline, idxs, idxs_baseline)

        ax_main.axvline(baseline_offset, linestyle='--', color='black', linewidth=1, alpha=0.5)
        ax_main.set_xlim(xlim_secs_plot[0], xlim_secs_plot[1])
        ax_main.set_xticks(time_ticks_plot)
        ax_main.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax_main.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax_main.set_ylabel(y_label)
        ax_main.legend([line_0, line_1], [self.label_0, self.label_1])

        # Add clusters
        if clusters:
            y_min, y_max = ax_main.get_ylim()
            y_range = y_max - y_min
            c_pos = y_min + 0.1 * y_range
            c_test_pos = c_pos + 0.05 * y_range

            for idx, cluster in enumerate(clusters):
                if cluster_probs[idx] < 0.05:
                    c_start = cluster[0] - diff
                    c_end = cluster[1] - diff
                    print(c_start, c_end)

                    ax_main.plot([c_start / self.freq, c_end / self.freq], [c_pos, c_pos], linewidth=2, color='orange', marker='|')
                    ax_main.annotate(f'p = {cluster_probs[idx]:.2f}', xy=(np.median([c_start / self.freq, c_end / self.freq]), c_test_pos), ha='center')

        # Subplot for test statistics
        ax_test = axs[1]
        ax_test.plot(xrange_secs, results['p_per_timepoint'][x_range], color='dimgrey')
        ax_test.axhline(self.p_threshold, linestyle='--', color='black', linewidth=1, alpha=0.5)
        ax_test.axvline(baseline_offset, linestyle='--', color='black', linewidth=1, alpha=0.5)
        ax_test.set_ylim(-0.1, 1.1)
        ax_test.set_xlim(xlim_secs_plot[0], xlim_secs_plot[1])
        ax_test.set_xticks(time_ticks_plot)
        ax_test.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax_test.set_ylabel('p')

        plt.tight_layout()
        plt.show()