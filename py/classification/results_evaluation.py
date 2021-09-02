# -*- coding: utf-8 -*-
"""
    This module provides the functions to evaluate the results of all the neural network runs with hyperparameters
    explored through random search.
"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import py.classification.neural_net as nn
import py.utilities.dataset as dataset_utils
import py.utilities.miscellaneous as misc_utils
import py.classification.class_utils as class_utils


def evaluate(top_n: int = 10) -> None:
    """
        This function allows to evaluate all the results, returning the global best top_n hyperaparemeter configurations
        in terms of sum of the f-score associated with the classes.

        Args:
            top_n: An integer representing the amount of best hyperparameter settings to display.
    """
    paths = []
    total_fscores = []
    prefix = misc_utils.get_relative_path()
    results_path = prefix + 'py/classification/results'
    dataset_path = results_path + '/' + 'AMR-UTI'
    features_considered_path = misc_utils.get_list_of_paths(dataset_path)
    for feature_dir in features_considered_path:
        layers_directories = misc_utils.get_list_of_paths(feature_dir)
        for layers_dir in layers_directories:
            neurons_directories = misc_utils.get_list_of_paths(layers_dir)
            for neurons_dir in neurons_directories:
                epoch_batch_directories = misc_utils.get_list_of_paths(neurons_dir)
                for epoch_batch_dir in epoch_batch_directories:
                    hyperparameters_directories = misc_utils.get_list_of_paths(epoch_batch_dir)
                    for hyperparameters_dir in hyperparameters_directories:
                        try:
                            df_results = pd.read_csv(hyperparameters_dir+'/final_metrics.csv')
                        except:
                            continue
                        f1_score_NIT = df_results['f1-score'].iloc[0]
                        f1_score_SXT = df_results['f1-score'].iloc[1]
                        total_fscores.append(f1_score_NIT + f1_score_SXT)
                        paths.append(hyperparameters_dir)

    top_n_scores_indexes = sorted(range(len(total_fscores)), key=lambda i: total_fscores[i], reverse=True)[:top_n]
    top_n_paths = np.array(paths)[top_n_scores_indexes]
    top_n_scores = np.array(total_fscores)[top_n_scores_indexes]
    print(top_n_paths)
    print(top_n_scores)


def evaluate_categories(top_n: int = 20, n_late_results: int = 100, max_variance: float = 0.05) -> None:
    """
        This function allows to evaluate all the results, returning the best top_n hyperaparemeter configurations
        in terms of sum of the f-score associated with the classes, distinguishing between different "categories",
        which we intend to be the number of \kappa features selected from the original dataset through the chi-square
        independence test.

        Args:
            top_n: An integer representing the amount of best hyperparameter settings to display.

            n_late_results: An integer representing the number of ending epochs to test whether they do not exhibit
                oscillations with variance above max_variance. E.g., with n_late_results equal to 100, we "test" the
                presence of strong oscillations in the last 100 epochs of training.

            max_variance: A float representing the max tolerated variance in the n_late_results last epochs.
    """
    categories = [[20, 30], [31, 50], [51, 70]]
    paths = [[] for _ in range(len(categories))]
    total_fscores = [[] for _ in range(len(categories))]
    prefix = misc_utils.get_relative_path()
    results_path = prefix + 'py/classification/results'
    dataset_path = results_path + '/' + 'AMR-UTI'
    features_considered_path = misc_utils.get_list_of_paths(dataset_path)
    for feature_dir in features_considered_path:
        n_features = int(feature_dir.split('/')[-1].split('_')[0])
        category = None
        for i, category_range in enumerate(categories):
            if n_features >= category_range[0] and n_features <= category_range[1]:
                category = i
                break
        layers_directories = misc_utils.get_list_of_paths(feature_dir)
        for layers_dir in layers_directories:
            neurons_directories = misc_utils.get_list_of_paths(layers_dir)
            for neurons_dir in neurons_directories:
                epoch_batch_directories = misc_utils.get_list_of_paths(neurons_dir)
                for epoch_batch_dir in epoch_batch_directories:
                    hyperparameters_directories = misc_utils.get_list_of_paths(epoch_batch_dir)
                    for hyperparameters_dir in hyperparameters_directories:
                        # here I control whether the results present too high oscillations between the late epochs
                        single_epoch_results_dir = hyperparameters_dir + '/single_epoch_results'
                        single_epoch_directories = misc_utils.get_list_of_paths(single_epoch_results_dir)[-n_late_results:]
                        late_results = np.zeros((2, n_late_results))
                        for i, single_epoch_dir in enumerate(single_epoch_directories):
                            epoch_str = single_epoch_dir.split('/')[-1]
                            metrics_file = single_epoch_dir + '/' + epoch_str + '_metrics.csv'
                            df_results = pd.read_csv(metrics_file)
                            f1_score_NIT = df_results['f1-score'].iloc[0]
                            f1_score_SXT = df_results['f1-score'].iloc[1]
                            late_results[0][i] = f1_score_NIT
                            late_results[1][i] = f1_score_SXT
                        variances = late_results.var(axis=1)
                        if (variances > max_variance).any():
                            continue
                        try:
                            df_results = pd.read_csv(hyperparameters_dir+'/final_metrics.csv')
                        except:
                            continue
                        f1_score_NIT = df_results['f1-score'].iloc[0]
                        f1_score_SXT = df_results['f1-score'].iloc[1]
                        total_fscores[category].append(f1_score_NIT + f1_score_SXT)
                        paths[category].append(hyperparameters_dir)

    for i in range(len(categories)):
        print(f'*** {categories[i]} ***')
        top_n_scores_indexes = sorted(range(len(total_fscores[i])), key=lambda k: total_fscores[i][k], reverse=True)[:top_n]
        top_n_paths = np.array(paths[i])[top_n_scores_indexes]
        for j in range(len(top_n_paths)):
            print(f'{j+1}) {top_n_paths[j]}')


def plot_selected_results_for_paper() -> None:
    """
        Table III of the paper shows the most proficient hyperparameters configurations for the empiric antibiotic
        prediction problem. In this function we are going to build better plots for pubblication.
    """

    prefix = misc_utils.get_relative_path()
    selected_results_path = prefix + 'py/classification/selected_results/AMR-UTI'
    run1_path = selected_results_path + '/29_features_considered/2_layers/145_2/500_epochs_306_batch_size/' \
                                        '6.70e-04_1.71e-04_4.38e-01/'
    run2_path = selected_results_path + '/42_features_considered/2_layers/177_2/500_epochs_350_batch_size/' \
                                        '5.40e-04_1.36e-04_7.49e-03/'
    run3_path = selected_results_path + '/59_features_considered/2_layers/215_2/500_epochs_601_batch_size/' \
                                        '5.18e-04_3.25e-05_7.65e-01/'
    runs = [run1_path, run2_path, run3_path]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    handles, labels = None, None
    for i, run in enumerate(runs):
        psi_str = r'$\psi=$'
        run_all_results_list = []
        single_epoch_results = misc_utils.get_list_of_paths(run+'single_epoch_results/')
        for single_epoch_dir in single_epoch_results:
            epoch = single_epoch_dir.split('/')[-1].split('_')[0]
            fscore_path = run + 'single_epoch_results/' + epoch + '_epoch/' + epoch + '_epoch_metrics.csv'
            df_fscore = pd.read_csv(fscore_path)
            f1_score_NIT = df_fscore['f1-score'].iloc[0]
            f1_score_SXT = df_fscore['f1-score'].iloc[1]
            attack1_path = run + 'attacks/single_epoch_results/' + epoch + '_epoch/' + epoch + '_epoch_predictions' \
                                                                                               '1_psi__attack.csv'

            df_attack1 = pd.read_csv(attack1_path)
            try:
                num = df_attack1[df_attack1['Ground_truth'] == 'predicted_SXT']['perturbed_NIT'].iloc[0]
                den = df_attack1[df_attack1['Ground_truth'] == 'All']['perturbed_NIT'].iloc[0]
                error1_NIT = num / den if den else 0
            except:
                error1_NIT = 0
            try:
                num = df_attack1[df_attack1['Ground_truth'] == 'predicted_NIT']['perturbed_SXT'].iloc[0]
                den = df_attack1[df_attack1['Ground_truth'] == 'All']['perturbed_SXT'].iloc[0]
                error1_SXT = num / den if den else 0
            except:
                error1_SXT = 0

            attack2_path = run + 'attacks/single_epoch_results/' + epoch + '_epoch/' + epoch + '_epoch_predictions' \
                                                                                               '2_psi__attack.csv'
            df_attack2 = pd.read_csv(attack2_path)
            try:
                num = df_attack2[df_attack2['Ground_truth'] == 'predicted_SXT']['perturbed_NIT'].iloc[0]
                den = df_attack2[df_attack2['Ground_truth'] == 'All']['perturbed_NIT'].iloc[0]
                error2_NIT = num / den if den else 0
            except:
                error2_NIT = 0
            try:
                num = df_attack2[df_attack2['Ground_truth'] == 'predicted_NIT']['perturbed_SXT'].iloc[0]
                den = df_attack2[df_attack2['Ground_truth'] == 'All']['perturbed_SXT'].iloc[0]
                error2_SXT = num / den if den else 0
            except:
                error2_SXT = 0

            run_all_results_list.append({'epoch': int(epoch), 'f-score NIT': f1_score_NIT, 'f-score SXT': f1_score_SXT,
                                         'error NIT (' + psi_str + '1)': error1_NIT,
                                         'error SXT (' + psi_str + '1)': error1_SXT,
                                         'error NIT (' + psi_str + '2)': error2_NIT,
                                         'error SXT (' + psi_str + '2)': error2_SXT})

        run_all_results = pd.DataFrame(run_all_results_list).sort_values('epoch')

        plain_results = run_all_results.melt('epoch', var_name='cols', value_name='%')
        sns.lineplot(data=plain_results, x='epoch', y='%', hue='cols', style='cols', ax=axes[i])

        if i == 0:
            handles, labels = axes[i].get_legend_handles_labels()
            axes[i].set_ylabel('%', fontsize=20)
            axes[i].yaxis.set_tick_params(width=2)
            for tick in axes[i].yaxis.get_major_ticks():
                tick.label.set_fontsize(14)
        else:
            axes[i].set_ylabel('')
        if i == 1:
            axes[i].set_xlabel('epoch', fontsize=20)
        else:
            axes[i].xaxis.set_label_text("")
        if i > 0:
            axes[i].set_yticklabels([])
        axes[i].xaxis.set_tick_params(width=2)
        axes[i].get_legend().remove()
        axes[i].set(ylim=(0, 1))
        axes[i].grid(axis='y', linewidth=0.1)
        axes[i].grid(axis='x', linestyle='--', linewidth=0.1)
        axes[i].text(0.44, 1.06, f'Run {i+1}', fontsize=18, transform=axes[i].transAxes, verticalalignment='top')
        for tick in axes[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        for axis in ['top', 'bottom', 'left', 'right']:
            axes[i].spines[axis].set_linewidth(0.1)
            axes[i].spines[axis].set_color("gray")
            axes[i].spines[axis].set_zorder(0)

    fig.tight_layout()
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 0.173), fancybox=True, fontsize=12, handles=handles, labels=labels, ncol=3)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(selected_results_path + '/final_results.pdf')
    plt.close(fig)
    plt.close()


def plot_perturbations_evaluations(values_of_psi: List[int] = None) -> None:
    """
        This function plots histograms representing both the number of times that a given feature has been perturbed,
        and the number of times that such perturbations brought to a successful elusion attack against the model.

        Args:
            values_of_psi: A list of integers which represent the amount of maximum features that the adversary
                can alter during the attack.
    """
    if values_of_psi is None:
        values_of_psi = [1, 2, 3, 4, 5]
    prefix = misc_utils.get_relative_path()
    selected_results_path = prefix + 'py/classification/selected_results/AMR-UTI'
    run1_path = selected_results_path + '/29_features_considered/2_layers/145_2/500_epochs_306_batch_size/' \
                                        '6.70e-04_1.71e-04_4.38e-01/'
    run2_path = selected_results_path + '/42_features_considered/2_layers/177_2/500_epochs_350_batch_size/' \
                                        '5.40e-04_1.36e-04_7.49e-03/'
    run3_path = selected_results_path + '/59_features_considered/2_layers/215_2/500_epochs_601_batch_size/' \
                                        '5.18e-04_3.25e-05_7.65e-01/'
    runs = [run1_path, run2_path, run3_path]
    list_of_df_perturbations = []
    list_of_runs_errors = []
    for i, run in enumerate(runs):

        model_path = run + 'final_model.pkl'

        network = nn.NeuralNetwork.load_parameters(model_path)

        network.model_path = run
        network.train_path = run

        dataset = dataset_utils.Amr_Uti_Dataset(n_features=network.features_in_dataset)
        network.perturbation_mask = dataset.perturbation_mask
        y_hat = network.predict(dataset.test_X)

        results, history_of_altered_features, corrupted_X = \
            network.aiXia2021_attack(dataset.test_X, dataset.test_Y_one_hot, y_hat, 500, values_of_psi=values_of_psi)

        predicted_classes_one_hot = class_utils.from_probabilities_to_labels(y_hat)

        correctly_classified_mask = (dataset.test_Y_one_hot == predicted_classes_one_hot)[:, 0]

        ground_truth = class_utils.one_hot(dataset.test_Y_one_hot[correctly_classified_mask], encode_decode=1)

        features_names = dataset.features_names

        run_results = []

        for psi in values_of_psi:
            corrupted_Y = network.predict(corrupted_X[psi-1])
            corrupted_Y_one_hot = class_utils.from_probabilities_to_labels(corrupted_Y)
            corrupted_Y_classes = class_utils.one_hot(corrupted_Y_one_hot, encode_decode=1)

            for sample in range(len(ground_truth)):
                ground = ground_truth[sample]
                predicted = corrupted_Y_classes[sample]
                success = 0
                if ground != predicted:
                    success = 1
                perturbed_features = history_of_altered_features[psi-1][sample]
                perturbed_features_indices = np.argwhere(perturbed_features == 1)
                for index in perturbed_features_indices:
                    results_for_sample = {'Ground': ground, 'Predicted': predicted,
                                          'Feature Name': features_names[index[0]],
                                          'psi': psi, 'Success': success}
                    run_results.append(results_for_sample)

        df_perturbations = pd.DataFrame(run_results)
        list_of_df_perturbations.append(df_perturbations)
        list_of_runs_errors.append(results)
    merged_df_perturbations = pd.concat(list_of_df_perturbations)
    all_perturbed_features_names = merged_df_perturbations['Feature Name'].unique()
    final_dicts = []
    for psi, group in merged_df_perturbations.groupby('psi'):
        feature_name_series = group['Feature Name'].value_counts()
        n_chosen = feature_name_series.sum()
        success_series = group.groupby('Feature Name')['Success'].sum()
        for feature in all_perturbed_features_names:
            if feature not in feature_name_series.index:
                feature_dict = {'Feature Name': feature, 'Chosen': 0,
                                'Success': 0, 'psi': psi}
            else:
                chosen_count = feature_name_series.loc[feature]
                success_count = success_series.loc[feature]
                feature_dict = {'Feature Name': feature, 'Chosen': chosen_count/n_chosen,
                                'Success': success_count/n_chosen,
                                'psi': psi}
            final_dicts.append(feature_dict)
    final_df = pd.DataFrame(final_dicts)

    success_heatmap = final_df.pivot(index='psi', columns='Feature Name', values='Success')

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(1, 1, 1)
    p = sns.heatmap(success_heatmap, cmap="Blues", linewidth=0.3, linecolor='black', cbar_kws={'label': 'Success %',
                                                                                               'shrink': .09},
                    ax=ax, square=True)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    p.figure.axes[-1].yaxis.label.set_size(18)
    _, ylabels = plt.yticks()
    p.set_yticklabels(ylabels, size=16, rotation=45)
    _, xlabels = plt.xticks()
    p.set_xticklabels(range(len(xlabels)), size=16, rotation=45)
    ax.set_xlabel('Features', fontsize=18)
    psi_str = r'$\psi$'
    ax.set_ylabel(psi_str, fontsize=18)
    ax.tick_params(left=False, top=False, bottom=False)
    fig.tight_layout()
    plt.savefig(selected_results_path + '/success_results.pdf', bbox_inches='tight')
    plt.close(fig)
    plt.close()

    count_heatmap = final_df.pivot(index='psi', columns='Feature Name', values='Chosen')

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(1, 1, 1)
    p = sns.heatmap(count_heatmap, cmap="Blues", linewidth=0.3, linecolor='black', cbar_kws={'label': 'Chosen %',
                                                                                               'shrink': .09},
                    ax=ax, square=True)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    p.figure.axes[-1].yaxis.label.set_size(18)
    _, ylabels = plt.yticks()
    p.set_yticklabels(ylabels, size=16, rotation=45)
    _, xlabels = plt.xticks()
    p.set_xticklabels(['' for _ in range(len(xlabels))], size=16, rotation=45)
    ax.set_xlabel('')
    psi_str = r'$\psi$'
    ax.set_ylabel(psi_str, fontsize=18)
    ax.tick_params(left=False, top=False, bottom=False)
    fig.tight_layout()
    plt.savefig(selected_results_path + '/chosen_results.pdf', bbox_inches='tight')
    plt.close(fig)
    plt.close()


if __name__ == '__main__':
    #plot_selected_results_for_paper()
    plot_perturbations_evaluations()
