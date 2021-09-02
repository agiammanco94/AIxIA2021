# -*- coding: utf-8 -*-
"""
    This module contains the preprocessing algorithm adopted for the AMR-UTI dataset,
    described in Section V-A of the paper, and all the other utilities functions to load the dataset.

    The AMR-UTI dataset has to be downloaded, and its main files ("all_prescriptions.csv", "all_uti_features.csv",
    "all_uti_resist_labels.csv", "data_dictionary.csv") have to be placed in the "datasets/AMR-UTI" folder.

    References:
        [1] http://kdd.ics.uci.edu/databases/kddcup99/task.html
        [2] TAVALLAEE, Mahbod, et al. A detailed analysis of the KDD CUP 99 data set. In: 2009 IEEE symposium on
        computational intelligence for security and defense applications. IEEE, 2009. p. 1-6
        [3] https://towardsdatascience.com/a-deeper-dive-into-the-nsl-kdd-data-set-15c753364657
        [4] https://stackoverflow.com/questions/3172509/numpy-convert-categorical-string-arrays-to-an-integer-array
        [5] https://stackoverflow.com/questions/42065501/how-to-select-rows-of-a-
        numpy-data-array-according-to-class-labels-in-a-separate
        [6] https://www.physionet.org/content/antimicrobial-resistance-uti/1.0.0/
        [7] Kanjilal, Sanjat, et al. "A decision algorithm to promote outpatient antimicrobial stewardship for
        uncomplicated urinary tract infection." Science Translational Medicine 12.568 (2020).
        [8] https://stackoverflow.com/questions/39658574/
        how-to-drop-columns-which-have-same-values-in-all-rows-via-pandas-or-spark-dataf
        [9] https://scikit-learn.org/stable/modules/feature_selection.html
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import py.utilities.miscellaneous as misc_utils
import py.classification.class_utils as class_utils


class Dataset:
    """
        This class models a generic Dataset, it will serve as parent class for the AMR-UTI dataset.

        Attributes:
            dataset_name: A string which represents the current dataset.

            X: The input features. It is a NumPy array of shape (n_samples_in_dataset, n_features).

            Y: The class labels associated with X.

            Y_one_hot: The one-hot encoded version of the class labels.

            classes_names: A list of strings representing the specific labels of the class, e.g., "NIT" or "SXT".

            features_names: A list of strings representing the names of the single features contained in the dataset.

            class_label: The name of the target variable in the dataset, e.g., "prescription".

            not_corruptible_features: A List of string representing the names of the features that cannot be corrupted,
                for example, the age of the patients, or their previous resistance to the antibiotic of the class label
                (NIT, SXT).

            perturbation_mask: A ndarray of shape (features_in_dataset,) containing values 0 for all the indices
                corresponding to the not_corruptible_features.
    """
    def __init__(self, dataset_name: str) -> None:
        """
            Inits a Dataset instance with default None values for the attributes.

            Args:
                dataset_name: A string representing the name of the current dataset, e.g., "AMR-UTI".
        """
        legit_dataset_types = ['AMR-UTI']
        try:
            assert dataset_name in legit_dataset_types
        except AssertionError:
            exit(f'Unsupported dataset type. Please insert one from {legit_dataset_types}')
        self.dataset_name = dataset_name
        self.X = None
        self.Y = None
        self.Y_one_hot = None
        self.classes_names = None
        self.features_names = None
        self.numerical_features = None
        self.categorical_features = None
        self.real_value_features = None
        self.binary_features = None
        self.class_label = None
        self.perturbation_mask = None
        self.not_corruptible_features = None

    def __repr__(self) -> str:
        return f'{self.dataset_name}'

    @property
    def features_in_dataset(self) -> int:
        return len(self.features_names)

    @property
    def classes_in_dataset(self) -> int:
        return len(self.classes_names)

    @property
    def numerical_features_in_dataset(self) -> int:
        return len(self.numerical_features)

    @property
    def categorical_features_in_dataset(self) -> int:
        return len(self.categorical_features)

    @property
    def perturbation_step_mask(self) -> np.ndarray:
        """
            This method returns a vector containing real values representing the smallest quantities which may be
            injected into the data during adversarial attacks.

            Returns:
                perturbations_steps: A ndarray of shape (n_features,) containing the smallest step which can be taken
                    in the direction of the gradient for each single feature in the dataset.
        """
        perturbation_steps = np.zeros(self.features_in_dataset)
        for feature_to_perturb in self.numerical_features:
            idx = self.get_index_feature(feature_to_perturb)
            # here I compute the minimum legal step which can be added
            column = self.get_feature_column(feature_to_perturb)
            unique_values = np.sort(np.unique(column))
            differences = np.diff(unique_values)
            try:
                min_perturbation = differences.min()
            except:
                min_perturbation = 0
            perturbation_steps[idx] = min_perturbation
        return perturbation_steps

    def get_index_feature(self, feature: str) -> int:
        """
            This utility method returns the index of a certain feature inside the X variable.

            Args:
                feature: A string representing the name of the feature.

            Returns:
                idx: The index of the feature inside the X matrix which represent all the input samples of the dataset.
        """
        idx = self.features_names.index(feature)
        return idx

    def get_feature_column(self, feature: str) -> np.ndarray:
        """
            This utility method returns the entire column of a given feature, that is, all the values that a specific
            feature assume over all the samples in the dataset.

            Args:
                feature: A string representing the name of the feature.

            Returns:
                column: A ndarray of shape (n_samples_in_dataset, 1) containing all the possible values of a given
                    feature in the dataset.
        """
        idx = self.get_index_feature(feature)
        column = self.X[:, idx]
        return column

    def get_column_unique_values(self, feature: str) -> dict:
        """
            This utility function returns a dicionary containing the single unique values of a specific feature in the
            dataset, together with their frequency of occurrence.

            Args:
                feature: A string representing the name of the feature.

            Returns:
                unique_values: A dictionary containing the frequency of occurrence of each single value in the dataset
                    for a specific feature.
        """
        column = self.get_feature_column(feature)
        unique, counts = np.unique(column, return_counts=True)
        unique_values = dict(zip(unique, counts))
        return unique_values

    def get_columns_unique_values(self) -> pd.DataFrame:
        """
            This utility function returns a DataFrame containing the unique values for all the features in the dataset,
            associated with their frequency of occurrence.

            Returns:
                df_unique_values: A pandas DataFrame containing all the unique values with their frequency of
                    occurrence.
        """
        list_of_dict = []
        for feature in self.features_names:
            unique_values = self.get_column_unique_values(feature)
            d = {'feature': feature, 'unique_values': unique_values}
            list_of_dict.append(d)
        df_unique_values = pd.DataFrame(list_of_dict)
        return df_unique_values

    def feature_histogram(self, feature: str, show: bool = True) -> None:
        """
            This function plots the histogram of values for a given feature in the dataset.

            Args:
                feature: A string representing the feature name.

                show: A boolean which establish whether to show the plot or save it inside
                    "datasets/{dataset_name}/histograms" folder.
        """
        column = self.get_feature_column(feature)
        plt.figure()
        sns.histplot(data=column)
        plt.title(feature)
        plt.xlabel(feature)
        if show:
            plt.show()
        else:
            prefix = misc_utils.get_relative_path()
            dataset_folder = prefix + f'datasets/{self.dataset_name}/'
            histograms_folder = dataset_folder + 'histograms'
            misc_utils.create_dir(histograms_folder)
            plt.savefig(histograms_folder+'/'+feature+'.png')
            plt.close()

    def features_histogram(self) -> None:
        """
            This function saves the histograms for all the features in the dataset.
        """
        misc_utils.print_with_timestamp('Plotting features histograms...')
        for feature in self.features_names:
            self.feature_histogram(feature, show=False)

    def feature_class_separation(self, feature: str, show: bool = True) -> None:
        """
            This function plots the distribution of a given feature across the labels in the dataset.

            Args:
                feature: A string representing the name of the feature.

                show: A boolean which establish whether to show the plot or save it inside
                    "datasets/{dataset_name}/classes_separation" folder.
        """
        column = self.get_feature_column(feature)
        sns.scatterplot(x=column, y=self.Y)
        plt.title(feature)
        plt.xlabel(feature)
        if show:
            plt.show()
        else:
            prefix = misc_utils.get_relative_path()
            dataset_folder = prefix + f'datasets/{self.dataset_name}/'
            classes_separation_folder = dataset_folder + 'classes_separation'
            misc_utils.create_dir(classes_separation_folder)
            plt.savefig(classes_separation_folder + '/' + feature + '.png')
            plt.close()

    def features_class_separation(self) -> None:
        """
            This function saves the classes separation plots for all the features in the dataset.
        """
        misc_utils.print_with_timestamp('Plotting features separation through classes...')
        for feature in self.features_names:
            self.feature_class_separation(feature, show=False)


class Amr_Uti_Dataset(Dataset):
    """
        This class models the AMR-UTI dataset with its peculiarities. It is a child class which inherits from Dataset.
        The presence of the attributes train_ and test_ is aimed at respecting the original train/test split proposed
        by the authors of the dataset [7].

        Attributes:
            dataset: A pandas DataFrame containing the whole dataset, with both features and class labels.

            train_X: A ndarray containing all the input samples in the training set.

            train_Y: A ndarray containing all the labels associated with the input samples in the training set.

            train_Y_one_hot: A ndarray containing all the one-hot encoded labels associated with the input samples
                in the training set.

            test_X: A ndarray containing all the input samples in the test set.

            test_Y: A ndarray containing all the labels associated with the input samples in the test set.

            test_Y_one_hot: A ndarray containing all the one-hot encoded labels associated with the input samples
                in the training set.
    """

    def __init__(self, force_read: bool = False, day_granularity: int = 180, n_features: int = 30,
                 verbose: bool = False) -> None:
        super().__init__(dataset_name='AMR-UTI')

        prefix = misc_utils.get_relative_path()
        dataset_folder = prefix + 'datasets/AMR-UTI/'
        dataset_folder_features = dataset_folder + str(n_features) + '_features/'
        misc_utils.create_dir(dataset_folder_features)

        train_set_fname = dataset_folder_features + 'preprocessed_train.csv'
        test_set_fname = dataset_folder_features + 'preprocessed_test.csv'

        # we don't use this file
        # data_dictionary_fname = dataset_folder + 'data_dictionary.csv'
        # preprocessing
        if force_read or (not os.path.isfile(train_set_fname) and not os.path.isfile(test_set_fname)):
            if verbose:
                misc_utils.print_with_timestamp('Dataset preprocessing...')
            all_prescriptions_fname = dataset_folder + 'all_prescriptions.csv'
            all_uti_features_fname = dataset_folder + 'all_uti_features.csv'
            all_uti_resist_labels_fname = dataset_folder + 'all_uti_resist_labels.csv'

            # [6]
            # "[this file] Contains empiric clinician prescription selections for specimens in the uncomplicated
            # UTI cohort only.
            # By construction, our uncomplicated cohort is filtered to only contain specimens for which clinicians
            # treated the infection with exactly one treatment in { NIT, SXT, CIP, LVX } in the empiric treatment
            # window.
            # We do not include prescriptions for the other specimens in the dataset, as clinicians may have treated
            # other specimens with multiple antibiotics, or with antibiotics from outside this set."
            all_prescriptions_dataset = pd.read_csv(all_prescriptions_fname)
            # [6]
            # "[this file] Contains constructed features for all specimens. Also contains columns indicating membership
            # of each specimen in training vs. test set and membership in the uncomplicated UTI cohort."
            all_uti_features_dataset = pd.read_csv(all_uti_features_fname)
            # [6]
            # "Contains resistance testing results for the most common antibiotics used for UTI infections
            # (nitrofurantoin (NIT), trimethoprim-sulfamethoxazole (SXT), ciprofloxacin (CIP), and levofloxacin (LVX))
            # for all specimens in our UTI cohort."
            all_uti_resist_dataset = pd.read_csv(all_uti_resist_labels_fname)
            # [6]
            # "We also include a data dictionary file for all of the columns in the files above"
            # we don't use this file
            # data_dictionary_dataset = pd.read_csv(data_dictionary_fname)

            # Since the clinical prescriptions are focused only on the "uncomplicated UTI cohort" (which is the set of
            # specimens which have been prescribed with exactly one treatment), the first thing we have to do is
            # restrict our dataset to this uncomplicated cohort.
            # All the 'example_id"s in "all_prescriptions.csv" are the specimen ids for this cohort.
            uncomplicated_uti_cohort_ids = all_prescriptions_dataset['example_id'].to_list()
            assert len(uncomplicated_uti_cohort_ids) == len(all_prescriptions_dataset)

            # Now, we select only these ids from the other two datasets, namely:
            # 1) "all_uti_resist_labels"
            boolean_mask_all_uti_resist_labels = all_uti_resist_dataset['example_id'].isin(uncomplicated_uti_cohort_ids)
            uncomplicated_cohort_all_uti_resist_labels = all_uti_resist_dataset.loc[boolean_mask_all_uti_resist_labels]
            # we now sort the columns in lexicographic order, and rename them before merging with prescriptions
            uncomplicated_cohort_all_uti_resist_labels = uncomplicated_cohort_all_uti_resist_labels[
                ['example_id', 'CIP', 'LVX', 'NIT', 'SXT', 'is_train']] \
                .rename(columns={
                'NIT': 'resists_NIT',
                'SXT': 'resists_SXT',
                'CIP': 'resists_CIP',
                'LVX': 'resists_LVX'}).drop('is_train', axis=1)
            assert len(uncomplicated_cohort_all_uti_resist_labels) == len(uncomplicated_uti_cohort_ids)

            # 2) "all_uti_features"
            boolean_mask_all_uti_features = all_uti_features_dataset['example_id'].isin(uncomplicated_uti_cohort_ids)
            uncomplicated_cohort_all_uti_features = all_uti_features_dataset.loc[boolean_mask_all_uti_features]\
                .drop('is_train', axis=1)
            assert len(uncomplicated_cohort_all_uti_features) == len(uncomplicated_uti_cohort_ids)

            # For the specific purpose of this study, we do not need to know the results coming from the microbiology
            # laboratory analysis of the urinal specimen. In other words, we do not need to know if the empiric
            # prescription decided from the clinician is effective or not. We only want to alter this empirical
            # decision.
            # df1 = uncomplicated_cohort_all_uti_resist_labels.merge(all_prescriptions_dataset, on='example_id')
            # df2 = df1.merge(uncomplicated_cohort_all_uti_features, on='example_id').drop('uncomplicated', axis=1)
            df2 = all_prescriptions_dataset.merge(uncomplicated_cohort_all_uti_features, on='example_id').\
                drop('uncomplicated', axis=1)

            # There are some feature names which contain the slash '/' character.
            # This may bring to some problems during processing.
            # We rename all these columns, replacing '/' with '_'
            for feature in df2.columns.to_list():
                if '/' in feature:
                    new_feature = feature.replace('/', '_')
                    df2 = df2.rename(columns={feature: new_feature})

            # The four antibiotics in the dataset are of two types: first-line and second-line antibiotics.
            # The second line antibiotics (fluoroquinolone antibiotics, i.e.: ciprofloxacin (CIP)
            # and levofloxacin (LVX)) expose patients to risks of serious adverse events,
            # such as Clostridioides difficile, and tendinopathy [7]
            # For such reason, playing the role of an adversary, we only want to alter the decision among
            # first line antibiotics (i.e., nitrofurantoin (NIT) and trimethoprim-sulfamethoxazole (TMP-SMX))
            # because we cannot risk to cause serious harm to patients, which may bring to
            # being discovered in our malicious intent.
            # In the whole dataset of uncomplicated UTI, these are the prescription counts of the four antibiotics:
            # - NIT: 3250
            # - TMP-SMX: 6183
            # - CIP: 5937
            # - LVX: 436
            # What we are now going to do is filter out those specimens which belong to an empiric treatment decision
            # among the second line antibiotics (CIP, LVX).
            # Note: in the dataset, the TMP-SMX is indicated as SXT
            df2 = df2[(df2['prescription'] != 'CIP') & (df2['prescription'] != 'LVX')]  # 9433 specimens

            # The 74 features with real values are the so called "colonization pressure" features,
            # which the authors constructed averaging the resistance in urine specimens to an antibiotic among
            # all the population.
            # We consider these features as irrelevant for the prediction task, thus,
            # we filter them out from the dataset
            colonization_pressure_labels = ['selected micro - colonization pressure AMC 90 - granular level',
                                            'selected micro - colonization pressure AMP 90 - granular level',
                                            'selected micro - colonization pressure ATM 90 - granular level',
                                            'selected micro - colonization pressure CAZ 90 - granular level',
                                            'selected micro - colonization pressure CIP 90 - granular level',
                                            'selected micro - colonization pressure CLI 90 - granular level',
                                            'selected micro - colonization pressure CRO 90 - granular level',
                                            'selected micro - colonization pressure DOX 90 - granular level',
                                            'selected micro - colonization pressure ERY 90 - granular level',
                                            'selected micro - colonization pressure FEP 90 - granular level',
                                            'selected micro - colonization pressure FOX 90 - granular level',
                                            'selected micro - colonization pressure GEN 90 - granular level',
                                            'selected micro - colonization pressure IPM 90 - granular level',
                                            'selected micro - colonization pressure LVX 90 - granular level',
                                            'selected micro - colonization pressure MEM 90 - granular level',
                                            'selected micro - colonization pressure MXF 90 - granular level',
                                            'selected micro - colonization pressure NIT 90 - granular level',
                                            'selected micro - colonization pressure OXA 90 - granular level',
                                            'selected micro - colonization pressure PEN 90 - granular level',
                                            'selected micro - colonization pressure SAM 90 - granular level',
                                            'selected micro - colonization pressure SXT 90 - granular level',
                                            'selected micro - colonization pressure TET 90 - granular level',
                                            'selected micro - colonization pressure TZP 90 - granular level',
                                            'selected micro - colonization pressure VAN 90 - granular level',
                                            'selected micro - colonization pressure AMC 90 - higher level',
                                            'selected micro - colonization pressure AMP 90 - higher level',
                                            'selected micro - colonization pressure ATM 90 - higher level',
                                            'selected micro - colonization pressure CAZ 90 - higher level',
                                            'selected micro - colonization pressure CIP 90 - higher level',
                                            'selected micro - colonization pressure CLI 90 - higher level',
                                            'selected micro - colonization pressure CRO 90 - higher level',
                                            'selected micro - colonization pressure DOX 90 - higher level',
                                            'selected micro - colonization pressure ERY 90 - higher level',
                                            'selected micro - colonization pressure FEP 90 - higher level',
                                            'selected micro - colonization pressure FOX 90 - higher level',
                                            'selected micro - colonization pressure GEN 90 - higher level',
                                            'selected micro - colonization pressure IPM 90 - higher level',
                                            'selected micro - colonization pressure LVX 90 - higher level',
                                            'selected micro - colonization pressure MEM 90 - higher level',
                                            'selected micro - colonization pressure MIN 90 - higher level',
                                            'selected micro - colonization pressure MXF 90 - higher level',
                                            'selected micro - colonization pressure NIT 90 - higher level',
                                            'selected micro - colonization pressure OXA 90 - higher level',
                                            'selected micro - colonization pressure PEN 90 - higher level',
                                            'selected micro - colonization pressure SAM 90 - higher level',
                                            'selected micro - colonization pressure SXT 90 - higher level',
                                            'selected micro - colonization pressure TET 90 - higher level',
                                            'selected micro - colonization pressure TZP 90 - higher level',
                                            'selected micro - colonization pressure VAN 90 - higher level',
                                            'selected micro - colonization pressure AMC 90 - overall',
                                            'selected micro - colonization pressure AMP 90 - overall',
                                            'selected micro - colonization pressure ATM 90 - overall',
                                            'selected micro - colonization pressure CAZ 90 - overall',
                                            'selected micro - colonization pressure CIP 90 - overall',
                                            'selected micro - colonization pressure CLI 90 - overall',
                                            'selected micro - colonization pressure CRO 90 - overall',
                                            'selected micro - colonization pressure DOX 90 - overall',
                                            'selected micro - colonization pressure ERY 90 - overall',
                                            'selected micro - colonization pressure FEP 90 - overall',
                                            'selected micro - colonization pressure FOX 90 - overall',
                                            'selected micro - colonization pressure GEN 90 - overall',
                                            'selected micro - colonization pressure IPM 90 - overall',
                                            'selected micro - colonization pressure LVX 90 - overall',
                                            'selected micro - colonization pressure MEM 90 - overall',
                                            'selected micro - colonization pressure MIN 90 - overall',
                                            'selected micro - colonization pressure MXF 90 - overall',
                                            'selected micro - colonization pressure NIT 90 - overall',
                                            'selected micro - colonization pressure OXA 90 - overall',
                                            'selected micro - colonization pressure PEN 90 - overall',
                                            'selected micro - colonization pressure SAM 90 - overall',
                                            'selected micro - colonization pressure SXT 90 - overall',
                                            'selected micro - colonization pressure TET 90 - overall',
                                            'selected micro - colonization pressure TZP 90 - overall',
                                            'selected micro - colonization pressure VAN 90 - overall']
            df2.drop(colonization_pressure_labels, axis=1, inplace=True)

            # 4 binary features are used to keep track of which hospital department registered the urinal
            # specimen; we consider these features irrelevant
            hospital_department_labels = ['hosp ward - ER',
                                          'hosp ward - ICU',
                                          'hosp ward - IP',
                                          'hosp ward - OP']
            df2.drop(hospital_department_labels, axis=1, inplace=True)

            # 6 binary features record whether the same patient has provided other specimen to the microbiology
            # laboratories, for example, blood specimens; we consider these features irrelevant
            infection_sites_labels = ['infection_sites - RESPIRATORY_TRACT',
                                      'infection_sites - BLOOD',
                                      'infection_sites - SKIN_SOFTTISSUE',
                                      'infection_sites - ABSCESS_OR_FLUID_NOS',
                                      'infection_sites - MUCOCUTANEOUS',
                                      'infection_sites - GENITOURINARY']
            df2.drop(infection_sites_labels, axis=1, inplace=True)

            # 4 binary features record whether the patient the patient had visited a nursing facilities in 4 different
            # time windows prior to the specimen collection; we consider these features irrelevant
            nursing_home_facilities = ['custom 7 - nursing home',
                                       'custom 14 - nursing home',
                                       'custom 30 - nursing home',
                                       'custom 90 - nursing home']
            df2.drop(nursing_home_facilities, axis=1, inplace=True)

            # 357 binary features describe prior antibiotic exposure of the patients, indicating either the specific
            # antibiotic or the particular class of drugs it belongs to.
            # In order to reduce redundancy, we only consider the features with label
            # medication [TIME_WINDOW] - [ANTIBIOTIC]
            prior_subclass_antibiotic_exposure = [x for x in df2.columns for s in x.split() if s == 'ab']
            df2.drop(prior_subclass_antibiotic_exposure, axis=1, inplace=True)

            # We now restrict the analysis to a prefixed time granularity
            demographics_columns = df2.columns.to_list()[3:6]
            columns_with_time_granularity = df2.columns.to_list()[6:]

            time_granularities = []
            for x in columns_with_time_granularity:
                for s in x.split():
                    if s.isdigit():
                        time_granularities.append(int(s))
                    elif s == 'ALL':
                        time_granularities.append(s)

            chosen_time_granularities = [x for x in df2.columns for s in x.split() if s == str(day_granularity)]
            columns_to_drop = [x for x in df2.columns if
                               (x not in chosen_time_granularities) and
                               (x not in demographics_columns) and
                               (x not in ['example_id', 'prescription', 'is_train'])]
            df2.drop(columns_to_drop, axis=1, inplace=True)

            count_features = dict()
            for x in df2.columns.to_list():
                feature = x.split()[0]
                if feature == 'micro':
                    feature = feature + '_' + x.split()[3]
                if feature in count_features:
                    count_features[feature] += 1
                else:
                    count_features[feature] = 1

            # We consider the demographic information "is_veteran", and "is_white" as irrelevant
            df2.drop('demographics - is_white', axis=1, inplace=True)
            df2.drop('demographics - is_veteran', axis=1, inplace=True)

            # Here we filter out those features which have only 1 value repeated on all the urinal specimens
            nunique = df2.apply(pd.Series.nunique)
            cols_to_drop = nunique[nunique == 1].index
            df2.drop(cols_to_drop, axis=1, inplace=True)

            # Here we filter out features according to the chi-square statistical test of independence between
            # features and classes.
            from sklearn.feature_selection import SelectKBest, chi2
            select_k_best_classifier = SelectKBest(chi2, k=n_features)
            select_k_best_classifier.fit(df2[df2.columns[~df2.columns
                                         .isin(['example_id', 'is_train', 'prescription'])]], df2['prescription'])
            mask = select_k_best_classifier.get_support()
            new_features = []
            for bool, feature in zip(mask, df2.columns[~df2.columns.isin(['example_id', 'is_train', 'prescription'])]):
                if bool:
                    new_features.append(feature)

            columns_to_drop = [x for x in df2.columns[~df2.columns.isin(['example_id', 'is_train', 'prescription'])]
                               if not x in new_features]
            df2.drop(columns_to_drop, axis=1, inplace=True)

            # Here we want to find out if there are rows with duplicate binary features values but different
            # empirical prescription
            columns_to_group = df2.columns[~df2.columns.isin(['example_id', 'is_train',
                                                              'demographics - age', 'prescription'])].to_list()
            indices_to_remove = []
            for name, group in df2.groupby(columns_to_group):
                if len(group) > 1:
                    if group['prescription'].nunique() > 1:
                        indices_to_remove.extend(group.index.to_list())

            df2.drop(indices_to_remove, inplace=True)

            # [6]
            # "This dataset was divided into a training and a test set, based on years. [...]
            # All training specimens are in the years 2007-2013.
            # All test specimens are in the years 2014-2016."
            # We now want to recreate the original split
            self.train_set = df2[df2['is_train'] == 1].drop('is_train', axis=1)\
                                                      .set_index('example_id')
            self.test_set = df2[df2['is_train'] == 0].drop('is_train', axis=1)\
                                                     .set_index('example_id')

            self.train_set.to_csv(train_set_fname, index=True)
            self.test_set.to_csv(test_set_fname, index=True)
        else:
            self.train_set = pd.read_csv(train_set_fname).set_index('example_id')
            self.test_set = pd.read_csv(test_set_fname).set_index('example_id')

        self.dataset = pd.concat([self.train_set, self.test_set])
        self.Y = self.dataset['prescription'].to_numpy()
        self.Y_one_hot = class_utils.one_hot(self.Y, encode_decode=0)
        self.classes_names = ['NIT', 'SXT']
        self.features_names = self.dataset.columns.drop('prescription').to_list()
        self.X = self.dataset.drop('prescription', axis=1).to_numpy()
        self.class_label = 'prescription'
        self.not_corruptible_features = [
                                    "demographics - age",
                                    "micro - prev resistance NIT "+str(day_granularity),
                                    "micro - prev resistance SXT "+str(day_granularity),
                                    "medication 180 - nitrofurantoin",
                                    "medication 180 - trimethoprim_sulfamethoxazole"
                                    ]
        self.perturbation_mask = np.ones(len(self.features_names))
        for feature in self.not_corruptible_features:
            if feature in self.features_names:
                idx = self.features_names.index(feature)
                self.perturbation_mask[idx] = 0

        self.train_X = self.train_set.drop('prescription', axis=1).to_numpy()
        self.train_Y = self.train_set['prescription'].to_numpy()
        self.train_Y_one_hot = class_utils.one_hot(self.train_Y, encode_decode=0)

        self.test_X = self.test_set.drop('prescription', axis=1).to_numpy()
        self.test_Y = self.test_set['prescription'].to_numpy()
        self.test_Y_one_hot = class_utils.one_hot(self.test_Y, encode_decode=0)

        if verbose:
            misc_utils.print_with_timestamp('Dataset loaded.')
