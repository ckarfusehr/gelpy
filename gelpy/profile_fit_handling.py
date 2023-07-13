import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display
from .utility_functions import cm_to_inch

class LineFits:
    def __init__(self, fit_model, selected_normalized_line_profiles, selected_labels, maxima_threshold, maxima_prominence,
                 save, save_name_fits):
        self.selected_normalized_line_profiles = selected_normalized_line_profiles
        self.selected_labels = selected_labels
        self.maxima_threshold = maxima_threshold
        self.maxima_prominence = maxima_prominence
        self.fit_model = fit_model()
        self.save = save
        self.save_name_fits = save_name_fits

    def fit(self):
        self.fit_model.fit(self.selected_normalized_line_profiles, self.maxima_threshold, self.maxima_prominence, self.selected_labels)
        self.fit_model.create_fit_dataframe(self.selected_normalized_line_profiles)
    
    def plot_fits_and_profiles(self):
        n_profiles = len(self.selected_normalized_line_profiles)
        fig, axs = plt.subplots(n_profiles, 2, figsize=(cm_to_inch(18), n_profiles*cm_to_inch(7)), sharey='row', squeeze=False)

        for i, (selected_lane_index, selected_label, optimized_parameters) in enumerate(self.fit_model.fitted_peaks):
            x = np.arange(len(self.selected_normalized_line_profiles[selected_lane_index])).astype("float64")

            self.plot_left_subplot(axs[i, 0], x, selected_lane_index, selected_label, optimized_parameters)
            self.plot_right_subplot(axs[i, 1], x, selected_lane_index, optimized_parameters, selected_label)

        # Set the figure level title
        fig.suptitle('Line profiles and fitted peaks')

        # You might need to adjust the subplots to not overlap the new title
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plt.show()
        if self.save:
            fig.savefig(self.save_name_fits)


    def plot_left_subplot(self, ax, x, selected_lane_index, selected_label, optimized_parameters):
        sns.lineplot(x=x, y=self.selected_normalized_line_profiles[selected_lane_index], color='black', alpha=0.9, ax=ax)
        ax.set_title(f'Normalized Line Profile - {selected_label}')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Intensity normalized to area')
        sns.lineplot(x=x, y=self.fit_model.multi_peak_function(x, *optimized_parameters), color='black', linestyle='dotted', ax=ax)

    def plot_right_subplot(self, ax, x, selected_lane_index, optimized_parameters, selected_label):
        for j in range(0, len(optimized_parameters), self.fit_model.params_per_peak()):
            optimized_parameters_structured = optimized_parameters[j:j+self.fit_model.params_per_peak()]
            sns.lineplot(x=x, y=self.fit_model.single_peak_function(x, *optimized_parameters_structured), label=f'Band {j//3 + 1}', ax=ax)

        ax.set_title(f'Fitted Peaks - {selected_label}')
        sns.lineplot(x=x, y=self.selected_normalized_line_profiles[selected_lane_index], color='black', alpha=0.9, ax=ax)
        ax.set_xlabel('Pixel')
        self.format_right_subplot(ax, selected_lane_index)

    def format_right_subplot(self, ax, selected_lane_index):
        ax.set_ylabel('')
        ax.yaxis.set_tick_params(left=False, labelleft=False)
        maxima_positions = self.fit_model.fit_df.loc[self.fit_model.fit_df['selected_lane_index'] == selected_lane_index, "maxima_position"].values
        ax.xaxis.set_ticks(maxima_positions)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        relative_areas = self.fit_model.fit_df.loc[self.fit_model.fit_df['selected_lane_index'] == selected_lane_index, 'relative_area'].values
        labels = [f"{int(np.round((area * 100), 0))} %" for area in relative_areas]
        ax.set_xticklabels(labels)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        ax.set_xlabel('')
    
    def display_dataframe(self, show_df, save_df, df_save_name):
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.precision', 3)
        
        if show_df == True:
            display(self.fit_model.fit_df)
        if save_df == True:
            self.fit_model.fit_df.to_csv(df_save_name, index=False)
        else:
            return