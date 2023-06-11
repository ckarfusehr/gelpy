import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

class LineFits:
    def __init__(self, fit_model, selected_normalized_line_profiles, selected_labels, maxima_threshold, maxima_prominence):
        self.selected_normalized_line_profiles = selected_normalized_line_profiles
        self.selected_labels = selected_labels
        self.maxima_threshold = maxima_threshold
        self.maxima_prominence = maxima_prominence
        self.fit_model = fit_model()

    @staticmethod
    def cm_to_inch(cm):
        return cm * 0.393701

    def fit(self):
        self.fit_model.fit(self.selected_normalized_line_profiles, self.maxima_threshold, self.maxima_prominence, self.selected_labels)
        self.fit_model.create_fit_dataframe(self.selected_normalized_line_profiles)
    
    ## To-do: remove the if statements for the fit model, and change the single_peak functions instead or something.
    def plot_fits_and_profiles(self):
        # Create a figure with two columns and as many rows as there are normalized_line_profiles
        n_profiles = len(self.selected_normalized_line_profiles)
        fig, axs = plt.subplots(n_profiles, 2, figsize=(LineFits.cm_to_inch(18), n_profiles*LineFits.cm_to_inch(7)), sharey='row', squeeze=False)

        for i, (selected_lane_index, selected_label, optimized_parameters) in enumerate(self.fit_model.fitted_peaks):
            x = np.arange(len(self.selected_normalized_line_profiles[selected_lane_index]))

            # Plot the normalized_line_profile in the left subplot
            axs[i, 0].plot(x, self.selected_normalized_line_profiles[selected_lane_index], color='black', alpha=0.5)
            axs[i, 0].set_title(f'Normalized Line Profile - {selected_label}')
            axs[i, 0].set_xlabel('Pixel')
            axs[i, 0].set_ylabel('Intensity normalized to area')

            # Plot the sum of all peaks fitted for this line profile in the left subplot
            axs[i, 0].plot(x, self.fit_model.multi_peak_function(x, *optimized_parameters), color='black', linestyle='dotted')
            
            # Plot all the peaks fitted for this line profile in the right subplot
            for j in range(0, len(optimized_parameters) - 2, self.fit_model.params_per_peak()):  # Exclude the last two parameters for the linear background
                optimized_parameters_structured = optimized_parameters[j:j+self.fit_model.params_per_peak()]
                print(f"opt_params_for_ind_plot: {optimized_parameters_structured}")
                axs[i, 1].plot(x, self.fit_model.single_peak_function(x, *optimized_parameters_structured)
                            + self.fit_model.linear_background_function(x, optimized_parameters[-2], optimized_parameters[-1]), label=f'Band {j//3 + 1}')

            axs[i, 1].plot(x, self.fit_model.linear_background_function(x, optimized_parameters[-2], optimized_parameters[-1]), color='black')
            axs[i, 1].set_title(f'Fitted Peaks - {selected_label}')

            # Plot the original normalized_line_profile in the right subplot
            axs[i, 1].plot(x, self.selected_normalized_line_profiles[selected_lane_index], color='black', alpha=0.5)
            axs[i, 1].set_xlabel('Pixel')
            
            # Remove the y-axis label and ticks of the right plots
            axs[i, 1].set_ylabel('')
            axs[i, 1].yaxis.set_tick_params(left=False, labelleft=False)

            # Create x-axis ticks on the top of the plot, at the maxima of each fit
            maxima_positions = self.fit_model.fit_df.loc[self.fit_model.fit_df['selected_lane_index'] == selected_lane_index, "maxima_position"].values
            axs[i, 1].xaxis.set_ticks(maxima_positions)
            axs[i, 1].xaxis.set_ticks_position('top')
            axs[i, 1].xaxis.set_label_position('top')

            # Label those peak ticks with the relative peak area
            relative_areas = self.fit_model.fit_df.loc[self.fit_model.fit_df['selected_lane_index'] == selected_lane_index, 'relative_area'].values
            labels = [f"{int(np.round((area * 100), 0))} %" for area in relative_areas]
            print(labels)
            axs[i, 1].set_xticklabels(labels)

            # Rotate the x-axis labels if they overlap
            plt.setp(axs[i, 1].xaxis.get_majorticklabels(), rotation=90)

            # Remove the label of the x-axis of the right plots
            axs[i, 1].set_xlabel('')

        plt.tight_layout()
        plt.show()
        
    def display_dataframe(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.precision', 3)
        display(self.fit_model.fit_df)