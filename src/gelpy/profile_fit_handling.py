import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display
from .utility_functions import cm_to_inch

class LineFits:
    """
    A class used to represent Line Fits for a gel image. This includes various operations such as fitting a model to 
    the selected normalized line profiles, plotting the fits, and displaying or saving the fit data.

    Attributes:
        selected_normalized_line_profiles (list): A list of selected normalized line profiles.
        selected_labels (list): A list of selected labels corresponding to the line profiles.
        maxima_threshold (float): A threshold for detecting maxima in the line profiles.
        maxima_prominence (float): A prominence value for detecting maxima. 
        peak_width (int): The width of the peaks in the line profile.
        sigma (float): A parameter for the Gaussian fitting method.
        fit_model (object): The fitting model to be used.
        save_fits (bool/str): A boolean or string value to indicate if and how to save fits.

    Methods:
        fit(): Fits the selected model to the normalized line profiles and creates a fit dataframe.
        plot_fits_and_profiles(): Plots the fits and the line profiles.
        plot_left_subplot(): Helper function to plot the left subplot in the graph.
        plot_right_subplot(): Helper function to plot the right subplot in the graph.
        format_right_subplot(): Helper function to format the right subplot in the graph.
        display_dataframe(show_df): Displays the fit dataframe if show_df is True.
        check_if_save_dataframe(save_df): Saves the fit dataframe if save_df is True or a valid filename.
    """
    def __init__(self, fit_model, selected_normalized_line_profiles, selected_labels, maxima_threshold, maxima_prominence, peak_width, sigma ,
                 save_fits):
        """
        Constructs all the necessary attributes for the LineFits object.

        Args:
            fit_model (object): The fitting model to be used.
            selected_normalized_line_profiles (list): A list of selected normalized line profiles.
            selected_labels (list): A list of selected labels corresponding to the line profiles.
            maxima_threshold (float): A threshold for detecting maxima in the line profiles.
            maxima_prominence (float): A prominence value for detecting maxima.
            peak_width (int): The width of the peaks in the line profile.
            sigma (float): A parameter for the Gaussian fitting method.
            save_fits (bool/str): A boolean or string value to indicate if and how to save fits. It can be either a 
                boolean value or a string representing the filename. If True, fits will be saved with a default filename. 
                If a string is provided, it will be used as the filename.
        """
        self.selected_normalized_line_profiles = selected_normalized_line_profiles
        self.selected_labels = selected_labels
        self.maxima_threshold = maxima_threshold
        self.maxima_prominence = maxima_prominence
        self.peak_width = peak_width
        self.sigma = sigma
        self.fit_model = fit_model()
        self.save_fits = save_fits

    def fit(self):
        """
        Fits the selected model to the normalized line profiles and creates a fit dataframe.
        """
        self.fit_model.fit(self.selected_normalized_line_profiles, self.maxima_threshold, self.maxima_prominence, self.peak_width, self.sigma, self.selected_labels)
        self.fit_model.create_fit_dataframe(self.selected_normalized_line_profiles)
    
    def plot_fits_and_profiles(self):
        """
        Plots the fits and the line profiles. If the save_fits attribute is True or a valid filename, the plots will be 
        saved. If the save_fits attribute is neither True nor a valid filename, a ValueError will be raised.

        Raises:
            ValueError: If save_fits is not a boolean or a valid filename string.
        """
        n_profiles = len(self.selected_normalized_line_profiles)
        fig, axs = plt.subplots(n_profiles, 2, figsize=(cm_to_inch(18), n_profiles*cm_to_inch(7)), sharey='row', squeeze=False)

        for i, (selected_lane_index, selected_label, optimized_parameters) in enumerate(self.fit_model.fitted_peaks):
            x = np.arange(len(self.selected_normalized_line_profiles[selected_lane_index])).astype("float64")
            add_legend = (i==0) #for the first, left plot, add a legend
            self.plot_left_subplot(axs[i, 0], x, selected_lane_index, selected_label, optimized_parameters, add_legend)
            self.plot_right_subplot(axs[i, 1], x, selected_lane_index, optimized_parameters, selected_label)

        fig.suptitle('Line profiles and fitted peaks')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plt.show()
        
        if self.save_fits == None or self.save_fits == False:
            return
        elif self.save_fits == True:
            fig.savefig("overview_selected_line_profiles.png")
        elif isinstance(self.save_fits, str):
            fig.savefig(self.save_fits)
        else:
            raise ValueError("save_fits must be a filename or True")



    def plot_left_subplot(self, ax, x, selected_lane_index, selected_label, optimized_parameters, add_legend):
        """
        Plots the left subplot in the graph.

        Args:
            ax (matplotlib.axes.Axes): The axes object to draw the plot on.
            x (np.array): The x-axis values for the plot.
            selected_lane_index (int): The index of the selected lane.
            selected_label (str): The label of the selected lane.
            optimized_parameters (list): A list of optimized parameters for the fit model.
            add_legend (bool): Whether or not to add a legend to the plot.
        """
        line1, = ax.plot(x, self.selected_normalized_line_profiles[selected_lane_index], color='black', alpha=0.9, zorder=0.5, label="norm. line profile")
        ax.set_title(f'Normalized Line Profile - {selected_label}')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Intensity normalized to area')
        line2, = ax.plot(x, self.fit_model.multi_peak_function(x, *optimized_parameters), color='black', linestyle='dotted', label="fitted profile")

        if add_legend:
            ax.legend(handles=[line1, line2])

    def plot_right_subplot(self, ax, x, selected_lane_index, optimized_parameters, selected_label): # added selected_label
        """
        Plots the right subplot in the graph.

        Args:
            ax (matplotlib.axes.Axes): The axes object to draw the plot on.
            x (np.array): The x-axis values for the plot.
            selected_lane_index (int): The index of the selected lane.
            optimized_parameters (list): A list of optimized parameters for the fit model.
            selected_label (str): The label of the selected lane.
        """
        for j in range(0, len(optimized_parameters), self.fit_model.params_per_peak()):
            optimized_parameters_structured = optimized_parameters[j:j+self.fit_model.params_per_peak()]
            ax.plot(x, self.fit_model.single_peak_function(x, *optimized_parameters_structured), label=f'Band {j//3 + 1}')

        ax.set_title(f'Fitted Peaks - {selected_label}') # now selected_label is defined in the function scope
        ax.plot(x, self.selected_normalized_line_profiles[selected_lane_index], color='black', alpha=0.9, zorder=0.5)
        ax.set_xlabel('Pixel')
        self.format_right_subplot(ax, selected_lane_index)

    def format_right_subplot(self, ax, selected_lane_index):
        """
        Formats the right subplot in the graph.

        Args:
            ax (matplotlib.axes.Axes): The axes object to draw the plot on.
            selected_lane_index (int): The index of the selected lane.
        """
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
    
    def display_dataframe(self, show_df):
        """
        Displays the fit dataframe if show_df is True.

        Args:
            show_df (bool): Whether or not to display the dataframe.
        """
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.precision', 3)
        
        if show_df == True:
            display(self.fit_model.fit_df)
        
    def check_if_save_dataframe(self, save_df):
        """
        Saves the fit dataframe if save_df is True or a valid filename. If save_df is neither True nor a valid filename, 
        a ValueError will be raised.

        Args:
            save_df (bool/str): A boolean or string value to indicate if and how to save the dataframe. It can be either 
                a boolean value or a string representing the filename. If True, dataframe will be saved with a default 
                filename. If a string is provided, it will be used as the filename.

        Raises:
            ValueError: If save_df is not a boolean or a valid filename string.
        """
        if save_df == None or save_df == False:
            return
        elif save_df == True:
            self.fit_model.fit_df.to_csv("fitted_selected_line_profiles.csv", index=False)
        elif isinstance(save_df, str):
            self.fit_model.fit_df.to_csv(save_df, index=False)
        else:
            raise ValueError("save_df must be a filename or True")