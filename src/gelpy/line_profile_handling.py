import numpy as np
import matplotlib.pyplot as plt
from .utility_functions import cm_to_inch
import re

class LineProfiles:
    def __init__(self, gel_image, labels, x_label_positions, select_lanes, slice_line_profile_length, normalization_type,
                 save_overview):
        """
        Initializer for LineProfiles class. Sets up important parameters for the line profile analysis.

        Parameters:
        gel_image (numpy.ndarray): The gel image to perform the analysis on.
        labels (list): The labels associated with each lane.
        x_label_positions (list): The positions of the lanes in the gel image.
        select_lanes (list, str, or range): The lanes to include in the analysis. 
                                             Can be a list of indices or labels, a range, or the string 'all'.
        slice_line_profile_length (tuple): The start and end points for slicing the line profiles.
        normalization_type (str): The type of normalization to apply to the line profiles.
        save_overview (bool or str): Whether and where to save the overview of the line profiles. 
                                      If True, saves to "overview_fitted_line_profiles.png". 
                                      If a string, uses the string as the file name.
        """
        self.gel_image = gel_image
        self.labels = labels
        self.x_label_positions = x_label_positions
        self.select_lanes = select_lanes
        self.slice_start, self.slice_end = slice_line_profile_length
        self.normalization_type = normalization_type
        self.save_overview = save_overview

    def _get_lane_indices(self):
        """
        Gets the indices of the lanes to include in the analysis based on the 'select_lanes' attribute.

        Returns:
        list: A list of the indices of the selected lanes.
        Raises a ValueError if an invalid 'select_lanes' is provided.
        """
        if self.select_lanes == "all":
            return list(range(len(self.labels)))
        elif isinstance(self.select_lanes, list):
            indices = []
            for i in self.select_lanes:
                if isinstance(i, int):
                    indices.append(i)
                elif isinstance(i, str):
                    indices.extend([j for j, label in enumerate(self.labels) if re.search(i, label)])
            return list(set(indices))  # Remove duplicates by converting to a set and then back to a list
        elif isinstance(self.select_lanes, str):
            return [i for i, label in enumerate(self.labels) if re.search(self.select_lanes, label)]
        elif isinstance(self.select_lanes, range):
            return list(self.select_lanes)
        else:
            raise ValueError("Invalid value for select_lanes.")


    def extract_line_profiles(self):
        """
        Extracts the line profiles from the selected lanes.
        Stores the results in the 'selected_lanes', 'selected_labels', and 'selected_line_profiles' attributes.
        """
        indices = self._get_lane_indices()
        self.selected_lanes = [self.x_label_positions[i] for i in indices]
        self.selected_labels = [self.labels[i] for i in indices]
        self.selected_line_profiles = [self.extract_line_profile(selected_lane) for selected_lane in self.selected_lanes]

    def extract_line_profile(self, selected_lane):
        """
        Extracts the line profile from a single lane.

        Parameters:
        selected_lane (int): The position of the lane in the gel image.

        Returns:
        numpy.ndarray: The line profile.
        """
        start_x = int(selected_lane - (self.line_profile_width / 2))
        end_x = int(selected_lane + (self.line_profile_width / 2))
        line_profile = np.mean(self.gel_image[:, start_x:end_x], axis=1)
        line_profile = line_profile[self.slice_start:self.slice_end]
        return line_profile

    def _normalize_line_profiles_to_type(self, line_profile):
        """
        Normalizes a line profile according to the 'normalization_type' attribute.

        Parameters:
        line_profile (numpy.ndarray): The line profile to normalize.

        Returns:
        numpy.ndarray: The normalized line profile.
        Raises a ValueError if an invalid 'normalization_type' is provided.
        """
        if self.normalization_type == "min_max":
            return self.normalize_line_profile_to_min_max(line_profile)
        elif self.normalization_type == "area":
            return self.normalize_line_profile_to_area(line_profile)
        else:
            raise ValueError("Invalid normalization type. Expected 'min_max' or 'area'")

    def normalize_line_profiles(self):
        """
        Normalizes all the selected line profiles according to the 'normalization_type' attribute.
        Stores the results in the 'selected_line_profiles_normalized' attribute.
        """
        self.selected_line_profiles_normalized = [
            self._normalize_line_profiles_to_type(line_profile) for line_profile in self.selected_line_profiles]

    def normalize_line_profile_to_min_max(self, line_profile):
        """
        Normalizes a line profile to the range [0, 1] based on the minimum and maximum values.

        Parameters:
        line_profile (numpy.ndarray): The line profile to normalize.

        Returns:
        numpy.ndarray: The normalized line profile.
        """
        min_val = np.min(line_profile)
        max_val = np.max(line_profile)
        return (line_profile - min_val) / (max_val - min_val)

    def normalize_line_profile_to_area(self, line_profile):
        """
        Normalizes a line profile to its area using the trapezoidal rule.

        Parameters:
        line_profile (numpy.ndarray): The line profile to normalize.

        Returns:
        numpy.ndarray: The normalized line profile.
        """
        area = np.trapz(line_profile)
        return line_profile / area
        
 
    def set_line_profile_width(self, line_profile_width):
        """
        Sets the line profile width. If 'line_profile_width' is None, guesses the width.

        Parameters:
        line_profile_width (int): The width of the line profile.
        Raises an AssertionError if 'line_profile_width' is not a positive integer.
        """
        assert isinstance(line_profile_width, int) and line_profile_width > 0, "Invalid line_profile_width"
        self.line_profile_width = line_profile_width
        if self.x_label_positions is not None and line_profile_width is None:
            self.line_profile_width = self.guess_line_profile_width()

    def guess_line_profile_width(x_label_positions, gel_image, line_profile_width):
        """
        Guesses the line profile width based on the gel image and the lane positions.

        Parameters:
        x_label_positions (list): The positions of the lanes in the gel image.
        gel_image (numpy.ndarray): The gel image to perform the analysis on.
        line_profile_width (int): The width of the line profile.

        Returns:
        int: The guessed line profile width.
        """
        if line_profile_width != None: # A workaround, to allow setup_gel to pass through custom line_profiles.
            return line_profile_width
        else:
            n = len(x_label_positions)
            line_profile_width = int(gel_image.shape[1] / (n * 2.5)) #2.5 is heuristic value, and depends on the gelcomb used
            return line_profile_width      

    def plot_selected_line_profiles(self):
        """
        Plots the selected line profiles. If only one profile is selected, also plots the raw intensity on the right y-axis.

        Returns:
        matplotlib.figure.Figure: The plotted figure.
        Raises a ValueError if an invalid 'save_overview' is provided.
        """
        fig, ax1 = plt.subplots(figsize=(cm_to_inch(18), cm_to_inch(10)))
        ax2 = ax1.twinx()
        x_axis = range(self.slice_start, self.slice_start + len(self.selected_line_profiles[0]))
        for lane_id, _ in enumerate(self.selected_line_profiles_normalized):
            ax1.plot(x_axis, self.selected_line_profiles_normalized[lane_id], label=self.selected_labels[lane_id])
            # If there is only one profile, plot the raw intensity line profile on the right y-axis with transparent color
            if len(self.selected_line_profiles) == 1:
                self.plot_raw_intensity(ax2, x_axis, self.selected_line_profiles[lane_id])
            else:
                ax2.axis('off')  # Remove the right y-axis
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.set(ylabel=f"normalized to {self.normalization_type}", xlabel="[px]")
        
        # Saving the figure
        if self.save_overview == None or self.save_overview == False:
            return
        elif self.save_overview == True:
            fig.savefig("overview_fitted_line_profiles.png", bbox_inches="tight")
        elif isinstance(self.save_overview, str):
            fig.savefig(self.save_overview, bbox_inches="tight")
        else:
            raise ValueError("save_overview must be a filename or True")

        return fig

    def plot_raw_intensity(self, axis, x_axis, line_profile):
        """
        Plots the raw intensity of a line profile on the provided axis.

        Parameters:
        axis (matplotlib.axes.Axes): The axis to plot on.
        x_axis (range): The x-axis values.
        line_profile (numpy.ndarray): The line profile to plot.
        """
        axis.plot(x_axis, line_profile, color='none')
        axis.set(ylabel="raw intensity")
        axis.yaxis.set_tick_params(labelleft=False)  # Hide the right y-axis label