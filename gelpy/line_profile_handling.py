import numpy as np
import matplotlib.pyplot as plt
from .utility_functions import cm_to_inch

class LineProfiles:
    def __init__(self, gel_image, labels, x_label_positions, select_lanes, slice_line_profile_length, normalization_type,
                 save, save_name_overview):
        self.gel_image = gel_image
        self.labels = labels
        self.x_label_positions = x_label_positions
        self.select_lanes = select_lanes
        self.slice_start, self.slice_end = slice_line_profile_length
        self.normalization_type = normalization_type
        self.save = save
        self.save_name_overview = save_name_overview

    def _get_lane_indices(self):
        if self.select_lanes == "all":
            return list(range(len(self.labels)))
        elif isinstance(self.select_lanes, list):
            if all(isinstance(i, int) for i in self.select_lanes):
                return self.select_lanes
            elif all(isinstance(i, str) for i in self.select_lanes):
                return [i for i, label in enumerate(self.labels) if label in self.select_lanes]
        elif isinstance(self.select_lanes, str):
            return [i for i, label in enumerate(self.labels) if label == self.select_lanes]
        elif isinstance(self.select_lanes, range):
            return list(self.select_lanes)
        else:
            raise ValueError("Invalid value for select_lanes.")


    def extract_line_profiles(self):
        indices = self._get_lane_indices()
        self.selected_lanes = [self.x_label_positions[i] for i in indices]
        self.selected_labels = [self.labels[i] for i in indices]
        self.selected_line_profiles = [self.extract_line_profile(selected_lane) for selected_lane in self.selected_lanes]

    def extract_line_profile(self, selected_lane):
        start_x = int(selected_lane - (self.line_profile_width / 2))
        end_x = int(selected_lane + (self.line_profile_width / 2))
        line_profile = np.mean(self.gel_image[:, start_x:end_x], axis=1)
        line_profile = line_profile[self.slice_start:self.slice_end]
        return line_profile

    def _normalize_line_profiles_to_type(self, line_profile):
        if self.normalization_type == "min_max":
            return self.normalize_line_profile_to_min_max(line_profile)
        elif self.normalization_type == "area":
            return self.normalize_line_profile_to_area(line_profile)
        else:
            raise ValueError("Invalid normalization type. Expected 'min_max' or 'area'")

    def normalize_line_profiles(self):
        self.selected_line_profiles_normalized = [
            self._normalize_line_profiles_to_type(line_profile) for line_profile in self.selected_line_profiles]

    def normalize_line_profile_to_min_max(self, line_profile):
        min_val = np.min(line_profile)
        max_val = np.max(line_profile)
        return (line_profile - min_val) / (max_val - min_val)

    def normalize_line_profile_to_area(self, line_profile):
        area = np.trapz(line_profile)
        return line_profile / area
        
 
    def set_line_profile_width(self, line_profile_width):
        assert isinstance(line_profile_width, int) and line_profile_width > 0, "Invalid line_profile_width"
        self.line_profile_width = line_profile_width
        if self.x_label_positions is not None and line_profile_width is None:
            self.line_profile_width = self.guess_line_profile_width()

    def guess_line_profile_width(x_label_positions, gel_image, line_profile_width):
        if line_profile_width != None: # A workaround, to allow setup_gel to pass through custom line_profiles.
            return line_profile_width
        else:
            n = len(x_label_positions)
            line_profile_width = int(gel_image.shape[1] / (n * 2.5)) #2.5 is heuristic value, and depends on the gelcomb used
            print(f"Used line width: {line_profile_width} px, with a gel width of: {gel_image.shape[1]} px, and {n} xlabel positions")
            return line_profile_width      

    def plot_selected_line_profiles(self):
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
        ax1.set(ylabel="normalized", xlabel="Pixel")
        
        if self.save:
            fig.savefig(self.save_name_overview, bbox_inches='tight')

        return fig

    def plot_raw_intensity(self, axis, x_axis, line_profile):
        axis.plot(x_axis, line_profile, color='none')
        axis.set(ylabel="raw intensity")
        axis.yaxis.set_tick_params(labelleft=False)  # Hide the right y-axis labels