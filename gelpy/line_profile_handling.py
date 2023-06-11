import numpy as np
import matplotlib.pyplot as plt

class LineProfiles:
    def __init__(self, gel_image, labels, x_label_positions, select_lanes, slice_line_profile_length, normalization_type,
                 save, save_name_overview):
        self.gel_image = gel_image
        self.labels = labels
        #Dummy attributes
        
        #self.slice_start, self.slice_end = None, None
        self.x_label_positions = x_label_positions
        self.select_lanes = select_lanes
        self.slice_start, self.slice_end = slice_line_profile_length
        self.normalization_type = normalization_type
        self.save = save
        self.save_name_overview = save_name_overview
        

    @staticmethod
    def cm_to_inch(cm):
        return cm * 0.393701

    def extract_line_profiles(self):
        if self.select_lanes == "all":
            indices = range(len(self.labels))
        elif isinstance(self.select_lanes, list) and all(isinstance(i, int) for i in self.select_lanes):
            indices = self.select_lanes
        elif isinstance(self.select_lanes, list) and all(isinstance(i, str) for i in self.select_lanes):
            indices = [i for i, label in enumerate(self.labels) if label in self.select_lanes]
        elif isinstance(self.select_lanes, str):
            indices = [i for i, label in enumerate(self.labels) if label == self.select_lanes] 

        self.selected_lanes = [self.x_label_positions[i] for i in indices]
        self.selected_labels = [self.labels[i] for i in indices]

        self.selected_line_profiles = []
        for selected_lane in self.selected_lanes:
            line_profile = self.extract_line_profile(selected_lane)
            self.selected_line_profiles.append(line_profile)
            
    def extract_line_profile(self, selected_lane):
        start_x = int(selected_lane - (self.line_profile_width / 2))
        end_x = int(selected_lane + (self.line_profile_width / 2))
        line_profile = np.mean(self.gel_image[:, start_x:end_x], axis=1)
        # cut legth of profiles
        line_profile = line_profile[self.slice_start:self.slice_end]
        return line_profile

    def normalize_line_profiles(self):
        if self.normalization_type == "min_max":
            self.selected_line_profiles_normalized = [self.normalize_line_profile_to_min_max(line_profile,) for line_profile in self.selected_line_profiles]
        elif self.normalization_type == "area":
            self.selected_line_profiles_normalized = [self.normalize_line_profile_to_area(line_profile,) for line_profile in self.selected_line_profiles]

    def normalize_line_profile_to_min_max(self, line_profile):
        min_val = np.min(line_profile)
        max_val = np.max(line_profile)
        return (line_profile - min_val) / (max_val - min_val)
    
    def normalize_line_profile_to_area(self, line_profile):
        area = np.trapz(line_profile)
        return line_profile / area
        
    def set_line_profile_width(self, line_profile_width):
        self.line_profile_width = line_profile_width
        if self.x_label_positions is not None and line_profile_width is None:
            self.line_profile_width = self.guess_line_profile_width(self.x_label_positions,self.gel_image, self.line_profile_width)
                
    @staticmethod
    def guess_line_profile_width(x_label_positions, gel_image, line_profile_width):
        n = len(x_label_positions)
        line_profile_width = int(gel_image.shape[1] / (n * 2.5)) #2.5 is heuristic value, and depends on the gelcomb used
        print(f"Used line width: {line_profile_width} px, with a gel width of: {gel_image.shape[1]} px, and {n} xlabel positions")
        return line_profile_width      

    def plot_selected_line_profiles(self):
        fig, ax1 = plt.subplots(figsize=(self.cm_to_inch(18), self.cm_to_inch(10)))
        ax2 = ax1.twinx()
        x_axis = range(self.slice_start, self.slice_start + len(self.selected_line_profiles[0]))
        for lane_id, _ in enumerate(self.selected_line_profiles_normalized):
            ax1.plot(x_axis, self.selected_line_profiles_normalized[lane_id], label=self.selected_labels[lane_id])
            # If there is only one profile, plot the raw intensity line profile on the right y-axis with transparent color
            if len(self.selected_line_profiles) == 1:
                ax2.plot(x_axis, self.selected_line_profiles[lane_id], color='none')
                ax2.set(ylabel="raw intensity")
                ax2.yaxis.set_tick_params(labelleft=False)  # Hide the right y-axis labels
            else:
                ax2.axis('off')  # Remove the right y-axis
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.set(ylabel="normalized", xlabel="Pixel")
        
        if self.save:
            fig.savefig(self.save_name_overview, bbox_inches='tight')

        return fig