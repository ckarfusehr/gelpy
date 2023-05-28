from .image_handling import Image
from .line_profile_handling import LineProfiles
from .profile_fitting_models import GaussianFitModel, EmgFitModel
from .profile_fit_handling import LineFits

class AgaroseGel:
    def __init__(self, path, labels):
        self.path = path
        self.labels = labels
        
        # Dummy attributes
        
        # Methods
        self.Image = Image(self.path, self.labels)
        
    def show_raw_gel(self):
        self.Image.show_raw_image()
        return
    
    def show_adjusted_images(self, x_label_pos, gamma=0.1, gain=1, intensity_range=(0,6000), img_height_factor=0.009, label_rotation=45):
        self.Image.gamma, self.Image.gain, self.Image.intensity_range = gamma, gain, intensity_range
        self.Image.img_height_factor, self.Image.label_rotation = img_height_factor, label_rotation
        self.Image.x_label_pos = x_label_pos
        self.Image.adjust_img_contrast_non_linear()
        self.Image.adjust_img_contrast_linear()
        #save_fig(fig, collage_file_path, save)
        self.x_label_positions = self.Image.plot_adjusted_gels()
        return
    
    def show_line_profiles(self, select_lanes="all", line_profile_width=None, slice_line_profile_length=(0,-1),
                           fit=None, maxima_threshold=0.001, maxima_prominence=None, plot_fits=False, normalization_type="area"):
        
        self.LineProfiles = LineProfiles(self.Image.gel_image, self.labels, self.x_label_positions,
                                         select_lanes, slice_line_profile_length, normalization_type)
        self.LineProfiles.set_line_profile_width(line_profile_width)
        self.LineProfiles.extract_line_profiles()
        self.LineProfiles.normalize_line_profiles()
        self.LineProfiles.plot_selected_line_profiles()
        
        if fit=="gaussian":
            fit_model = GaussianFitModel
        elif fit == "emg":
            fit_model = EmgFitModel
        else:
            print(f"Invalid fit type")
            return
            
        self.LineFits = LineFits(fit_model, self.LineProfiles.selected_line_profiles_normalized, self.LineProfiles.selected_labels,
                                    maxima_threshold, maxima_prominence)
        self.LineFits.fit()
        self.LineFits.display_dataframe()
        
        if plot_fits == True:
            self.LineFits.plot_fits_and_profiles()
