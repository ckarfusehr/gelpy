from .image_handling import Image
from .line_profile_handling import LineProfiles
from .profile_fitting_models import GaussianFitModel, EmgFitModel
from .profile_fit_handling import LineFits

class AgaroseGel:
    def __init__(self, path):
        self.path = path
        
    def setup_gel(self, labels, x_label_pos, gamma=0.1, gain=1, intensity_range=(0.05, 0.95),
                             img_height_factor=0.005, label_rotation=45, save=False, show_type="non_linear",
                             line_profile_width=None):
        """Has to be used as the first function after initiating the Gel"""
        
        self.Image = Image(self.path, labels, x_label_pos, label_rotation,
                           img_height_factor=img_height_factor, gamma=gamma, gain=gain, intensity_range=intensity_range)
        self.Image.adjust_img_contrast_non_linear()
        self.Image.adjust_img_contrast_linear()
        self.x_label_positions = self.Image.plot_adjusted_gels(show_type, save)
        self.x_label_pos = x_label_pos # A workaround. Instead extrcat positions calculation from plotting function
        self.labels = labels
        
        # Utility functions for setting up gel well.
        self.global_line_profile_width = LineProfiles.guess_line_profile_width(self.x_label_positions, self.Image.gel_image, line_profile_width)
        return
    
    def show_adjusted_images(self, gamma=0.1, gain=1, intensity_range=(0.05, 0.95),
                             img_height_factor=0.009, label_rotation=45, save=False, show_type="both"):
        self.Image = Image(self.path, self.labels, self.x_label_pos, label_rotation, #extract the passing of x_label_pos here?
                           img_height_factor=img_height_factor, gamma=gamma, gain=gain, intensity_range=intensity_range)
        
        self.Image.adjust_img_contrast_non_linear()
        self.Image.adjust_img_contrast_linear()
        self.x_label_positions = self.Image.plot_adjusted_gels(show_type, save)
        return
    
    def show_raw_gel(self):
        self.Image.show_raw_image()
        return
    
    # set line_profile width to be what was picted before, but to still be able to be changed by the user.
    def show_line_profiles(self, select_lanes="all", slice_line_profile_length=(0,-1),
                           fit=None, maxima_threshold=0.001, maxima_prominence=None, plot_fits=False,
                           normalization_type="area", save=False, save_name_overview="selected_line_profiles.svg",
                           save_name_fits="selected_fitted_profiles.svg"):
        
        self.LineProfiles = LineProfiles(self.Image.gel_image, self.labels, self.x_label_positions,
                                         select_lanes, slice_line_profile_length, normalization_type,
                                         save, save_name_overview)
        self.LineProfiles.set_line_profile_width(self.global_line_profile_width)
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
                                    maxima_threshold, maxima_prominence, save, save_name_fits)
        self.LineFits.fit()
        self.LineFits.display_dataframe()
        
        if plot_fits == True:
            self.LineFits.plot_fits_and_profiles()
