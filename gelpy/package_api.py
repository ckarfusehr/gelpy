from .image_handling import Image
from .line_profile_handling import LineProfiles
from .profile_fitting_models import GaussianFitModel, EmgFitModel
from .profile_fit_handling import LineFits
from .background_ransac_fit_models import PlaneFit2d

# Magic numbers and strings
DEFAULT_GAMMA = 0.1
DEFAULT_GAIN = 1
DEFAULT_INTENSITY_RANGE = (0.05, 0.95)
DEFAULT_IMG_HEIGHT_FACTOR = 0
DEFAULT_LABEL_ROTATION = 45
DEFAULT_SHOW_TYPE = "non_linear"
MODEL_2D_PLANE_FIT_NAME = "2d_plane_fit"
GAUSSIAN_FIT_NAME = "gaussian"
EMG_FIT_NAME = "emg"


class AgaroseGel:
    def __init__(self, path):
        self.path = path
        self.labels = None
        self.x_label_positions = None
        self.global_line_profile_width = None

    def setup_gel(self, labels, x_label_pos, gamma=DEFAULT_GAMMA, gain=DEFAULT_GAIN, 
                  intensity_range=DEFAULT_INTENSITY_RANGE, img_height_factor=DEFAULT_IMG_HEIGHT_FACTOR, 
                  label_rotation=DEFAULT_LABEL_ROTATION, save=False, 
                  show_type=DEFAULT_SHOW_TYPE, line_profile_width=None,
                    remove_bg=False, bg_model=MODEL_2D_PLANE_FIT_NAME, bg_model_input=None):
        """Has to be used as the first function after initiating the Gel"""
        
        self.init_image(labels, x_label_pos, gamma, gain, intensity_range, img_height_factor, label_rotation)
        if remove_bg == False:
            self.plot_and_adjust_gels(show_type, save)
            self.setup_line_profile(line_profile_width)
        elif remove_bg == True:
            self.remove_background(bg_model=bg_model, bg_model_input=bg_model_input)

    def show_adjusted_images(self, save=True, show_type="both"):
        self.plot_and_adjust_gels(show_type, save)

    def init_image(self, labels, x_label_pos, gamma, gain, intensity_range, img_height_factor, label_rotation):
        self.Image = Image(self.path, labels, x_label_pos, label_rotation,
                           img_height_factor=img_height_factor, gamma=gamma, gain=gain, intensity_range=intensity_range)
        self.labels = labels
        self.x_label_pos = x_label_pos 

    def plot_and_adjust_gels(self, show_type, save):
        self.x_label_positions = self.Image.plot_adjusted_gels(show_type, save)
        
    def setup_line_profile(self, line_profile_width):
        self.global_line_profile_width = LineProfiles.guess_line_profile_width(self.x_label_positions, self.Image.gel_image, line_profile_width)
        self.Image.color_line_profile_area(self.global_line_profile_width, color="r")
    
    def remove_background(self, bg_model, bg_model_input):
        self.init_background_model(bg_model, bg_model_input)
        self.apply_background_model()

    def init_background_model(self, model, model_input):
        if model == MODEL_2D_PLANE_FIT_NAME:
            self.background_model = PlaneFit2d(self.Image.gel_image, model_input)
        else:
            print("No valid model selected")
    
    def apply_background_model(self):
        self.background_model.extract_fit_data_from_image()
        self.background_model.fit_model_to_data()
        self.Image.gel_image = self.background_model.substract_background()  # this sets the original gel_image to the bg corrected image
        self.background_model.visualize_fit_data()

    def show_raw_gel(self):
        self.Image.show_raw_image()

    # The line profiles logic is separated into a different method
    def show_line_profiles(self, select_lanes="all", slice_line_profile_length=(0,-1),
                           fit=None, maxima_threshold=0.001, maxima_prominence=None, plot_fits=False,
                           normalization_type="area", save=False, save_name_overview="selected_line_profiles.svg",
                           save_name_fits="selected_fitted_profiles.svg", show_df=True, save_df=False,
                           df_save_name="selected_fitted_params.csv", show_cumulative_lane_plot=True):
        
        self.init_line_profiles(select_lanes, slice_line_profile_length, normalization_type,
                                save, save_name_overview, show_cumulative_lane_plot)
        self.apply_line_profiles(fit, maxima_threshold, maxima_prominence, plot_fits, save, save_name_fits, show_df, save_df, df_save_name)

    def init_line_profiles(self, select_lanes, slice_line_profile_length, normalization_type, save, save_name_overview, show_cumulative_lane_plot):
        self.LineProfiles = LineProfiles(self.Image.gel_image, self.labels, self.x_label_positions,
                                         select_lanes, slice_line_profile_length, normalization_type,
                                         save, save_name_overview)
        self.LineProfiles.set_line_profile_width(self.global_line_profile_width)
        self.LineProfiles.extract_line_profiles()
        self.LineProfiles.normalize_line_profiles()
        if show_cumulative_lane_plot:
            self.LineProfiles.plot_selected_line_profiles()

    def apply_line_profiles(self, fit, maxima_threshold, maxima_prominence, plot_fits, save, save_name_fits, show_df, save_df, df_save_name):
        if fit is None:
            return
        elif fit == GAUSSIAN_FIT_NAME:
            fit_model = GaussianFitModel
        elif fit == EMG_FIT_NAME:
            fit_model = EmgFitModel
        else:
            raise ValueError("Invalid fit type")
            return

        
        self.LineFits = LineFits(fit_model, self.LineProfiles.selected_line_profiles_normalized, self.LineProfiles.selected_labels,
                                 maxima_threshold, maxima_prominence, save, save_name_fits)
        self.LineFits.fit()
        self.LineFits.display_dataframe(show_df, save_df, df_save_name)
        
        if plot_fits:
            self.LineFits.plot_fits_and_profiles()

