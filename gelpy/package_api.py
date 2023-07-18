from .image_handling import Image
from .line_profile_handling import LineProfiles
from .profile_fitting_models import GaussianFitModel, EmgFitModel
from .profile_fit_handling import LineFits
from .background_ransac_fit_models import PlaneFit2d
import seaborn as sns
import pickle
import gzip
import os

# Set figure style

sns.set_context(context="paper")

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


class Gel:
    def __init__(self, path):
        self.labels = None
        self.x_label_positions = None
        self.global_line_profile_width = None
        self.image = Image.open_image(path)
        self.file_name_without_ext = os.path.splitext(os.path.basename(path))[0]

    def setup_gel(self, labels=None, x_label_pos=None, gamma=DEFAULT_GAMMA, gain=DEFAULT_GAIN, 
                  intensity_range=DEFAULT_INTENSITY_RANGE, img_height_factor=DEFAULT_IMG_HEIGHT_FACTOR, 
                  label_rotation=DEFAULT_LABEL_ROTATION, save=False, 
                  show_type=DEFAULT_SHOW_TYPE, line_profile_width=None,
                    remove_bg=False, bg_model=MODEL_2D_PLANE_FIT_NAME, bg_model_input=None):
        """Has to be used as the first function after initiating the Gel"""
        
        self.init_image(labels, x_label_pos, gamma, gain, intensity_range, img_height_factor, label_rotation)

        self.plot_and_adjust_gels(show_type)
        self.setup_line_profile(line_profile_width)
        
        if remove_bg == True:
            self.remove_background(bg_model=bg_model, bg_model_input=bg_model_input)

    def show_adjusted_images(self, save_adjusted_gels=False, show_type="both"):
        self.plot_and_adjust_gels(show_type, save_adjusted_gels)

    def plot_and_adjust_gels(self, show_type, save_adjusted_gels=False):
        self.Image.plot_adjusted_gels(show_type, save_adjusted_gels)

    def init_image(self, labels, x_label_pos, gamma, gain, intensity_range, img_height_factor, label_rotation):
        self.Image = Image(self.image, self.file_name_without_ext, labels, x_label_pos, label_rotation,
                           img_height_factor=img_height_factor, gamma=gamma, gain=gain, intensity_range=intensity_range)
        self.labels = self.Image.labels
        self.x_label_pos = x_label_pos 
        
    def setup_line_profile(self, line_profile_width):
        self.x_label_positions = self.Image.x_label_positions
        self.global_line_profile_width = LineProfiles.guess_line_profile_width(self.x_label_positions, self.Image.gel_image, line_profile_width)
        self.Image.color_line_profile_area(self.global_line_profile_width, color="darkred")
    
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

    def show_line_profiles(self, select_lanes="all", slice_line_profile_length=(0,-1),
                           fit=False, maxima_threshold=0.001, maxima_prominence=None, peak_width=1, sigma=5,
                           plot_fits=False, normalization_type="area", save_overview=False,
                           save_fits=False, show_df=True, save_df=False,
                           show_overview=True):
        
        self.init_line_profiles(select_lanes, slice_line_profile_length, normalization_type,
                                save_overview, show_overview)
        self.apply_line_profiles(fit, maxima_threshold, maxima_prominence, peak_width, sigma , plot_fits, save_fits, show_df, save_df)

    def init_line_profiles(self, select_lanes, slice_line_profile_length, normalization_type, save_overview, show_overview):
        self.LineProfiles = LineProfiles(self.Image.gel_image, self.labels, self.x_label_positions,
                                         select_lanes, slice_line_profile_length, normalization_type,
                                         save_overview)
        self.LineProfiles.set_line_profile_width(self.global_line_profile_width)
        self.LineProfiles.extract_line_profiles()
        self.LineProfiles.normalize_line_profiles()
        if show_overview:
            self.LineProfiles.plot_selected_line_profiles()

    def apply_line_profiles(self, fit, maxima_threshold, maxima_prominence, peak_width, sigma , plot_fits, save_fits, show_df, save_df):
        if fit == False:
            return
        elif fit == GAUSSIAN_FIT_NAME:
            fit_model = GaussianFitModel
        elif fit == True or fit == EMG_FIT_NAME:
            fit_model = EmgFitModel
        else:
            raise ValueError("Invalid fit type")

        
        self.LineFits = LineFits(fit_model, self.LineProfiles.selected_line_profiles_normalized, self.LineProfiles.selected_labels,
                                 maxima_threshold, maxima_prominence, peak_width, sigma , save_fits)
        self.LineFits.fit()
        self.LineFits.display_dataframe(show_df)
        self.LineFits.check_if_save_dataframe(save_df)
        
        if plot_fits:
            self.LineFits.plot_fits_and_profiles()

    def save(self, name, compress=False):
        if compress:
            with gzip.open(f'{name}.pkl.gz', 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        else:
            with open(f'{name}.pkl', 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path):
        _, ext = os.path.splitext(file_path)
        if ext == '.gz':
            with gzip.open(file_path, 'rb') as input:
                return pickle.load(input)
        else:
            with open(file_path, 'rb') as input:
                return pickle.load(input)
        
        

