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
        """
        Initialize Gel object with the given path of an image file.
        
        Parameters:
        path (str): Path to an image file.
        """
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
        """
        Initializes the gel instance by loading the image, adjusting its properties and optionally 
        removing its background.

        Args:
            labels (list, optional): A list of labels for the lanes in the gel image.
            x_label_pos (list, optional): A list of the x-axis positions of the labels.
            gamma (float, optional): A gamma correction factor for the image. Default is DEFAULT_GAMMA.
            gain (float, optional): A gain correction factor for the image. Default is DEFAULT_GAIN.
            intensity_range (tuple, optional): A tuple specifying the intensity range for the image. 
                Default is DEFAULT_INTENSITY_RANGE.
            img_height_factor (float, optional): A factor specifying the relative height of the image. 
                Default is DEFAULT_IMG_HEIGHT_FACTOR.
            label_rotation (int, optional): The rotation angle of the labels. Default is DEFAULT_LABEL_ROTATION.
            save (bool, optional): A flag indicating whether the processed image should be saved. Default is False.
            show_type (str, optional): The type of display for the image. Default is DEFAULT_SHOW_TYPE.
            line_profile_width (int, optional): The width of the line profile for the gel.
            remove_bg (bool, optional): A flag indicating whether the background should be removed from the image. Default is False.
            bg_model (str, optional): The name of the background model to use if the background is to be removed. 
                Default is MODEL_2D_PLANE_FIT_NAME.
            bg_model_input (optional): Additional parameters for the background model.

        Raises:
            ValueError: If an invalid fit type is provided.
        """
        
        self.init_image(labels, x_label_pos, gamma, gain, intensity_range, img_height_factor, label_rotation)

        self.plot_and_adjust_gels(show_type)
        self.setup_line_profile(line_profile_width)
        
        if remove_bg == True:
            self.remove_background(bg_model=bg_model, bg_model_input=bg_model_input)

    def show_adjusted_images(self, save_adjusted_gels=False, show_type="both"):
        """
        Displays the adjusted images.

        Args:
            save_adjusted_gels (bool, optional): If set to True, the adjusted images are saved. Default is False.
            show_type (str, optional): The type of display for the image. Default is 'both'.
        """
        self.plot_and_adjust_gels(show_type, save_adjusted_gels)

    def plot_and_adjust_gels(self, show_type, save_adjusted_gels=False):
        """
        Plots and adjusts the gel images.

        Args:
            show_type (str): The type of display for the image.
            save_adjusted_gels (bool): If set to True, the adjusted images are saved.
        """
        self.Image.plot_adjusted_gels(show_type, save_adjusted_gels)

    def init_image(self, labels, x_label_pos, gamma, gain, intensity_range, img_height_factor, label_rotation):
        """
        Initializes the gel image with the provided parameters.

        Args:
            labels (list): A list of labels for the lanes in the gel image.
            x_label_pos (list): A list of the x-axis positions of the labels.
            gamma (float): A gamma correction factor for the image.
            gain (float): A gain correction factor for the image.
            intensity_range (tuple): A tuple specifying the intensity range for the image.
            img_height_factor (float): A factor specifying the relative height of the image.
            label_rotation (int): The rotation angle of the labels.
        """
        self.Image = Image(self.image, self.file_name_without_ext, labels, x_label_pos, label_rotation,
                           img_height_factor=img_height_factor, gamma=gamma, gain=gain, intensity_range=intensity_range)
        self.labels = self.Image.labels
        self.x_label_pos = x_label_pos 
        
    def setup_line_profile(self, line_profile_width):
        """
        Sets up the line profile for the gel.

        Args:
            line_profile_width (int): The width of the line profile.
        """
        self.x_label_positions = self.Image.x_label_positions
        self.global_line_profile_width = LineProfiles.guess_line_profile_width(self.x_label_positions, self.Image.gel_image, line_profile_width)
        self.Image.color_line_profile_area(self.global_line_profile_width, color="darkred")
    
    def remove_background(self, bg_model, bg_model_input):
        """
        Removes the background from the gel image.

        Args:
            bg_model (str): The name of the background model to use.
            bg_model_input: Additional parameters for the background model.
        """
        self.init_background_model(bg_model, bg_model_input)
        self.apply_background_model()

    def init_background_model(self, model, model_input):
        """
        Initializes the background model for the gel.

        Args:
            model (str): The name of the background model to use.
            model_input: Additional parameters for the background model.
        """
        if model == MODEL_2D_PLANE_FIT_NAME:
            self.background_model = PlaneFit2d(self.Image.gel_image, model_input)
        else:
            print("No valid model selected")
    
    def apply_background_model(self):
        """
        Applies the previously initialized background model to the gel image.
        """
        self.background_model.extract_fit_data_from_image()
        self.background_model.fit_model_to_data()
        self.Image.gel_image = self.background_model.substract_background()  # this sets the original gel_image to the bg corrected image
        self.background_model.visualize_fit_data()

    def show_raw_gel(self):
        """
        Display the raw gel image.
        """
        self.Image.show_raw_image()

    def show_line_profiles(self, select_lanes="all", slice_line_profile_length=(0,-1),
                           fit=False, maxima_threshold=0.001, maxima_prominence=None, peak_width=1, sigma=5,
                           plot_fits=False, normalization_type="area", save_overview=False,
                           save_fits=False, show_df=True, save_df=False,
                           show_overview=True):
        """
        Displays the line profiles of the gel.

        Args:
            select_lanes (str or list, optional): A string or list specifying which lanes to select for displaying line profiles. 
                Default is 'all'.
            slice_line_profile_length (tuple, optional): A tuple specifying the length of the line profile to slice. 
                Default is (0,-1) which represents the entire length.
            fit (bool or str, optional): If set to True, a fitting method will be applied to the line profiles. If a string 
                specifying the fitting method is provided, that method will be used. Default is False.
            maxima_threshold (float, optional): A threshold for detecting maxima in the line profiles. Default is 0.001.
            maxima_prominence (float, optional): A prominence value for detecting maxima. If None, the method will try to guess 
                an appropriate value. Default is None.
            peak_width (int, optional): The width of the peaks in the line profile. Default is 1.
            sigma (float, optional): A parameter for the Gaussian fitting method. Default is 5.
            plot_fits (bool, optional): If set to True, the fits will be plotted. Default is False.
            normalization_type (str, optional): The type of normalization to apply to the line profiles. Default is 'area'.
            save_overview (bool, optional): If set to True, the overview of line profiles will be saved. Default is False.
            save_fits (bool, optional): If set to True, the fits will be saved. Default is False.
            show_df (bool, optional): If set to True, the dataframe will be displayed. Default is True.
            save_df (bool, optional): If set to True, the dataframe will be saved. Default is False.
            show_overview (bool, optional): If set to True, the overview of line profiles will be displayed. Default is True.

        Raises:
            ValueError: If an invalid fit type is provided.
        """
        self.init_line_profiles(select_lanes, slice_line_profile_length, normalization_type,
                                save_overview, show_overview)
        self.apply_line_profiles(fit, maxima_threshold, maxima_prominence, peak_width, sigma , plot_fits, save_fits, show_df, save_df)

    def init_line_profiles(self, select_lanes, slice_line_profile_length, normalization_type, save_overview, show_overview):
        """
        Initializes the line profiles for the gel.

        Args:
            select_lanes (str or list): A string or list specifying which lanes to select for displaying line profiles.
            slice_line_profile_length (tuple): A tuple specifying the length of the line profile to slice.
            normalization_type (str): The type of normalization to apply to the line profiles.
            save_overview (bool): If set to True, the overview of line profiles will be saved.
            show_overview (bool): If set to True, the overview of line profiles will be displayed.
        """
        self.LineProfiles = LineProfiles(self.Image.gel_image, self.labels, self.x_label_positions,
                                         select_lanes, slice_line_profile_length, normalization_type,
                                         save_overview)
        self.LineProfiles.set_line_profile_width(self.global_line_profile_width)
        self.LineProfiles.extract_line_profiles()
        self.LineProfiles.normalize_line_profiles()
        if show_overview:
            self.LineProfiles.plot_selected_line_profiles()

    def apply_line_profiles(self, fit, maxima_threshold, maxima_prominence, peak_width, sigma , plot_fits, save_fits, show_df, save_df):
        """
        Apply fitting models to the extracted line profiles and perform a variety of related tasks.

        Args:
            fit (bool or str): Specifies the type of fit to apply. "gaussian" for Gaussian, "emg" for EMG. 
                            True is treated as "emg", False means no fit will be applied.
            maxima_threshold (float): The absolute height that a peak must have to be considered by the fitting algorithm.
            maxima_prominence (float): A parameter controlling how prominant a peak must be to be considered by the fitting algorithm.
            peak_width (int): The width of peaks to consider in the fitting algorithm.
            sigma (float): Parameter for the fitting model.
            plot_fits (bool): Whether to plot the results of the fits.
            save_fits (bool): Whether to save the fit results.
            show_df (bool): Whether to show the data frame of fit results.
            save_df (bool): Whether to save the data frame of fit results.

        Raises:
            ValueError: If the fit type is not recognized.
        """
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
        """
        Saves the state of the current Gel object to a pickle file. 

        Args:
            name (str): The name to save the file under.
            compress (bool): Whether to compress the pickle file using gzip.

        """
        if compress:
            with gzip.open(f'{name}.pkl.gz', 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        else:
            with open(f'{name}.pkl', 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path):
        """
        Loads a previously saved Gel object from a pickle file. 

        Args:
            file_path (str): Path to the pickle file to load.

        Returns:
            Gel: The loaded Gel object.
        """
        _, ext = os.path.splitext(file_path)
        if ext == '.gz':
            with gzip.open(file_path, 'rb') as input:
                return pickle.load(input)
        else:
            with open(file_path, 'rb') as input:
                return pickle.load(input)
        
        

