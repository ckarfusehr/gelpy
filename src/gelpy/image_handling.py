import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
import matplotlib.patches as patches
from .utility_functions import cm_to_inch


class Image:
    """ 
    Image manipulation and analysis class. It provides utilities for opening, displaying, 
    adjusting and plotting images.
    """
    DEFAULT_COLOR = None

    def __init__(self, image, file_name_withou_extension, labels, x_label_pos, label_rotation,
                 img_height_factor, gamma, gain, intensity_range):
        """
        Initialize the Image object.
        
        Parameters:
        image (np.ndarray): The image to manipulate.
        file_name_withou_extension (str): Name of the file without extension.
        labels (list): Labels to use for the image. If None, dummy labels will be used.
        x_label_pos (tuple): Tuple containing left_side, right_side, and number of points n for label positions. If None, positions will be guessed.
        label_rotation (int): The rotation angle for the labels in degrees.
        img_height_factor (float): Factor to apply to image height for displaying.
        gamma (float): Gamma correction factor.
        gain (float): Gain to apply after gamma correction.
        intensity_range (tuple): Intensity range for linear contrast adjustment. If None, no adjustment will be made.
        """
        # Attributes
        self.gel_image = image
        self.image_height, self.image_width  = self.gel_image.shape
        self.file_name_without_extension = file_name_withou_extension
        self.labels = labels
        self.x_label_pos = x_label_pos
        self.gamma = gamma
        self.gain = gain
        self.intensity_range = intensity_range
        self.img_height_factor =  img_height_factor
        self.label_rotation = label_rotation
        
        # setup_classs:
        self.compute_x_label_positions()
        self.create_dummy_labels_if_needed()
        self.adjust_img_contrast_non_linear()
        self.adjust_img_contrast_linear()    
    
    def create_dummy_labels_if_needed(self):
        """
        Creates dummy labels if no labels are provided during initialization. 
        The dummy labels will be a series of integers starting from 1.
        Raises a ValueError if labels is neither None nor a list.
        """
        if self.labels is None:
            self.labels = [str(i + 1) for i in range(len(self.x_label_positions))]
        elif isinstance(self.labels, (list, range)):
            pass
        else:
            raise ValueError("labels must be None or a list")
    
    def show_raw_image(self):
        """
        Display the raw gel image using matplotlib. The figure size is set to (18cm, 8cm).
        The image is displayed in grayscale. The title of the figure is 'Raw gel image', 
        x-label is '[px]', and y-label is '[px]'.
        """
        fig, ax = plt.subplots(1, figsize=(cm_to_inch(18), cm_to_inch(8)))
        ax.imshow(self.gel_image, cmap='gray')
        ax.set(title="Raw gel image", xlabel="[px]", ylabel="[px]")
        plt.show()

    @staticmethod
    def open_image(path):
        """
        Open an image from a given file path.

        Parameters:
        path (str): The path to the image file.

        Returns:
        numpy.ndarray: The image as a numpy array.
        """
        return io.imread(path)

    def adjust_img_contrast_non_linear(self):
        """
        Adjusts the contrast of the image using gamma correction.
        The gamma and gain values provided during initialization are used for the adjustment.
        The result is stored in the attribute 'non_lin_contrast_adjusted'.
        """
        img_adjusted_non_linear = exposure.adjust_gamma(self.gel_image, gamma=self.gamma, gain=self.gain)
        self.non_lin_contrast_adjusted = img_adjusted_non_linear

    def adjust_img_contrast_linear(self):
        """
        Adjusts the contrast of the image linearly based on the intensity range provided during initialization. 
        If no intensity range is provided, no adjustment is made and the original image is retained.
        The result is stored in the attribute 'lin_contrast_adjusted'.
        Raises a TypeError if the values in the intensity range are neither integers nor floats.
        """
        if self.intensity_range:
            if all(isinstance(i, int) for i in self.intensity_range):
                img_adjusted_linear = exposure.rescale_intensity(self.gel_image, in_range=self.intensity_range)
            elif all(isinstance(i, float) for i in self.intensity_range):
                q_low, q_high = np.percentile(self.gel_image, [self.intensity_range[0]*100, self.intensity_range[1]*100])
                img_adjusted_linear = exposure.rescale_intensity(self.gel_image, in_range=(q_low, q_high))
            else:
                raise TypeError(f'Invalid input type in intensity_range, expected ints or floats but got {type(self.intensity_range)}')
        else:
            img_adjusted_linear = self.gel_image
        self.lin_contrast_adjusted = img_adjusted_linear
    

    def setup_figure(self, show_type):
        """
        Sets up a figure for displaying the image.

        Parameters:
        show_type (str): Specifies the type of display. Options are 'both', 'linear', 'non_linear'. 
                         'both' will create a 2x1 subplot, otherwise it will create a single plot.

        Returns:
        tuple: A tuple containing the figure and the axes for the plots.
        Raises a ValueError if an invalid show_type is provided.
        """
        SINGLE_GEL_HEIGHT = 12 #cm
        TWO_GELS_HEIGHT = 24 #cm
        if show_type == 'both':
            fig_height = cm_to_inch(TWO_GELS_HEIGHT) + self.image_width/self.image_height * cm_to_inch(self.img_height_factor)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(cm_to_inch(18), fig_height))
            axes = [ax1, ax2]
        elif show_type in ['non_linear', 'linear']:
            fig_height = cm_to_inch(SINGLE_GEL_HEIGHT) + self.image_width/self.image_height * cm_to_inch(self.img_height_factor)
            fig, ax1 = plt.subplots(figsize=(cm_to_inch(18), fig_height))
            axes = [ax1]
        else:
            raise ValueError(f"Invalid show_type. Expected one of: 'both', 'linear', 'non_linear' but got {show_type}")

        fig.suptitle(self.file_name_without_extension, fontsize=14, y=1)

        return fig, axes
    
    def configure_axis(self, ax, img, title):
        """
        Configures the provided axis for image display. 

        Parameters:
        ax (matplotlib.axes.Axes): The axis to be configured.
        img (numpy.ndarray): The image to display.
        title (str): The title of the plot.
        """
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.set_ylabel("[px]")
        self.configure_ticks(ax)

    def configure_ticks(self, ax):
        """
        Configures the x and y ticks for the provided axis. 

        Parameters:
        ax (matplotlib.axes.Axes): The axis to be configured.
        """
        ax.set_xticks(self.x_label_positions)
        ax.set_xticklabels(self.labels, rotation=self.label_rotation)
        ax.xaxis.set_ticks_position('top')

        y_ticks = np.arange(0, self.image_height, 150)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='both', labelsize=8)

    def compute_x_label_positions(self):
        """
        Computes the x positions for labels based on 'x_label_pos' provided during initialization.
        If 'x_label_pos' is None, the method will guess the positions. If it's a tuple with three 
        values, these values will be used to compute a linspace. 
        Raises a ValueError if 'x_label_pos' is neither None nor a tuple with three values.
        """
        if self.x_label_pos is None:
            left_guess, right_guess, n = 0.05*self.image_width, 0.95*self.image_width, 10
            self.x_label_positions = np.linspace(left_guess, right_guess, n)

        elif isinstance(self.x_label_pos, tuple) and len(self.x_label_pos) == 3:
            left_side, right_side, n = self.x_label_pos
            self.x_label_positions = np.linspace(left_side, right_side, n)

        else:
            raise ValueError("x_label_pos must be None or a tuple with three values")

        return


    def color_line_profile_area(self, line_profile_width, color = DEFAULT_COLOR):
        """
        Colors the line profile area of the image. 

        Parameters:
        line_profile_width (int): The width of the line profile area.
        color (str): The color to use for the line profile area. If None, the default color will be used.
        """
        y_pos = 0
        previous_end_x = 0

        def draw_rectangle(start_x, end_x):
            rectangle = patches.Rectangle((start_x, y_pos), end_x - start_x, self.image_height,
                                        linewidth=0.5, edgecolor=color, facecolor=(0, 0, 0, 0.3))
            self.adjusted_gel_axes[0].add_patch(rectangle)

        positions = sorted(self.x_label_positions) + [self.image_width + line_profile_width/2]

        for x in positions: 
            rect_x = x - line_profile_width / 2
            if rect_x > previous_end_x:
                # Draw a rectangle in the gap between the previous rectangle and the current one
                draw_rectangle(previous_end_x, rect_x)

            previous_end_x = rect_x + line_profile_width

        return


        
    def plot_adjusted_gels(self, show_type, save_adjusted_gels):
        """
        Plots the adjusted gel images. 

        Parameters:
        show_type (str): Specifies the type of plot. Options are 'both', 'linear', 'non_linear'. 
                         'both' will plot both non-linear and linear adjusted images.
        save_adjusted_gels (str or bool): Specifies whether and where to save the figure.
                                          If True, the figure will be saved with the file name provided during initialization.
                                          If a string, it will be used as the file name for saving the figure.
        Raises a ValueError if an invalid 'show_type' or 'save_adjusted_gels' is provided.
        """
        images = {
            'non_linear': [self.non_lin_contrast_adjusted],
            'linear': [self.lin_contrast_adjusted],
            'both': [self.non_lin_contrast_adjusted, self.lin_contrast_adjusted]
        }
        titles = {
            'non_linear': ['Non-linear contrast adjusted'],
            'linear': ['Linear contrast adjusted'],
            'both': ['Non-linear contrast adjusted', 'Linear contrast adjusted']
        }
        fig, axes = self.setup_figure(show_type)

        for ax, img, title in zip(axes, images[show_type], titles[show_type]):
            self.configure_axis(ax, img, title)
        
        self.adjusted_gel_fig = fig
        self.adjusted_gel_axes = axes
        
        self.check_if_save_figure(fig, save_adjusted_gels) 
        
        return


    def check_if_save_figure(self, fig, save_adjusted_gels):
        """
        Checks whether to save a figure and saves it if required. 

        Parameters:
        fig (matplotlib.figure.Figure): The figure to be saved.
        save_adjusted_gels (str or bool): Specifies whether and where to save the figure.
                                          If True, the figure will be saved with the file name provided during initialization.
                                          If a string, it will be used as the file name for saving the figure.
        Raises a ValueError if an invalid 'save_adjusted_gels' is provided.
        """
        if save_adjusted_gels == None or save_adjusted_gels == False:
            return
        elif save_adjusted_gels == True:
            fig.savefig(f"{self.file_name_without_extension}.png", bbox_inches='tight')
        elif isinstance(save_adjusted_gels, str):
            fig.savefig(save_adjusted_gels, bbox_inches='tight')
        else:
            raise ValueError("save_adjusted_gels must be a filename or True")