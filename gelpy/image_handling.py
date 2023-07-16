import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
import os
import matplotlib.patches as patches
from .utility_functions import cm_to_inch


class Image:
    DEFAULT_COLOR = None

    def __init__(self, image, file_name_withou_extension, labels, x_label_pos, label_rotation,
                 img_height_factor, gamma, gain, intensity_range):
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
        if self.labels is None:
            self.labels = [str(i + 1) for i in range(len(self.x_label_positions))]
        elif isinstance(self.labels, (list, range)):
            pass
        else:
            raise ValueError("labels must be None or a list")
    
    def show_raw_image(self):
        fig, ax = plt.subplots(1, figsize=(cm_to_inch(18), cm_to_inch(8)))
        ax.imshow(self.gel_image, cmap='gray')
        ax.set(title="Raw gel image", xlabel="[px]", ylabel="[px]")
        plt.show()

    @staticmethod
    def open_image(path):
        return io.imread(path)

    def adjust_img_contrast_non_linear(self):
        img_adjusted_non_linear = exposure.adjust_gamma(self.gel_image, gamma=self.gamma, gain=self.gain)
        self.non_lin_contrast_adjusted = img_adjusted_non_linear

    def adjust_img_contrast_linear(self):
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
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.set_ylabel("[px]")
        self.configure_ticks(ax)

    def configure_ticks(self, ax):
            ax.set_xticks(self.x_label_positions)
            ax.set_xticklabels(self.labels, rotation=self.label_rotation)
            ax.xaxis.set_ticks_position('top')

            y_ticks = np.arange(0, self.image_height, 150)
            ax.set_yticks(y_ticks)
            ax.tick_params(axis='both', labelsize=8)

    def compute_x_label_positions(self):
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
        if save_adjusted_gels == None or save_adjusted_gels == False:
            return
        elif save_adjusted_gels == True:
            fig.savefig(f"{self.file_name_without_extension}.png", bbox_inches='tight')
        elif isinstance(save_adjusted_gels, str):
            fig.savefig(save_adjusted_gels, bbox_inches='tight')
        else:
            raise ValueError("save_adjusted_gels must be a filename or True")