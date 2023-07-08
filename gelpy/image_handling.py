import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
import os
import matplotlib.patches as patches
from .utility_functions import cm_to_inch

class Image:
    DEFAULT_COLOR = None

    def __init__(self, path, labels, x_label_pos, label_rotation,
                 img_height_factor, gamma, gain, intensity_range):
        # Attributes
        self.path = path
        self.labels = labels
        self.x_label_pos = x_label_pos
        self.gamma = gamma
        self.gain = gain
        self.intensity_range = intensity_range
        self.img_height_factor =  img_height_factor
        self.label_rotation = label_rotation
        
        # setup_classs:
        self.read_gel_image()
        self.process_path_and_create_file_names_for_saving()        
    
    def show_raw_image(self):
        plt.imshow(self.gel_image, cmap='gray')

    def read_gel_image(self):
        self.gel_image = io.imread(self.path)
        self.image_height, self.image_width  = self.gel_image.shape

    def process_path_and_create_file_names_for_saving(self):
        file_name_without_ext = os.path.splitext(os.path.basename(self.path))[0]
        base_path = os.path.dirname(self.path)
        collage_file_name = file_name_without_ext + '_collage.png'
        collage_file_path = os.path.join(base_path, collage_file_name)
        self.file_name_without_ext = file_name_without_ext
        self.collage_file_path = collage_file_path

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
    
    def color_line_profile_area(self, line_profile_width, color = DEFAULT_COLOR):
        y_pos = 0
        for x in self.x_label_positions:
            rect_x = x - line_profile_width/2
            rectangle_height = self.image_height
            rectangle = patches.Rectangle((rect_x, y_pos), line_profile_width, rectangle_height,
                                          linewidth=1, edgecolor=color, facecolor=(1,0,0,0.1))
            self.adjusted_gel_axes[0].add_patch(rectangle)

    def setup_figure(self, show_type):
        fig_height = self.image_height * self.img_height_factor

        if show_type == 'both':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(cm_to_inch(18), fig_height))
            axes = [ax1, ax2]
        elif show_type in ['non_linear', 'linear']:
            fig, ax1 = plt.subplots(figsize=(cm_to_inch(18), fig_height))
            axes = [ax1]
        else:
            raise ValueError(f"Invalid show_type. Expected one of: 'both', 'linear', 'non_linear' but got {show_type}")

        fig.suptitle(self.file_name_without_ext, fontsize=14, y=1)

        return fig, axes

    def configure_axis(self, ax, img, title):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        self.configure_ticks(ax)

    def configure_ticks(self, ax):
        if self.x_label_pos and self.labels:
            self.x_label_positions = self.compute_x_label_positions()
            ax.set_xticks(self.x_label_positions)
            ax.set_xticklabels(self.labels, rotation=self.label_rotation)
            ax.xaxis.set_ticks_position('top')
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        y_ticks = np.arange(0, self.image_height, 100)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='both', labelsize=8)

    def compute_x_label_positions(self):
        left_side, right_side, n = self.x_label_pos
        return np.linspace(left_side, right_side, n)

    def save_figure(self, fig, file_path):
        fig.savefig(file_path, bbox_inches='tight')
        
    def plot_adjusted_gels(self, show_type, save=False):
        images = {
            'non_linear': [self.non_lin_contrast_adjusted],
            'linear': [self.lin_contrast_adjusted],
            'both': [self.non_lin_contrast_adjusted, self.lin_contrast_adjusted]
        }
        titles = {
            'non_linear': ['Non-linear Contrast Adjustment'],
            'linear': ['Linear Contrast Adjustment'],
            'both': ['Non-linear Contrast Adjustment', 'Linear Contrast Adjustment']
        }
        fig, axes = self.setup_figure(show_type)

        for ax, img, title in zip(axes, images[show_type], titles[show_type]):
            self.configure_axis(ax, img, title)
        
        if save:
            self.save_figure(fig, self.collage_file_path)
        
        self.adjusted_gel_fig = fig
        self.adjusted_gel_axes = axes
        
        return self.x_label_positions
