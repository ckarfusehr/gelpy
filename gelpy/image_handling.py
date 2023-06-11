import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
import os
import matplotlib.patches as patches

class Image:
    def __init__(self, path, labels, x_label_pos, label_rotation,
                 img_height_factor, gamma, gain, intensity_range):
        # Attributed
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
    
    @staticmethod
    def cm_to_inch(cm):
        return cm * 0.393701
    
    def show_raw_image(self):
        plt.imshow(self.gel_image, cmap='gray')

    def read_gel_image(self):
        gel_image = io.imread(self.path)
        self.gel_image = gel_image

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
                raise TypeError('Invalid input type in intensity_range')
        else:
            img_adjusted_linear = self.gel_image
        self.lin_contrast_adjusted = img_adjusted_linear
    
    def color_line_profile_area(self, line_profile_width, alpha=0.5, color = None):
        # The y position starts from 0 as we want the rectangle to span the entire height of the image
        y_pos = 0

        for x in self.x_label_positions:
            # Calculate the left lower corner position of the rectangle
            # We subtract half the width from the x_label_position to center the rectangle
            rect_x = x - line_profile_width/2
            rectangle_height = self.gel_image.shape[0]
            rectangle = patches.Rectangle((rect_x, y_pos), line_profile_width, rectangle_height,
                                          linewidth=1, edgecolor=color, facecolor=color, alpha=alpha)
            self.adjusted_gel_axes[0].add_patch(rectangle)

    def plot_adjusted_gels(self, show_type, save=False):
        fig_height = self.gel_image.shape[0] * self.img_height_factor

        if show_type == 'both':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.cm_to_inch(18), fig_height))
            axes = [ax1, ax2]
            images = [self.non_lin_contrast_adjusted, self.lin_contrast_adjusted]
            titles = ['Non-linear Contrast Adjustment', 'Linear Contrast Adjustment']
        elif show_type == 'non_linear':
            fig, ax1 = plt.subplots(figsize=(self.cm_to_inch(18), fig_height))
            axes = [ax1]
            images = [self.non_lin_contrast_adjusted]
            titles = ['Non-linear Contrast Adjustment']
        elif show_type == 'linear':
            fig, ax1 = plt.subplots(figsize=(self.cm_to_inch(18), fig_height))
            axes = [ax1]
            images = [self.lin_contrast_adjusted]
            titles = ['Linear Contrast Adjustment']
        else:
            raise ValueError("Invalid show_type. Expected one of: 'both', 'linear', 'non_linear'")

        fig.suptitle(self.file_name_without_ext, fontsize=14, y=1)

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)

            if self.x_label_pos and self.labels:
                left_side, right_side, n = self.x_label_pos
                x_label_positions = np.linspace(left_side, right_side, n)
                ax.set_xticks(x_label_positions)
                ax.set_xticklabels(self.labels, rotation=self.label_rotation)
                ax.xaxis.set_ticks_position('top')
            else:
                ax.set_xticks([])
                ax.set_yticks([])

            y_ticks = np.arange(0, self.gel_image.shape[0], 100)
            ax.set_yticks(y_ticks)
            ax.tick_params(axis='both', labelsize=8)
        
        if save:
            fig.savefig(self.collage_file_path, bbox_inches='tight')
        
        self.adjusted_gel_fig = fig
        self.adjusted_gel_axes = axes
        self.x_label_positions = x_label_positions
        
        return x_label_positions
    

        