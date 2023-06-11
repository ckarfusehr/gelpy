import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
import os

class Image:
    def __init__(self, path, labels):
        # Attributed
        self.path = path
        self.labels = labels
        
        # Dummy attributes
        self.gamma, self.gain, self.intensity_range = 0.1, 1, (0,6000)
        
        # Methods
        self.gel_image = self.read_gel_image()
        self.file_name_without_ext, self.collage_file_path = self.process_path_and_create_file_names_for_saving()
        self.lin_contrast_adjusted = self.adjust_img_contrast_linear()
        self.non_lin_contrast_adjusted = self.adjust_img_contrast_non_linear()
        
    
    @staticmethod
    def cm_to_inch(cm):
        return cm * 0.393701
    
    def show_raw_image(self):
        plt.imshow(self.gel_image, cmap='gray')

    def read_gel_image(self):
        gel_image = io.imread(self.path)
        return gel_image

    def process_path_and_create_file_names_for_saving(self):
        file_name_without_ext = os.path.splitext(os.path.basename(self.path))[0]
        base_path = os.path.dirname(self.path)
        collage_file_name = file_name_without_ext + '_collage.png'
        collage_file_path = os.path.join(base_path, collage_file_name)
        return file_name_without_ext, collage_file_path

    def adjust_img_contrast_non_linear(self):
        img_adjusted_non_linear = exposure.adjust_gamma(self.gel_image, gamma=self.gamma, gain=self.gain)
        return img_adjusted_non_linear

    def adjust_img_contrast_linear(self):
        if self.intensity_range:
            img_adjusted_linear = exposure.rescale_intensity(self.gel_image, in_range=self.intensity_range)
        else:
            img_adjusted_linear = self.gel_image
        return img_adjusted_linear

    def plot_adjusted_gels(self, save):
        fig_height = self.gel_image.shape[0] * self.img_height_factor
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.cm_to_inch(18), fig_height))
        fig.suptitle(self.file_name_without_ext, fontsize=14, y=1)

        ax1.imshow(self.non_lin_contrast_adjusted, cmap='gray')
        ax1.set_title('Non-linear Contrast Adjustment')

        ax2.imshow(self.lin_contrast_adjusted, cmap='gray')
        ax2.set_title('Linear Contrast Adjustment')

        if self.x_label_pos and self.labels:
            left_side, right_side, n = self.x_label_pos
            x_label_positions = np.linspace(left_side, right_side, n)
            ax1.set_xticks(x_label_positions)
            ax2.set_xticks(x_label_positions)
            ax1.set_xticklabels(self.labels, rotation=self.label_rotation)
            ax2.set_xticklabels(self.labels, rotation=self.label_rotation)
            ax1.xaxis.set_ticks_position('top')
            ax2.xaxis.set_ticks_position('top')
        else:
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])

        y_ticks = np.arange(0, self.gel_image.shape[0], 100)
        ax1.set_yticks(y_ticks)
        ax2.set_yticks(y_ticks)
        ax1.tick_params(axis='both', labelsize=8)
        ax2.tick_params(axis='both', labelsize=8)

        plt.show()
        if save:
            fig.savefig(self.collage_file_path, bbox_inches='tight')
        
        return x_label_positions