from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import lstsq
from sklearn.linear_model import RANSACRegressor, LinearRegression

class BackgroundHandler(ABC):
    def __init__(self, image):
        self.image = image
        self.fit_data = None
        self.fit_data_X = None
        self.fit_data_Y = None
        self.params = None

    @abstractmethod
    def extract_fit_data_from_image(self):
        pass

    @abstractmethod
    def fit_model_to_data(self):
        pass

    def substract_background(self, show_new_image=False):
        self.new_image = self.image - self.reconstructed_bg
        self.new_image[self.new_image < 0] = 0  # clipping negative values to zero
        
        if show_new_image:
            vminnew_img, vmaxnew_img = np.percentile(self.new_image, [1, 99])             # Determine the 2nd and 98th percentiles of the image data
            plt.imshow(self.new_image, cmap='gray', vmin=vminnew_img, vmax=vmaxnew_img)
            plt.title('Image after background subtraction')
            plt.show()
            
        return
    
    @abstractmethod
    def visualize_fit_data(self):
        pass

    @abstractmethod
    def visualize_img_bgfit_newimg(self):
        pass
