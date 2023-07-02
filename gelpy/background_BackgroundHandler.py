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

    def substract_background(self):
        self.new_image = self.image - self.reconstructed_bg
        self.new_image[self.new_image < 0] = 0  # clipping negative values to zero
            
        return self.new_image
    
    @abstractmethod
    def visualize_fit_data(self):
        pass
