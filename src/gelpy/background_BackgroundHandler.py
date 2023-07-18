from abc import ABC, abstractmethod

class BackgroundHandler(ABC):
    def __init__(self, image):
        """
        Initializes the BackgroundHandler class with the image to handle.

        Args:
            image (np.ndarray): The image that the class will handle.
        """
        self.image = image
        self.fit_data = None
        self.fit_data_X = None
        self.fit_data_Y = None
        self.params = None

    @abstractmethod
    def extract_fit_data_from_image(self):
        """
        Extracts fit data from the image. 

        This function should be overridden in a child class based on the specific approach to 
        extracting fit data from the image.
        """
        pass

    @abstractmethod
    def fit_model_to_data(self):
        """
        Fits the model to the data extracted from the image. 

        This function should be overridden in a child class based on the specific approach 
        to fitting a model to the extracted fit data.
        """
        pass

    def substract_background(self):
        """
        Subtracts the background from the image, according to the model.

        The method uses the background model fitted by `fit_model_to_data` method 
        and subtracts it from the original image.

        Returns:
            np.ndarray: The image with the background subtracted.
        """
        self.new_image = self.image - self.reconstructed_bg
        self.new_image[self.new_image < 0] = 0  # clipping negative values to zero
            
        return self.new_image
    
    @abstractmethod
    def visualize_fit_data(self):
        """
        Visualizes the fit data. 

        This function should be overridden in a child class based on the specific approach 
        to visualizing the fit data.
        """
        pass
