from .profile_fit_handling import LineFits
from scipy.signal import find_peaks
import numpy as np
import scipy.optimize as opt
import pandas as pd
from scipy.integrate import simps
from scipy.stats import norm
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.ndimage.filters import gaussian_filter1d

class FitModel(ABC):
    def __init__(self):
        """
        The base abstract class for creating a fitting model for line profiles.
        """
        self.fitted_peaks = []
        self.fit_df = None

    @abstractmethod
    def single_peak_function(self, x, *params):
        """
        An abstract method to implement a function for a single peak of the model.

        Args:
            x (numpy.ndarray): The x-values to calculate the function at.
            params: The parameters of the function.
        """
        pass
    
    @abstractmethod
    def find_single_peak_maxima(self, *params):
        """
        An abstract method to find the maximum point of a single peak of the model.

        Args:
            params: The parameters of the function.

        Returns:
            The x-value of the maximum point of the function.
        """
        pass

    def peak_area(self, *params, start, end):
        """
        Calculate the area under a single peak.

        Args:
            params: The parameters of the function.
            start (int): The start point for the area calculation.
            end (int): The end point for the area calculation.

        Returns:
            float: The calculated area under the peak.
        """
        x = np.arange(start, end)
        y = self.single_peak_function(x, *params)
        area = simps(y, x)
        return area

    def multi_peak_function(self, x, *params):
        """
        Compute the function values of a multi-peak model.

        Args:
            x (numpy.ndarray): The x-values to calculate the function at.
            params: The parameters of the function.

        Returns:
            numpy.ndarray: The calculated function values.
        """
        result = np.zeros_like(x)
        for i in range(0, len(params), self.params_per_peak()):
            result += self.single_peak_function(x, *params[i:i+self.params_per_peak()])
        return result

    @abstractmethod
    def params_per_peak(self):
        """
        An abstract method to return the number of parameters per peak in the model.
        """
        pass

    @abstractmethod
    def initial_guess(self, max_index, line_profile):
        """
        An abstract method to generate the initial guess for the parameters.

        Args:
            max_index (int): The index of a maximum in the line profile.
            line_profile (numpy.ndarray): The line profile.

        Returns:
            list: The initial guess for the parameters.
        """
        pass
    
    @staticmethod
    def find_maxima_of_line_profile(line_profile, maxima_threshold, prominence, peak_width, sigma):
        """
        Identify maxima in the provided line profile.

        Args:
            line_profile (np.array): The line profile data.
            maxima_threshold (float): The minimum height of a peak.
            prominence (float): The prominence value for peak detection.
            peak_width (float): The minimum width of a peak.
            sigma (float): The standard deviation for Gaussian kernel, used for smoothing.

        Returns:
            np.array: Indices of the detected peaks.
        """
        blurred_line_profile = gaussian_filter1d(line_profile, sigma)
        maxima_indices, _ = find_peaks(blurred_line_profile, height=maxima_threshold, prominence=prominence, width=peak_width)
        return maxima_indices

    def fit_single_profile(self, selected_lane_index, selected_normalized_line_profile, maxima_threshold, maxima_prominence, peak_width, sigma , selected_label):
        """
        Fit the model to a single line profile.

        Args:
            selected_lane_index (int): The index of the selected lane.
            selected_normalized_line_profile (np.array): The line profile data.
            maxima_threshold (float): The minimum height of a peak.
            maxima_prominence (float): The prominence value for peak detection.
            peak_width (float): The minimum width of a peak.
            sigma (float): The standard deviation for Gaussian kernel, used for smoothing.
            selected_label (str): The label for the line profile.

        Raises:
            RuntimeError: If the curve fitting fails.
        """
        maxima_indices = self.find_maxima_of_line_profile(selected_normalized_line_profile, maxima_threshold, maxima_prominence, peak_width, sigma )

        # Initial guess for the parameters
        initial_guess = [param for max_index in maxima_indices for param in self.initial_guess(max_index, selected_normalized_line_profile)]

        # Define the lower and upper bounds for the parameters
        lower_bounds = [0] * len(initial_guess)
        upper_bounds = [np.inf] * len(initial_guess)
        
        # Fix the mean values during fitting to the previously detected maxima +-1
        if self.fit_type == "gaussian":
            for i, max_index in enumerate(maxima_indices):
                lower_bounds[i*self.params_per_peak()+1] = max_index -1
                upper_bounds[i*self.params_per_peak()+1] = max_index +1
        bounds = (lower_bounds, upper_bounds)

        try:
            optimized_parameters, _ = opt.curve_fit(self.multi_peak_function, np.arange(len(selected_normalized_line_profile)), selected_normalized_line_profile, p0=initial_guess, bounds=bounds)
            self.fitted_peaks.append((selected_lane_index, selected_label, optimized_parameters))
        except RuntimeError:
            print(f"Failed to fit curve for line profile {selected_label}")

    def fit(self, normalized_line_profiles, maxima_threshold, maxima_prominence, peak_width, sigma , selected_labels):
        """
        Fit the model to multiple line profiles.

        Args:
            normalized_line_profiles (list of np.array): The list of line profile data.
            maxima_threshold (float): The minimum height of a peak.
            maxima_prominence (float): The prominence value for peak detection.
            peak_width (float): The minimum width of a peak.
            sigma (float): The standard deviation for Gaussian kernel, used for smoothing.
            selected_labels (list of str): The labels for the line profiles.
        """
        for selected_lane_index, selected_normalized_line_profile in enumerate(normalized_line_profiles):
            self.fit_single_profile(selected_lane_index, selected_normalized_line_profile, maxima_threshold, maxima_prominence, peak_width, sigma , selected_labels[selected_lane_index])


    def create_fit_dataframe(self, selected_normalized_line_profiles):
        """
        Create a DataFrame from the fitted peak data.

        Args:
            selected_normalized_line_profiles (list of np.array): The list of line profile data.

        Returns:
            pd.DataFrame: DataFrame of fitted peak data.
        """
        data = []

        for i, (selected_lane_index, label, params) in enumerate(self.fitted_peaks):
            total_area = 0
            temp_data = []

            for band_number, j in enumerate(range(0, len(params), self.params_per_peak())): 
                area = self.peak_area(*params[j:j+self.params_per_peak()], start=0, end=len(selected_normalized_line_profiles[i]))
                total_area += area
                maxima_position = self.find_single_peak_maxima(*params[j:j+self.params_per_peak()])
                peak_data = [selected_lane_index, label, band_number, area, maxima_position, *params[j:j+self.params_per_peak()]]
                temp_data.append(peak_data)
                
            for peak_data in temp_data:
                # Update area to be the relative area
                peak_data[3] /= total_area
                data.append(peak_data)

        # Create a DataFrame from the data
        columns = ["selected_lane_index", "label", "band_number", "relative_area", "maxima_position"] + self.param_labels()
        self.fit_df = pd.DataFrame(data, columns=columns)
        return self.fit_df


    @abstractmethod
    def param_labels(self):
        pass

class GaussianFitModel(FitModel):
    def __init__(self):
        """
        Class for fitting Gaussian models to line profiles.
        """
        super().__init__()
        self.fit_type = "gaussian"

    def single_peak_function(self, x, amplitude, mean, stddev):
        """
        The function of a single Gaussian peak.

        Args:
            x (numpy.ndarray): The x-values to calculate the function at.
            amplitude (float): The amplitude of the Gaussian peak.
            mean (float): The mean value of the Gaussian peak.
            stddev (float): The standard deviation of the Gaussian peak.

        Returns:
            numpy.ndarray: The calculated function values.
        """
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * (stddev ** 2)))

    def find_single_peak_maxima(self, amplitude, mean, stddev):
        """
        Find the maximum point of a single Gaussian peak.

        Args:
            amplitude (float): The amplitude of the Gaussian peak.
            mean (float): The mean value of the Gaussian peak.
            stddev (float): The standard deviation of the Gaussian peak.

        Returns:
            float: The x-value of the maximum point of the Gaussian peak.
        """
        return mean
    
    def params_per_peak(self):
        """
        Get the number of parameters used per peak in Gaussian model.

        Returns:
            int: The number of parameters per peak.
        """
        return 3  # amplitude, mean, stddev

    def initial_guess(self, max_index, line_profile):
        """
        Generate an initial guess for the parameters of a Gaussian peak.

        Args:
            max_index (int): The index of the maximum point in the peak.
            line_profile (np.array): The line profile data.

        Returns:
            list: Initial guesses for the amplitude, mean, and standard deviation.
        """
        return [line_profile[max_index], max_index, 1]  # amplitude, mean, stddev

    def param_labels(self):
        """
        Get labels for the parameters of a Gaussian peak.

        Returns:
            list: Labels for the amplitude, mean, and standard deviation.
        """
        return ["Amplitude", "Mean", "Standard Deviation"]

class EmgFitModel(FitModel):
    def __init__(self):
        """
        Class for fitting Exponentially Modified Gaussian (EMG) models to line profiles.
        """
        super().__init__()
        self.fit_type="emg"
    
        # standard emg
    # def single_peak_function(self, x, amplitude, mean, stddev, lambda_):
    #     term1 = (mean - x) / lambda_ + (stddev ** 2) / (2 * lambda_ ** 2)
    #     term2 = (x - mean) / stddev - stddev / lambda_
    #     return (amplitude / lambda_ * np.exp(term1) * norm.cdf(term2)).astype(np.float64)
    
    def single_peak_function(self, x, amplitude, mean, stddev, lambda_):
        """
        The function of a single EMG peak.

        Args:
            x (numpy.ndarray): The x-values to calculate the function at.
            amplitude (float): The amplitude of the EMG peak.
            mean (float): The mean value of the EMG peak.
            stddev (float): The standard deviation of the EMG peak.
            lambda_ (float): The rate parameter of the exponential part of the EMG peak.

        Returns:
            numpy.ndarray: The calculated function values.
        """
        term1 = (x - mean) / lambda_ + (stddev ** 2) / (2 * lambda_ ** 2)
        term2 = (mean - x) / stddev - stddev / lambda_
        return amplitude / lambda_ * np.exp(term1) * norm.cdf(term2)
    
    def find_single_peak_maxima(self, amplitude, mean, stddev, lambda_):
        """
        Find the maximum point of a single EMG peak.

        Args:
            amplitude (float): The amplitude of the EMG peak.
            mean (float): The mean value of the EMG peak.
            stddev (float): The standard deviation of the EMG peak.
            lambda_ (float): The rate parameter of the exponential part of the EMG peak.

        Returns:
            float: The x-value of the maximum point of the EMG peak.
        """
        def function_to_minimize(x):
            return -self.single_peak_function(x, amplitude, mean, stddev, lambda_)
        initial_guess = mean
        result = minimize(function_to_minimize, initial_guess)
        return result.x[0]

    def params_per_peak(self):
        """
        Get the number of parameters used per peak in EMG model.

        Returns:
            int: The number of parameters per peak.
        """
        return 4  # amplitude, mean, stddev, lambda_

    def initial_guess(self, max_index, line_profile):
        """
        Generate an initial guess for the parameters of an EMG peak.

        Args:
            max_index (int): The index of the maximum point in the peak.
            line_profile (np.array): The line profile data.

        Returns:
            list: Initial guesses for the amplitude, mean, standard deviation, and lambda.
        """
        return [line_profile[max_index], max_index, 1, 1]

    def param_labels(self):
        """
        Get labels for the parameters of an EMG peak.

        Returns:
            list: Labels for the amplitude, mean, standard deviation, and lambda.
        """
        return ["Amplitude", "Mean", "Standard Deviation", "Lambda"]