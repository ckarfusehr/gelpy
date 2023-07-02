from .profile_fit_handling import LineFits

from scipy.signal import find_peaks
import numpy as np
import scipy.optimize as opt
import pandas as pd
from scipy.integrate import simps
from scipy.stats import norm
from abc import ABC, abstractmethod
from scipy.optimize import minimize

class FitModel(ABC):
    def __init__(self):
        self.fitted_peaks = []
        self.fit_df = None

    @abstractmethod
    def single_peak_function(self, x, *params):
        pass
    
    @abstractmethod
    def find_single_peak_maxima(self, *params):
        pass

    @abstractmethod
    def peak_area(self, *params, start, end):
        pass

    def multi_peak_function(self, x, *params):
        x = x.astype(np.float64)
        result = np.zeros_like(x)
        for i in range(0, len(params), self.params_per_peak()):
            result += self.single_peak_function(x, *params[i:i+self.params_per_peak()])
        return result

    @abstractmethod
    def params_per_peak(self):
        pass

    @abstractmethod
    def initial_guess(self, max_index, line_profile):
        pass
    
    @staticmethod
    def find_maxima_of_line_profile(line_profile, maxima_threshold, prominence, window_length=4):
        maxima_indices, _ = find_peaks(line_profile, height=maxima_threshold, prominence=prominence)
        return maxima_indices

    def fit(self, normalized_line_profiles, maxima_threshold, maxima_prominence, selected_labels):
        for selected_lane_index, selected_normalized_line_profile in enumerate(normalized_line_profiles):
            maxima_indices = self.find_maxima_of_line_profile(selected_normalized_line_profile, maxima_threshold, maxima_prominence)

            # Initial guess for the parameters
            initial_guess = []
            for max_index in maxima_indices:
                initial_guess.extend(self.initial_guess(max_index, selected_normalized_line_profile))

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
                self.fitted_peaks.append((selected_lane_index, selected_labels[selected_lane_index], optimized_parameters))
            except RuntimeError:
                print(f"Failed to fit curve for line profile {selected_labels[selected_lane_index]}")

    def create_fit_dataframe(self, selected_normalized_line_profiles):
        data = []
        for i, (selected_lane_index, label, params) in enumerate(self.fitted_peaks):
            total_area = 0
            # Calculate the area under each peak
            start = 0
            end = len(selected_normalized_line_profiles[i])
            for j in range(0, len(params), self.params_per_peak()):  
                area = self.peak_area(*params[j:j+self.params_per_peak()], start, end)
                total_area += area

            # Calculate parameter of each peak and create a new row for each peak
            for band_number, j in enumerate(range(0, len(params), self.params_per_peak())): 
                area = self.peak_area(*params[j:j+self.params_per_peak()], start, end)
                relative_area = area / total_area
                maxima_position = self.find_single_peak_maxima(*params[j:j+self.params_per_peak()])
                peak_data = [selected_lane_index, label, band_number, relative_area, maxima_position, *params[j:j+self.params_per_peak()]]
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
        super().__init__()
        self.fit_type = "gaussian"

    def single_peak_function(self, x, amplitude, mean, stddev):
        return (amplitude * np.exp(-((x - mean) ** 2) / (2 * (stddev ** 2)))).astype(np.float64)

    def peak_area(self, amplitude, mean, stddev, start, end):
        x = np.arange(start, end)
        y = self.single_peak_function(x, amplitude, mean, stddev)
        area = simps(y, x)
        return area
    
    def find_single_peak_maxima(self, amplitude, mean, stddev):
        return mean
    
    def params_per_peak(self):
        return 3  # amplitude, mean, stddev

    def initial_guess(self, max_index, line_profile):
        return [line_profile[max_index], max_index, 1]  # amplitude, mean, stddev

    def param_labels(self):
        return ["Amplitude", "Mean", "Standard Deviation"]


class EmgFitModel(FitModel):
    def __init__(self):
        super().__init__()
        self.fit_type="emg"
    
    def single_peak_function(self, x, amplitude, mean, stddev, lambda_):
        term1 = (x - mean) / lambda_ + (stddev ** 2) / (2 * lambda_ ** 2)
        term2 = (mean - x) / stddev - stddev / lambda_
        return (amplitude / lambda_ * np.exp(term1) * norm.cdf(term2)).astype(np.float64)

    def peak_area(self, amplitude, mean, stddev, lambda_, start, end):
        x = np.linspace(start, end, 1000)
        y = self.single_peak_function(x, amplitude, mean, stddev, lambda_)
        area = simps(y, x)
        return area
    
    def find_single_peak_maxima(self, amplitude, mean, stddev, lambda_):
        # Define a function to minimize (negative of single_peak_function)
        def function_to_minimize(x):
            return -self.single_peak_function(x, amplitude, mean, stddev, lambda_)
        # Initial guess for the maximum is the mean
        initial_guess = mean
        # Use scipy's minimize function to find the maximum
        result = minimize(function_to_minimize, initial_guess)
        # Return the x value of the maximum
        return result.x[0]

    def params_per_peak(self):
        return 4  # amplitude, mean, stddev, lambda_

    def initial_guess(self, max_index, line_profile):
        return [line_profile[max_index], max_index, 1, 1]

    def param_labels(self):
        return ["Amplitude", "Mean", "Standard Deviation", "Lambda"]
