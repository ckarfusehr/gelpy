from .background_BackgroundHandler import BackgroundHandler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

class PlaneFit2d(BackgroundHandler):
    def __init__(self, image, model_input):
        """
        Initialize the PlaneFit2d class.

        Parameters
        ----------
        image : np.array
            Input image to process.
        model_input : tuple
            Tuple of tuples. Each inner tuple contains stripe position and height.

        Returns
        -------
        None.
        """
        super().__init__(image)
        self.inlier_mask = None
        self.reconstructed_bg = None
        self.y_len_img, self.x_len_img = self.image.shape
        self.model_input = model_input if model_input else ((10, 10), (self.y_len_img - 15, 10))  # Set heuristic default parameter if not provided

        # Set the stripe data during initialization
        self.top_stripe_data = self.get_stripe_data(*self.model_input[0])
        self.bottom_stripe_data = self.get_stripe_data(*self.model_input[1])
    
    def get_stripe_data(self, stripe_position, stripe_height):
        """
        Get the stripe data from the image.

        Parameters
        ----------
        stripe_position : int
            Starting position of the stripe.
        stripe_height : int
            Height of the stripe.

        Returns
        -------
        stripe : np.array
            Stripe data from the image.
        X : np.array
            Array of x coordinates.
        Y : np.array
            Array of y coordinates.
        """
        stripe = self.image[stripe_position:stripe_position + stripe_height, :]
        X, Y = np.meshgrid(np.arange(stripe.shape[1]), np.arange(stripe_position, stripe_position + stripe_height))
        return stripe, X, Y

    def extract_fit_data_from_image(self):
        """
        Extract the fit data from the image.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        top_stripe, X_top, Y_top = self.top_stripe_data
        bottom_stripe, X_bottom, Y_bottom = self.bottom_stripe_data

        self.fit_data = np.concatenate((top_stripe.ravel(), bottom_stripe.ravel()))
        self.fit_data_X = np.concatenate((X_top.ravel(), X_bottom.ravel()))
        self.fit_data_Y = np.concatenate((Y_top.ravel(), Y_bottom.ravel()))

    def compute_new_image(self, image, params, X, Y):
        """
        Compute the new image after removing the background.

        Parameters
        ----------
        image : np.array
            The original image.
        params : list
            The parameters for the background plane.
        X : np.array
            Array of x coordinates.
        Y : np.array
            Array of y coordinates.

        Returns
        -------
        new_image : np.array
            The new image after background removal.
        bg_plane : np.array
            The background plane.
        """
        bg_plane = params[0]*X + params[1]*Y + params[2]  # Reconstruct the background plane
        new_image = image - bg_plane
        new_image[new_image < 0] = 0  # clip negative values to zero
        return new_image, bg_plane

    def compute_overlay(self):
        """
        Compute the overlay for visualizing the stripes.

        Parameters
        ----------
        None.

        Returns
        -------
        overlay : np.array
            The overlay image.
        """
        # Create an RGBA image with the same size as the original image
        overlay = np.zeros((self.y_len_img, self.x_len_img, 4))

        # Set alpha (transparency) to 0.5 for the stripes, 0 elsewhere
        for stripe_position, stripe_height in self.model_input:
            overlay[stripe_position:stripe_position + stripe_height, :, 3] = 0.9

        # Set color of the stripes to red
        overlay[:, :, 0] = 1
        return overlay
    
    def plot_image_with_overlay(self, fig, gs, image, overlay, vmin, vmax):
        """
        Plot the image with overlay.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object to plot on.
        gs : matplotlib.gridspec.GridSpec
            GridSpec object for the figure.
        image : np.array
            The image to plot.
        overlay : np.array
            The overlay to apply on the image.
        vmin : float
            Minimum value for the colormap.
        vmax : float
            Maximum value for the colormap.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Axes object for the plot.
        """
        ax = fig.add_subplot(gs[0:1, :]) # gs[0:2, :] -> this takes all columns in first two rows
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
        ax.imshow(overlay)
        ax.set_title('Image with stripes used for background plane fitting')
        return ax


    def plot_profiles(self, fig, gs, row, col, original_slice, bg_slice, new_slice):
        """
        Plot the row-averaged profiles for original, background and new slices.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object to plot on.
        gs : matplotlib.gridspec.GridSpec
            GridSpec object for the figure.
        row : int
            Row index for the subplot.
        col : int
            Column index for the subplot.
        original_slice : np.array
            Slice of the original image.
        bg_slice : np.array
            Slice of the background image.
        new_slice : np.array
            Slice of the new image.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Axes object for the plot.
        """
        # Compute the row-averaged profiles
        original_profile = original_slice.mean(axis=1)
        bg_profile = bg_slice.mean(axis=1)
        new_profile = new_slice.mean(axis=1)

        # Plot the profiles
        ax = fig.add_subplot(gs[row, col]) # We modify this line to assign subplot to proper row and column
        ax.plot(original_profile, label='Original')
        ax.plot(bg_profile, label='Background')
        ax.plot(new_profile, label='New') 
        ax.legend()
        ax.set_title(f'Slice {col+1}') # We modify the slice index from col+1 instead of i+1
        return ax


    def calculate_and_plot_slices(self, slice_indices, bg_plane, new_image, gs, fig):
        """
        Calculate and plot the slices of the image.

        Parameters
        ----------
        slice_indices : np.array
            Indices of the slices.
        bg_plane : np.array
            The background plane.
        new_image : np.array
            The new image after background removal.
        gs : matplotlib.gridspec.GridSpec
            GridSpec object for the figure.
        fig : matplotlib.figure.Figure
            Figure object to plot on.

        Returns
        -------
        None.
        """
        for i in range(6): # Calculate only 6 slices
            start, end = slice_indices[i], slice_indices[i+1]
            original_slice = self.image[:, start:end]
            bg_slice = bg_plane[:, start:end]
            new_slice = new_image[:, start:end]

            row = 1 if i < 3 else 2  # Calculate row based on i
            self.plot_profiles(fig, gs, row, i % 3, original_slice, bg_slice, new_slice) # We modify the column calculation as i % 3 


    def visualize_fit_data(self):
        """
        Visualize the fit data.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        X, Y = np.meshgrid(np.arange(self.x_len_img), np.arange(self.y_len_img))
        new_image, bg_plane = self.compute_new_image(self.image, self.params, X, Y)

        # Define the slices
        slice_indices = np.linspace(0, self.x_len_img, 7).astype(int)  # We modify this line to return 7 points (for 6 slices)

        fig = plt.figure(figsize=(20, 16))  # Adjust size as needed
        gs = GridSpec(3, 3, figure=fig) # We modify the GridSpec to have 3 rows and 3 columns

        overlay = self.compute_overlay()

        # Determine the 2nd and 98th percentiles of the image data
        vmin, vmax = np.percentile(self.image, [5, 95])

        self.plot_image_with_overlay(fig, gs, self.image, overlay, vmin, vmax)

        self.calculate_and_plot_slices(slice_indices, bg_plane, new_image, gs, fig)

        plt.tight_layout()
        plt.show()

    def fit_model_to_data(self):
        """
        Fit the model to the data.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # Stack X and Y into the data array
        data = np.c_[self.fit_data_X, self.fit_data_Y]

        # Fit plane using RANSAC
        ransac = RANSACRegressor(base_estimator=LinearRegression(fit_intercept=True), random_state=0)
        ransac.fit(data, self.fit_data)

        # Extract the parameters of the fitted plane
        self.params = [ransac.estimator_.coef_[0], ransac.estimator_.coef_[1], ransac.estimator_.intercept_]
        self.inlier_mask = ransac.inlier_mask_
        
        X, Y = np.meshgrid(np.arange(self.x_len_img), np.arange(self.y_len_img))
        self.reconstructed_bg = self.params[0]*X + self.params[1]*Y + self.params[2]