from .background_BackgroundHandler import BackgroundHandler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression
from matplotlib.gridspec import GridSpec
from .utility_functions import cm_to_inch
import seaborn as sns

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
        self.model_input = model_input if model_input else ((20, 10), (self.y_len_img - 15, 10))  # Set heuristic default parameter if not provided

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
            The matplotlib figure to draw on.
        gs : matplotlib.gridspec.GridSpec
            The matplotlib gridspec to manage the layout of the figure.
        image : np.array
            The image data to plot.
        overlay : np.array
            The overlay image data.
        vmin, vmax : scalar
            The minimum and maximum values to use for the grayscale colormap normalization.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes on which the image and overlay are plotted.
        """
        ax = fig.add_subplot(gs[0:1, :])
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
        ax.imshow(overlay)
        ax.set_title('Red lines are used for background plane fitting')
        ax.set_ylabel("[px]")
        
        # Remove x labels and ticks
        ax.set_xticks([])
        ax.set_xlabel('')

        return ax

    def plot_profiles(self, fig, gs, row, col, original_slice, bg_slice, new_slice, shared_ax=None):
        """
        Plot profiles for the original, background, and corrected slices of the image.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The matplotlib figure to draw on.
        gs : matplotlib.gridspec.GridSpec
            The matplotlib gridspec to manage the layout of the figure.
        row, col : int
            The row and column indices in the gridspec to place the plot.
        original_slice, bg_slice, new_slice : np.array
            Slices of the original, background, and corrected images.
        shared_ax : matplotlib.axes.Axes, optional
            The axes to share between different subplots. If None, a new axes will be created.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes on which the profiles are plotted.
        """
        original_profile = original_slice.mean(axis=1)
        bg_profile = bg_slice.mean(axis=1)
        new_profile = new_slice.mean(axis=1)
        if shared_ax is None:
            ax = fig.add_subplot(gs[row, col])
        else:
            ax = fig.add_subplot(gs[row, col], sharex=shared_ax, sharey=shared_ax)
        sns.lineplot(data=original_profile, label='Original', ax=ax, linestyle="--", color="gray", alpha=0.8)
        sns.lineplot(data=bg_profile, label='Background', ax=ax, color="gray", alpha=0.8)
        sns.lineplot(data=new_profile, label='Corrected', ax=ax, color="green", alpha=0.8)
        
        # Calculate and set y-axis upper limit
        y_upper_limit = 6 * bg_profile.mean()
        ax.set_ylim(0,y_upper_limit)

        # Set title only for the middle plot in the second row
        if row == 1 and col == 1:
            ax.set_title('Visual background fit validation')
        else:
            ax.set_title('')

        # Remove x-ticks and labels for plots not in bottom row
        if row != 2:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel('[px]')
            
        # Remove y-ticks, y-labels, and y-axis for all plots
        ax.set_yticklabels([])
        ax.set_ylabel('')
        ax.yaxis.set_ticks([]) # This line removes y-axis ticks

        return ax


    def calculate_and_plot_slices(self, slice_indices, bg_plane, new_image, gs, fig):
        """
        Calculate and plot slices of the original, background, and corrected images.

        Parameters
        ----------
        slice_indices : list of int
            List of indices where the image is sliced.
        bg_plane : np.array
            The 2D background plane.
        new_image : np.array
            The new image after background removal.
        gs : matplotlib.gridspec.GridSpec
            The matplotlib gridspec to manage the layout of the figure.
        fig : matplotlib.figure.Figure
            The matplotlib figure to draw on.

        Returns
        -------
        None
        """
        shared_ax = None
        for i in range(6):
            start, end = slice_indices[i], slice_indices[i+1]
            original_slice = self.image[:, start:end]
            bg_slice = bg_plane[:, start:end]
            new_slice = new_image[:, start:end]
            row = 1 if i < 3 else 2
            shared_ax = self.plot_profiles(fig, gs, row, i % 3, original_slice, bg_slice, new_slice, shared_ax)
            if i == 0:  # add legend only to the first plot
                shared_ax.legend()
            else:
                shared_ax.get_legend().remove()

    def visualize_fit_data(self):
        """
        Visualize the fit data.

        Returns
        -------
        None
        """
        X, Y = np.meshgrid(np.arange(self.x_len_img), np.arange(self.y_len_img))
        new_image, bg_plane = self.compute_new_image(self.image, self.params, X, Y)
        slice_indices = np.linspace(0, self.x_len_img, 7).astype(int)
        fig = plt.figure(figsize=(cm_to_inch(18), cm_to_inch(16)))
        gs = GridSpec(3, 3, figure=fig)
        overlay = self.compute_overlay()
        vmin, vmax = np.percentile(self.image, [5, 95])
        self.plot_image_with_overlay(fig, gs, self.image, overlay, vmin, vmax)
        self.calculate_and_plot_slices(slice_indices, bg_plane, new_image, gs, fig)
        plt.tight_layout()
        plt.show()

    def fit_model_to_data(self):
        """
        Fit the model to the data.

        Returns
        -------
        None.
        """
        # Stack X and Y into the data array
        data = np.c_[self.fit_data_X, self.fit_data_Y]

        # Fit plane using RANSAC
        ransac = RANSACRegressor(estimator=LinearRegression(fit_intercept=True), random_state=0)
        ransac.fit(data, self.fit_data)

        # Extract the parameters of the fitted plane
        self.params = [ransac.estimator_.coef_[0], ransac.estimator_.coef_[1], ransac.estimator_.intercept_]
        self.inlier_mask = ransac.inlier_mask_
        
        X, Y = np.meshgrid(np.arange(self.x_len_img), np.arange(self.y_len_img))
        self.reconstructed_bg = self.params[0]*X + self.params[1]*Y + self.params[2]