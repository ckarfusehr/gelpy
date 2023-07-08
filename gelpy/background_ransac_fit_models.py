from .background_BackgroundHandler import BackgroundHandler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression
from matplotlib.gridspec import GridSpec

class PlaneFit2d(BackgroundHandler):
    def __init__(self, image, model_input):
        super().__init__(image)
        self.inlier_mask = None
        self.reconstructed_bg = None
        self.y_len_img, self.x_len_img = self.image.shape
        self.model_input = model_input if model_input else ((10, 10), (self.y_len_img - 15, 10))  # Set heuristic default parameter if not provided

        # Set the stripe data during initialization
        self.top_stripe_data = self.get_stripe_data(*self.model_input[0])
        self.bottom_stripe_data = self.get_stripe_data(*self.model_input[1])
    
    def get_stripe_data(self, stripe_position, stripe_height):
        stripe = self.image[stripe_position:stripe_position + stripe_height, :]
        X, Y = np.meshgrid(np.arange(stripe.shape[1]), np.arange(stripe_position, stripe_position + stripe_height))
        return stripe, X, Y

    def extract_fit_data_from_image(self):
        top_stripe, X_top, Y_top = self.top_stripe_data
        bottom_stripe, X_bottom, Y_bottom = self.bottom_stripe_data

        self.fit_data = np.concatenate((top_stripe.ravel(), bottom_stripe.ravel()))
        self.fit_data_X = np.concatenate((X_top.ravel(), X_bottom.ravel()))
        self.fit_data_Y = np.concatenate((Y_top.ravel(), Y_bottom.ravel()))


    def compute_new_image(self, image, params, X, Y):
        bg_plane = params[0]*X + params[1]*Y + params[2]  # Reconstruct the background plane
        new_image = image - bg_plane
        new_image[new_image < 0] = 0  # clip negative values to zero
        return new_image, bg_plane

    def compute_overlay(self):
        # Create an RGBA image with the same size as the original image
        overlay = np.zeros((self.y_len_img, self.x_len_img, 4))

        # Set alpha (transparency) to 0.5 for the stripes, 0 elsewhere
        for stripe_position, stripe_height in self.model_input:
            overlay[stripe_position:stripe_position + stripe_height, :, 3] = 0.5

        # Set color of the stripes to red
        overlay[:, :, 0] = 1
        return overlay

    def plot_image_with_overlay(self, fig, gs, image, overlay, vmin, vmax):
        ax = fig.add_subplot(gs[0:2, 0:5])
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
        ax.imshow(overlay)
        ax.set_title('Original image with stripes used for fitting')
        return ax

    def plot_histogram(self, fig, gs):
        ax = fig.add_subplot(gs[0:2, 5:10])
        for color, stripe_data in zip(['red', 'blue'], [self.top_stripe_data[0], self.bottom_stripe_data[0]]):
            ax.hist(stripe_data.flatten(), bins=100, color=color, alpha=0.5, label=f'{color.capitalize()} Stripe')
        ax.set_title('Pixel Intensity Distribution')
        ax.legend()
        return ax

    def plot_profiles(self, fig, gs, row, i, original_slice, bg_slice, new_slice):
        # Compute the row-averaged profiles
        original_profile = original_slice.mean(axis=1)
        bg_profile = bg_slice.mean(axis=1)
        new_profile = new_slice.mean(axis=1)

        # Plot the profiles
        ax = fig.add_subplot(gs[row, i*2:i*2+2])
        ax.plot(original_profile, label='Original')
        ax.plot(bg_profile, label='Background')
        ax.plot(new_profile, label='New') 
        ax.legend()
        ax.set_title(f'Slice {i+1}')
        return ax

    def calculate_and_plot_slices(self, slice_indices, bg_plane, new_image, gs, fig):
        for i in range(10):
            start, end = slice_indices[i], slice_indices[i+1]
            original_slice = self.image[:, start:end]
            bg_slice = bg_plane[:, start:end]
            new_slice = new_image[:, start:end]
            
            row = 2 if i < 5 else 3  # Calculate row based on i
            self.plot_profiles(fig, gs, row, i % 5, original_slice, bg_slice, new_slice)


    def visualize_fit_data(self):
        X, Y = np.meshgrid(np.arange(self.x_len_img), np.arange(self.y_len_img))
        new_image, bg_plane = self.compute_new_image(self.image, self.params, X, Y)

        # Define the slices
        slice_indices = np.linspace(0, self.x_len_img, 11).astype(int)  # Returns 11 points, but we will use it as start/end indices for 10 slices

        fig = plt.figure(figsize=(20, 16))  # Adjust size as needed
        gs = GridSpec(4, 10, figure=fig)

        overlay = self.compute_overlay()

        # Determine the 2nd and 98th percentiles of the image data
        vmin, vmax = np.percentile(self.image, [2, 98])

        self.plot_image_with_overlay(fig, gs, self.image, overlay, vmin, vmax)
        self.plot_histogram(fig, gs)

        self.calculate_and_plot_slices(slice_indices, bg_plane, new_image, gs, fig)

        plt.tight_layout()
        plt.show()

    def fit_model_to_data(self):
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
             
        
## Old implementation, fitting only along the y-axis:

# class TopBottomPlaneFit:
#     def __init__(self, image):
#         self.image = image

#     def extract_fit_data_from_image(self, top_stripe_height, top_stripe_position, bottom_stripe_height, bottom_stripe_position):
#         top_stripe = self.image[top_stripe_position:top_stripe_position+top_stripe_height, :]
#         bottom_stripe = self.image[bottom_stripe_position:bottom_stripe_position+bottom_stripe_height, :]
        
#         # Compute median values along the x-axis for each row in the top and bottom stripes
#         top_stripe_medians = np.median(top_stripe, axis=1)
#         bottom_stripe_medians = np.median(bottom_stripe, axis=1)

#         # The corresponding Y coordinates are just the row indices
#         Y_top = np.arange(top_stripe_position, top_stripe_position+top_stripe_height)
#         Y_bottom = np.arange(bottom_stripe_position, bottom_stripe_position+bottom_stripe_height)

#         # Concatenate the median values and their Y coordinate arrays
#         self.fit_data = np.concatenate((top_stripe_medians, bottom_stripe_medians))
#         self.fit_data_Y = np.concatenate((Y_top, Y_bottom))


#         # Debugging: display original image with stripes overlayed in red
#         # Create an RGBA image with the same size as the original image
#         overlay = np.zeros((self.image.shape[0], self.image.shape[1], 4))

#         # Set alpha (transparency) to 0.5 for the stripes, 0 elsewhere
#         overlay[top_stripe_position:top_stripe_position+top_stripe_height, :, 3] = 0.5
#         overlay[bottom_stripe_position:bottom_stripe_position+bottom_stripe_height, :, 3] = 0.5

#         # Set color of the stripes to red
#         overlay[:, :, 0] = 1

#         # Determine the 2nd and 98th percentiles of the image data
#         vmin, vmax = np.percentile(self.image, [2, 98])

#         fig, ax = plt.subplots(1, 2, figsize=(15, 5)) # New line to create a subplot

#         ax[0].imshow(self.image, cmap='gray', vmin=vmin, vmax=vmax) # Changed from plt.imshow to ax[0].imshow
#         ax[0].imshow(overlay)
#         ax[0].set_title('Original image with stripes used for fitting')

#         # New lines to plot histograms
#         #ax[1].hist(self.image.flatten(), bins=100, color='gray', alpha=0.5, label='Entire Image')  # Entire image histogram
#         ax[1].hist(top_stripe.flatten(), bins=100, color='red', alpha=0.5, label='Top Stripe')  # Top stripe histogram
#         ax[1].hist(bottom_stripe.flatten(), bins=100, color='blue', alpha=0.5, label='Bottom Stripe')  # Bottom stripe histogram
#         ax[1].set_title('Pixel Intensity Distribution')
#         ax[1].legend()

#         plt.tight_layout()
#         plt.show()

#     def extract_background(self):
#         # Fit line using RANSAC
#         ransac = RANSACRegressor(base_estimator=LinearRegression(fit_intercept=True),
#                                  random_state=0)
#         ransac.fit(self.fit_data_Y.reshape(-1, 1), self.fit_data)

#         # Extract the parameters of the fitted line
#         self.params = [ransac.estimator_.coef_[0], ransac.estimator_.intercept_]
#         self.inlier_mask = ransac.inlier_mask_



#     def remove_background(self, show_new_image=False):
#         y_len, x_len = self.image.shape
#         Y = np.arange(y_len).reshape(-1, 1)
#         line_z = self.params[0]*Y + self.params[1]
#         line_z_broadcasted = np.broadcast_to(line_z, (y_len, x_len))
#         self.new_image = self.image - line_z_broadcasted

#         # clipping negative values to zero
#         self.new_image[self.new_image < 0] = 0

#         if show_new_image:
#                     # Determine the 2nd and 98th percentiles of the image data
#             vminnew_img, vmaxnew_img = np.percentile(self.new_image, [1, 99])
#             plt.imshow(self.new_image, cmap='gray', vmin=vminnew_img, vmax=vmaxnew_img)
#             plt.title('Image after background subtraction')
#             plt.show()

#         return self.new_image
    
#     def visualize_profiles(self):
#         y_len, x_len = self.image.shape
#         Y = np.arange(y_len).reshape(-1, 1)
#         bg_line = self.params[0]*Y + self.params[1]  # Reconstruct the background line
#         bg_line_broadcasted = np.broadcast_to(bg_line, (y_len, x_len))

#         # Compute the new image
#         new_image = self.image - bg_line_broadcasted
#         new_image[new_image < 0] = 0  # clip negative values to zero

#         # Define the slices
#         slice_indices = np.linspace(0, x_len, 11).astype(int)  # Returns 11 points, but we will use it as start/end indices for 10 slices

#         fig, axs = plt.subplots(2, 5, figsize=(20, 8))  # Adjust size as needed
#         axs = axs.flatten()  # So we can iterate over the axes in a single loop

#         for i in range(10):
#             start, end = slice_indices[i], slice_indices[i+1]
#             original_slice = self.image[:, start:end]
#             bg_slice = bg_line_broadcasted[:, start:end]
#             new_slice = new_image[:, start:end]  # New line

#             # Compute the row-averaged profiles
#             original_profile = original_slice.mean(axis=1)
#             bg_profile = bg_slice.mean(axis=1)
#             new_profile = new_slice.mean(axis=1)  # New line

#             # Plot the profiles
#             axs[i].plot(original_profile, label='Original')
#             axs[i].plot(bg_profile, label='Background')
#             axs[i].plot(new_profile, label='New')  # New line
#             axs[i].legend()
#             axs[i].set_title(f'Slice {i+1}')

#         plt.tight_layout()
#         plt.show()