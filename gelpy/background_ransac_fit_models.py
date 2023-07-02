from .background_BackgroundHandler import BackgroundHandler
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import lstsq
from sklearn.linear_model import RANSACRegressor, LinearRegression

class PlaneFit2d(BackgroundHandler):
    def __init__(self, image, model_input):
        super().__init__(image)
        self.inlier_mask = None
        self.reconstructed_bg = None
        if model_input != None:
            self.top_stripe_position, self.top_stripe_height = model_input[0]
            self.bottom_stripe_position, self.bottom_stripe_height = model_input[1]
        if model_input == None: # set default parameter
            self.top_stripe_position, self.top_stripe_height = 10, 10
            self.bottom_stripe_position, self.bottom_stripe_height =self.image.shape[0]-15, 10 #heuristic params

    def extract_fit_data_from_image(self):
        if not all([self.top_stripe_height, self.top_stripe_position, self.bottom_stripe_height, self.bottom_stripe_position]):
            raise ValueError("All parameters must be provided for PlaneFit2d")

        self.top_stripe = self.image[self.top_stripe_position:self.top_stripe_position+self.top_stripe_height, :]
        self.bottom_stripe = self.image[self.bottom_stripe_position:self.bottom_stripe_position+self.bottom_stripe_height, :]

        X_top, Y_top = np.meshgrid(np.arange(self.top_stripe.shape[1]), np.arange(self.top_stripe_position, self.top_stripe_position+self.top_stripe_height))
        X_bottom, Y_bottom = np.meshgrid(np.arange(self.bottom_stripe.shape[1]), np.arange(self.bottom_stripe_position, self.bottom_stripe_position+self.bottom_stripe_height))

        self.fit_data = np.concatenate((self.top_stripe.ravel(), self.bottom_stripe.ravel()))
        self.fit_data_X = np.concatenate((X_top.ravel(), X_bottom.ravel()))
        self.fit_data_Y = np.concatenate((Y_top.ravel(), Y_bottom.ravel()))


    def visualize_fit_data(self):
        if not all([self.top_stripe_height, self.top_stripe_position, self.bottom_stripe_height, self.bottom_stripe_position]):
            raise ValueError("All parameters must be provided for PlaneFit2d")
        
        # Create an RGBA image with the same size as the original image
        overlay = np.zeros((self.image.shape[0], self.image.shape[1], 4))

        # Set alpha (transparency) to 0.5 for the stripes, 0 elsewhere
        overlay[self.top_stripe_position:self.top_stripe_position+self.top_stripe_height, :, 3] = 0.5
        overlay[self.bottom_stripe_position:self.bottom_stripe_position+self.bottom_stripe_height, :, 3] = 0.5

        # Set color of the stripes to red
        overlay[:, :, 0] = 1

        # Determine the 2nd and 98th percentiles of the image data
        vmin, vmax = np.percentile(self.image, [2, 98])

        fig, ax = plt.subplots(1, 2, figsize=(15, 5)) # New line to create a subplot

        ax[0].imshow(self.image, cmap='gray', vmin=vmin, vmax=vmax) # Changed from plt.imshow to ax[0].imshow
        ax[0].imshow(overlay)
        ax[0].set_title('Original image with stripes used for fitting')

        # New lines to plot histograms
        #ax[1].hist(self.image.flatten(), bins=100, color='gray', alpha=0.5, label='Entire Image')  # Entire image histogram
        ax[1].hist(self.top_stripe.flatten(), bins=100, color='red', alpha=0.5, label='Top Stripe')  # Top stripe histogram
        ax[1].hist(self.bottom_stripe.flatten(), bins=100, color='blue', alpha=0.5, label='Bottom Stripe')  # Bottom stripe histogram
        ax[1].set_title('Pixel Intensity Distribution')
        ax[1].legend()

        plt.tight_layout()
        plt.show()

    def fit_model_to_data(self):
        # Stack X and Y into the data array
        data = np.c_[self.fit_data_X, self.fit_data_Y]

        # Fit plane using RANSAC
        ransac = RANSACRegressor(base_estimator=LinearRegression(fit_intercept=True),
                                 random_state=0)
        ransac.fit(data, self.fit_data)

        # Extract the parameters of the fitted plane
        self.params = [ransac.estimator_.coef_[0], ransac.estimator_.coef_[1], ransac.estimator_.intercept_]
        self.inlier_mask = ransac.inlier_mask_
        
        y_len, x_len = self.image.shape
        X, Y = np.meshgrid(np.arange(x_len), np.arange(y_len))
        self.reconstructed_bg = self.params[0]*X + self.params[1]*Y + self.params[2]

    def visualize_img_bgfit_newimg(self):
        
        y_len, x_len = self.image.shape
        X, Y = np.meshgrid(np.arange(x_len), np.arange(y_len))
        bg_plane = self.params[0]*X + self.params[1]*Y + self.params[2]  # Reconstruct the background plane

        # Compute the new image
        new_image = self.image - bg_plane
        new_image[new_image < 0] = 0  # clip negative values to zero

        # Define the slices
        slice_indices = np.linspace(0, x_len, 11).astype(int)  # Returns 11 points, but we will use it as start/end indices for 10 slices

        fig, axs = plt.subplots(2, 5, figsize=(20, 8))  # Adjust size as needed
        axs = axs.flatten()  # So we can iterate over the axes in a single loop

        for i in range(10):
            start, end = slice_indices[i], slice_indices[i+1]
            original_slice = self.image[:, start:end]
            bg_slice = bg_plane[:, start:end]
            new_slice = new_image[:, start:end]  # New line

            # Compute the row-averaged profiles
            original_profile = original_slice.mean(axis=1)
            bg_profile = bg_slice.mean(axis=1)
            new_profile = new_slice.mean(axis=1)  # New line

            # Plot the profiles
            axs[i].plot(original_profile, label='Original')
            axs[i].plot(bg_profile, label='Background')
            axs[i].plot(new_profile, label='New')  # New line
            axs[i].legend()
            axs[i].set_title(f'Slice {i+1}')

        plt.tight_layout()
        plt.show()
             
        
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