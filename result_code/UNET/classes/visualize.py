import numpy as np
from matplotlib import pyplot as plt
from skimage import measure


class VisualizeUtils:

    @staticmethod
    def plot_learning_curve(results):
        plt.figure(figsize=(8, 8))
        plt.title("Learning curve")
        plt.plot(results.history["loss"], label="loss")
        plt.plot(results.history["val_loss"], label="val_loss")
        plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), 
                marker="x", color="r", label="best model")
        plt.xlabel("Epochs")
        plt.ylabel("log_loss")
        plt.legend()

    
    @staticmethod
    def plot_results_dataset(images, masks, predicted_results):
        for idx in range(len(images)):
            plt.figure(figsize=(8,15))
            plt.subplot(131)
            plt.title('Original image')
            plt.imshow(images[idx])
            plt.subplot(132)
            plt.title('Grund truth')
            plt.imshow(masks[idx], cmap='gray')
            plt.subplot(133)
            plt.title('Predicted image')
            plt.imshow(np.squeeze(predicted_results[idx]), cmap='gray')
            
            
    @staticmethod
    def plot_result_single_image(image, mask, predicted_result):
        plt.figure(figsize=(15,15))
        plt.subplot(131)
        plt.title('Original image')
        plt.imshow(image)
        plt.subplot(132)
        plt.title('Grund truth')
        plt.imshow(np.squeeze(mask), cmap='gray')
        plt.subplot(133)
        plt.title('Predicted image')
        plt.imshow(np.squeeze(predicted_result), cmap='gray')
        
        
    @staticmethod   
    def draw_contours_dataset(images, masks, filtered_images):
        for idx in range(len(images)):
            contours = measure.find_contours(filtered_images[idx], 0.8)
            
            plt.figure()
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(15,10))
            ax1.set_title('Image')
            ax1.imshow(images[idx], interpolation='nearest', cmap=plt.cm.gray)
            ax2.set_title('Grund truth')
            ax2.imshow(np.squeeze(masks[idx]), cmap='gray')
            ax3.set_title('Predicted_image')
            ax3.imshow(filtered_images[idx], cmap='gray')
            ax4.set_title('Predicted contour in image')
            ax4.imshow(images[idx], interpolation='nearest', cmap=plt.cm.gray)
            for n, contour in enumerate(contours):
                ax4.plot(contour[:, 1], contour[:, 0], linewidth=4, color='magenta')
            ax4.plot(contour[:, 1], contour[:, 0], linewidth=4, color='magenta')
   
    @staticmethod   
    def draw_contours_single_image(image, mask, filtered_image):
        contours = measure.find_contours(filtered_image, 0.8)

        plt.figure()
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(15,10))
        ax1.set_title('Image')
        ax1.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
        ax2.set_title('Grund truth')
        ax2.imshow(np.squeeze(mask), cmap='gray')
        ax3.set_title('Predicted_image')
        ax3.imshow(filtered_image, cmap='gray')
        ax4.set_title('Predicted contour in image')
        ax4.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
        for n, contour in enumerate(contours):
            ax4.plot(contour[:, 1], contour[:, 0], linewidth=4, color='magenta')
        ax4.plot(contour[:, 1], contour[:, 0], linewidth=4, color='magenta')