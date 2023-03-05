import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def kmeans(X: np.ndarray, k: int, centroids=None, max_iter=30, tolerance=1e-2):
    """This functions implements the Kmeans or Kmeans++ algorithm on a given dataset
    comparable to Scikit-Learn's"""
    N = X.shape[0]
    if centroids == 'kmeans++':

        # compute kmeans ++
        # pick first centroid randomly
        centroids = [X[np.random.choice(N, 1, replace=False), :]]
        for centroid_idx in range(1, k):
            distances = np.zeros((N, centroid_idx))
            for i in range(len(centroids)):
                # find closest centroid to x
                distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
            min_distance = np.min(distances, axis=1)
            next_centroid = X[np.argmax(min_distance), :]
            centroids.append(next_centroid)
    else:
        # pick k random initilization of centriods
        centroids = X[np.random.choice(N, k, replace=False), :]
    distances = np.zeros((N, k))
    for _ in range(max_iter):
        for i in range(len(centroids)):
            # find closest centroid to x
            distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
        labels = np.argmin(distances, axis=1)
        prev_centroids = centroids.copy()
        for i in range(len(centroids)):
            centroids[i] = np.mean(X[labels == i], axis=0)
        if np.array_equal(centroids, prev_centroids):
            break
    return centroids, labels


def kmeans_image_compression(filename, k, color=True):
    """Takes a filepath to an image and compress the image to k color 
    then displayes the original image and compressed image"""
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if color is True:
        pixel_vals = image.reshape((-1, 3))
    else:
        pixel_vals = image.reshape((-1, 1))
    pixel_vals = np.float32(pixel_vals)
    centers, labels = kmeans(
        pixel_vals, k=k, centroids='kmeans++', max_iter=30)
    centers = np.stack(centers, axis=0)
    centers = np.uint8(centers)
    segmented_data = centers[labels]
    segmented_image = segmented_data.reshape((image.shape))

    f = plt.figure(figsize=(14, 14))
    f.add_subplot(1, 2, 1)
    plt.imshow(image)
    f.add_subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.show(block=True)


def display_2_images(image1, image2):
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)
    f = plt.figure(figsize=(14, 14))
    f.add_subplot(1, 2, 1)
    plt.axis('off')
    fig = plt.imshow(image1)
    f.add_subplot(1, 2, 2)
    plt.axis('off')
    fig = plt.imshow(image2)
    plt.show(block=True)


def display_image(image):
    image = cv2.imread(image)
    f = plt.figure(figsize=(14, 14))
    f.add_subplot(1, 2, 1)
    fig = plt.imshow(image)
    plt.axis('off')
