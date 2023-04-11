import cv2
import numpy as np

def process_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale', gray)

    # Apply Gaussian blur to the image
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    cv2.imshow('Blur', blur_gray)

    # Apply Canny edge detection to the image
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    cv2.imshow('Edges', edges)

    # Create a mask for the region of interest
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (450, 320), (490, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    cv2.imshow('Masked Edges', masked_edges)

    # Apply Hough transform to detect lines in the image
    rho = 2
    theta = np.pi / 180
    threshold = 40
    min_line_length = 100
    max_line_gap = 50
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    # Draw the detected lines on a blank image
    line_image = np.zeros_like(image)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    cv2.imshow('Lines', line_image)

    # Combine the lines image with the original image
    combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    cv2.imshow('Combined', combined_image)

    # Wait for a key press and then close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Open the sample video file
cap = cv2.VideoCapture('input3.mp4')

# Loop through the frames in the video
while cap.isOpened():
    # Read the next frame
    ret, frame = cap.read()
    if ret:
        # Process the image
        process_image(frame)
    else:
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
