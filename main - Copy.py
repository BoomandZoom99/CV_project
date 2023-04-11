import cv2
import numpy as np


def grayscale(image):
    """Convert an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def canny(image, low_threshold, high_threshold):
    """Apply Canny edge detection to an image."""
    return cv2.Canny(image, low_threshold, high_threshold)


def gaussian_blur(image, kernel_size):
    """Apply Gaussian smoothing to an image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def region_of_interest(image, vertices):
    """Apply a region of interest mask to an image."""
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def draw_lines(image, lines, color=[255, 0, 0], thickness=5):
    """Draw lines on an image."""
    # Create arrays to hold the left and right lane line endpoints
    left_lane_points = []
    right_lane_points = []

    # Iterate over each line segment detected by Hough transform
    for line in lines:
        # Unpack the line segment endpoints
        x1, y1, x2, y2 = line[0]

        # Calculate the slope and intercept of the line segment
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Classify the line segment as belonging to the left or right lane line
        # based on its slope
        if slope < -0.5:
            left_lane_points.append((x1, y1))
            left_lane_points.append((x2, y2))
        elif slope > 0.5:
            right_lane_points.append((x1, y1))
            right_lane_points.append((x2, y2))

    # Fit a line to the left lane line points using linear regression
    if len(left_lane_points) > 0:
        left_x, left_y = zip(*left_lane_points)
        left_fit = np.polyfit(left_x, left_y, 1)
        left_lane_line = np.poly1d(left_fit)
        # Draw the left lane line
        cv2.line(image, (int((image.shape[0] - left_fit[1]) / left_fit[0]), image.shape[0]),
                 (int((320 - left_fit[1]) / left_fit[0]), 320), color, thickness)

    # Fit a line to the right lane line points using linear regression
    if len(right_lane_points) > 0:
        right_x, right_y = zip(*right_lane_points)
        right_fit = np.polyfit(right_x, right_y, 1)
        right_lane_line = np.poly1d(right_fit)
        # Draw the right lane line
        cv2.line(image, (int((540 - right_fit[1]) / right_fit[0]), 540),
                 (int((320 - right_fit[1]) / right_fit[0]), 320), color, thickness)

    return image


def process_image(image):
    """Process an image to detect and draw the lane lines."""
    # Convert the image to grayscale
    gray = grayscale(image)

    # Apply Gaussian smoothing to the grayscale image
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # Apply Canny edge detection to the blurred image
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    # Define the vertices of a trapezoid to be used as a mask
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (450, 320), (490, 320), (imshape[1], imshape[0])]], dtype=np.int32)

    # Apply the region of interest mask to the edge image
    masked_edges = region_of_interest(edges, vertices)
    cv2.imshow('Masked Edges', masked_edges)
    # Apply Hough transform to the masked edge image to detect lane lines
    rho = 2
    theta = np.pi / 180
    threshold = 15
    min_line_len = 40
    max_line_gap = 20
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    # Draw the detected lane lines on the original image
    line_image = np.zeros_like(image)
    line_image = draw_lines(line_image, lines)
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    return result


# Open the video file
cap = cv2.VideoCapture('input2.mp4')

# Process each frame of the video in real-time
while True:
    # Read a frame from the video file
    ret, frame = cap.read()

    # Exit the loop if the end of the video file is reached
    if not ret:
        break

    # Process the frame to detect and draw the lane lines
    result = process_image(frame)

    # Show the processed image in a separate window
    cv2.imshow('Lane Detection', result)

    # Wait for a key press and exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()

