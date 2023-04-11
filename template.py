import cv2
import numpy as np

cam = cv2.VideoCapture('input3.mp4')

left_xs = []
left_ys = []

right_xs = []
right_ys = []

left_top = 0, 0
left_bottom = 0, 0

right_top = 0, 0
right_bottom = 0, 0

while True:

    ret, frame = cam.read()
    # (w, h, c) = frame.shape
    # ret (bool): Return code of the `read` operation. Did we get an image or not?
    #             (if not maybe the camera is not detected/connected etc.)

    # frame (array): The actual frame as an array.
    #                Height x Width x 3 (3 colors, BGR) if color image.
    #                Height x Width if Grayscale
    #                Each element is 0-255.
    #                You can slice it, reassign elements to change pixels, etc.

    if ret is False:
        break

    # 2.	Shrink the frame!

    img = cv2.resize(frame, (390, 240))  # 380, 240

    height, width, c = img.shape

    cv2.imshow('Small', img)

    # 3.	Convert the frame to Grayscale! -> a)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Grayscale', gray)

    # 4.	Select only the road!

    # i.
    upper_right = (int(width * 0.56), int(height * 0.79))  # 0.55 prima data , apoi 75
    upper_left = (int(width * 0.44), int(height * 0.79))
    lower_left = (int(width * 0.0), int(height * 1.0))
    lower_right = (int(width * 1.0), int(height * 1.0))

    # ii.
    trapezoid_bounds = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)

    # iii.
    np_trapezoid = np.zeros((height, width), dtype=np.uint8)

    trapezoid = cv2.fillConvexPoly(np_trapezoid, trapezoid_bounds, 1)

    cv2.imshow('Trapezoid', trapezoid * 255)

    # b.
    road = gray * trapezoid

    cv2.imshow('Road', road)

    # 5.	Get a top-down view! (sometimes called a birds-eye view)

    # a.
    screen_bounds = np.array([(width, 0), (0, 0), (0, height), (width, height)], dtype=np.int32)

    trapezoid_bounds = np.float32(trapezoid_bounds)
    screen_bounds = np.float32(screen_bounds)

    # b.
    magic_matrix = cv2.getPerspectiveTransform(trapezoid_bounds, screen_bounds)

    # c.
    top_down = cv2.warpPerspective(road, magic_matrix, (width, height))

    cv2.imshow('Top-Down', top_down)

    # 6.	Add a bit of blur!

    # a.
    n = 7
    blur = cv2.blur(top_down, ksize=(n, n))

    cv2.imshow('Blur', blur)

    # 7.	Do edge detection!
    # a.
    sobel_horizontal = np.float32([[-1, 0, +1],
                                   [-2, 0, +2],
                                   [-1, 0, +1]])

    sobel_vertical = np.float32([[-1, -2, -1],
                                 [0, 0, 0],
                                 [+1, +2, +1]])

    # b.
    blur_ = np.float32(blur)
    blur__ = np.float32(blur)

    frame_sh = cv2.filter2D(blur_, -1, sobel_horizontal)
    frame_sv = cv2.filter2D(blur__, -1, sobel_vertical)

    frame_sh_ = cv2.convertScaleAbs(frame_sh)
    frame_sv_ = cv2.convertScaleAbs(frame_sv)

    cv2.imshow('Sobel-Horizontal', frame_sh_)
    cv2.imshow('Sobel-Vertical', frame_sv_)

    # c.
    geometric_mean = np.sqrt((frame_sh ** 2 + frame_sv ** 2))
    sobel_final = cv2.convertScaleAbs(geometric_mean)

    cv2.imshow('Sobel', sobel_final)

    # 8.	Binarize the frame!
    treshold = int(255 / 2)-50
    ret, binarized = cv2.threshold(sobel_final, treshold, 255, cv2.THRESH_BINARY)

    cv2.imshow('Binarized', binarized)

    # 9.	Get the coordinates of street markings on each side of the road!
    # a.
    binarized_copy = binarized.copy()

    binarized_copy[:, :int(width * 0.05)] = 0.0
    binarized_copy[:, int(width * 0.95):] = 0.0

    binarized_copy[int(height * 0.90):, :] = 0.0 #97

    # b.
    # binarized_copy[:, int(width / 2)] = width / 2

    array_1 = np.argwhere(binarized_copy[:, :int(width // 2)] > 1)
    array_2 = np.argwhere(binarized_copy[:, int(width // 2) + 1:] > 1)  # //

    left_xs = array_1[:, 1]
    left_ys = array_1[:, 0]

    right_xs = array_2[:, 1] + int(width // 2)
    right_ys = array_2[:, 0]

    # 10.	Find the lines that detect the edges of the lane!
    # a.

    if left_xs.any():
        left_line = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
    if right_xs.any():
        right_line = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

    # b.
    left_top_y = 0
    left_top_x = int((left_top_y - left_line[0]) / left_line[1])

    left_bottom_y = height
    left_bottom_x = int((left_bottom_y - left_line[0]) / left_line[1])

    right_top_y = 0
    right_top_x = int((right_top_y - right_line[0]) / right_line[1])

    right_bottom_y = height
    right_bottom_x = int((right_bottom_y - right_line[0]) / right_line[1])

    # c.
    if -10 ** 8 <= left_top_x <= 10 ** 8:
        left_top = int(left_top_x), int(left_top_y)

    if -10 ** 8 <= left_bottom_x <= 10 ** 8:
        left_bottom = int(left_bottom_x), int(left_bottom_y)

    if -10 ** 8 <= right_top_x <= 10 ** 8:
        right_top = int(right_top_x), int(right_top_y)

    if -10 ** 8 <= right_bottom_x <= 10 ** 8:
        right_bottom = int(right_bottom_x), int(right_bottom_y)

    # d.
    cv2.line(binarized_copy, left_top, left_bottom, (200, 0, 0), 3)
    cv2.line(binarized_copy, right_top, right_bottom, (100, 0, 0), 3)

    cv2.imshow('Binarized_copy', binarized_copy)

    # 11.   Create a final visualization!
    # a.
    blank_frame = np.zeros((width, height), dtype=np.uint8)

    # b.
    cv2.line(blank_frame, left_top, left_bottom, (255, 0, 0), 3)
    # cv2.line(blank_frame, right_top, right_bottom, (0, 255, 0), 3)

    # c.
    magic_matrix_left = cv2.getPerspectiveTransform(screen_bounds, trapezoid_bounds)

    # d.
    final1 = cv2.warpPerspective(blank_frame, magic_matrix_left, (width, height))

    # e.
    # array_left = np.argwhere(final1[:, :int(width // 2)] > 1)
    array_left = np.argwhere(final1 > 1)
    # array_2 = np.argwhere(final[:, int(width // 2) + 1:] > 1)  # //

    # f.
    # a-2.
    blank_frame_2 = np.zeros((width, height), dtype=np.uint8)

    # b-2.
    # cv2.line(blank_frame_2, left_top, left_bottom, (50, 50, 250), 3)
    cv2.line(blank_frame_2, right_top, right_bottom, (255, 0, 0), 3)

    cv2.imshow('linia', blank_frame_2)
    # c-2.
    magic_matrix_right = cv2.getPerspectiveTransform(screen_bounds, trapezoid_bounds)

    # d-2.
    final2 = cv2.warpPerspective(blank_frame_2, magic_matrix_right, (width, height))

    # e-2.
    # array_left = np.argwhere(final1[:, :int(width // 2)] > 1)
    # array_right = np.argwhere(final2[:, int(width // 2) + 1:] > 1)  # //
    array_right = np.argwhere(final2 > 1)
    # g.
    final = img.copy()
    # cv2.line(final, array_left[0], array_left[1], (50, 50, 250))
    # cv2.line(final, array_right[0], array_right[1], (50, 50, 250))

    for i in array_left:
        final[i[0]][i[1]][0] = 50
        final[i[0]][i[1]][1] = 50
        final[i[0]][i[1]][2] = 250

    for i in array_right:
        final[i[0]][i[1]][0] = 50
        final[i[0]][i[1]][1] = 250
        final[i[0]][i[1]][2] = 50

    cv2.imshow('Final', final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()

cv2.destroyAllWindows()
