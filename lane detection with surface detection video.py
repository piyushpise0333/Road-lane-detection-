import cv2 as cv
import numpy as np

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygon = np.array([
        [(int(width*0.0), height), (int(width*1.0), height),(width//2, int(height * 0.53))]
    ])
    mask = np.zeros_like(image)
    cv.fillPoly(mask, polygon, 255)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):

    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    result = cv.addWeighted(image, 0.9, line_image, 1, 1)
    return result


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordiantes(image, left_fit_average)
    right_line = make_coordiantes(image, right_fit_average)
    return np.array([left_line, right_line])


def make_coordiantes(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(0.64))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def plot_surface(image, co_ordinates):
    height = image.shape[0]
    width = image.shape[1]
    masked_image = np.zeros_like(image)
    x1, y1, x2, y2 = co_ordinates[0].reshape(4)
    x3, y3, x4, y4 = co_ordinates[1].reshape(4)
    polygon = np.array([
    [(x1,y1),(x2,y2),(x4,y4),(x3,y3)]
    ])
    mask = np.zeros_like(image)
    cv.fillPoly(mask, polygon, (255, 255, 0))
    masked_image = cv.bitwise_and(image, mask)
    result = cv.addWeighted(image, 0.9, masked_image, 1, 1)
    return result

cap = cv.VideoCapture('video5.mp4')

while True:
    
    ret , frame = cap.read()
    if not ret:
        continue
    img1 = frame.copy()
    gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 50, 150)
    region = region_of_interest(canny)
    lines = cv.HoughLinesP(region, 2, np.pi/180, 100,np.array([]), minLineLength=40, maxLineGap=5)
    drawline = display_lines(frame, lines)
    averaged_lines = average_slope_intercept(img1, lines)
    drawline = display_lines(frame, averaged_lines)
    surface_to_drive_on = plot_surface(drawline, averaged_lines)
    cv.imshow('lane', surface_to_drive_on)
    cv.waitKey(1)
