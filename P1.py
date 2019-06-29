from scipy.stats import linregress
import numpy as np
import cv2
import os
import math
import socket
import sys
import json
import time

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def stack_images(img_a, img_b):
    return np.hstack((img_a, img_b))

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2] 
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    cv2.imshow("Masked", masked_image)
    return masked_image

def crop_roi(image, top_left, top_right, bottom_right, bottom_left):
    roi = [np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)]
    return region_of_interest(image, roi)

def crop_by_ref(img, ref_width, ref_height, ref_top_x, ref_top_y, ref_bot_x, ref_bot_y):
    width = img.shape[1]
    image_height = img.shape[0]
    middle_x = int(width / 2)
    image_offset_bottom_x = int(width * ref_bot_x / ref_width)
    image_offset_bottom_y = int(image_height * ref_bot_y / ref_height)
    image_offset_top_x = int(width * ref_top_x / ref_width)
    image_offset_top_y = int(image_height * ref_top_y / ref_height)
    top_left = [middle_x - image_offset_top_x, image_offset_top_y]
    top_right = [middle_x + image_offset_top_x, image_offset_top_y]
    bottom_right = [width - image_offset_bottom_x, image_offset_bottom_y]
    bottom_left = [image_offset_bottom_x, image_offset_bottom_y]

    return crop_roi(img, top_left, top_right, bottom_right, bottom_left)

def crop(image, bottom_offset = 0):
    ref_width = 640
    ref_height = 480
    ref_top_x = 100
    ref_top_y = 280
    ref_bottom_x = 0
    ref_bottom_y = 480 - bottom_offset
    return crop_by_ref(image, ref_width, ref_height, ref_top_x, ref_top_y, ref_bottom_x, ref_bottom_y)

def equalize_histogram(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def clache(img):
    cl = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8, 8))
    
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)
    y_clache = cl.apply(y)
    img_yuv = cv2.merge((y_clache, u, v))
    
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) 

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def biliteral(img):
    return cv2.bilateralFilter(img, 13, 75, 75)

def unsharp_mask(image, blured):
    return cv2.addWeighted(blured, 1.5, blured, -0.5, 0, image)


def edges(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = np.median(gray_img)
    sigma = 0.33
    lower = int(max(150, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return canny(gray_img, lower, upper)

def binary_hsv_mask(img, color_range):
    lower = np.array(color_range[0])
    upper = np.array(color_range[1])

    return cv2.inRange(img, lower, upper)

def binary_gray_mask(img, color_range):
    lower = np.array(color_range[0])
    upper = np.array(color_range[1])

    return cv2.inRange(img, color_range[0][0], color_range[1][0])

def binary_mask_apply(img, binary_mask):
    masked_image = np.zeros_like(img)
    
    for i in range(3): 
        masked_image[:,:,i] = binary_mask.copy()
        
    return masked_image

def binary_mask_apply_color(img, binary_mask):
    return cv2.bitwise_and(img, img, mask = binary_mask)

def filter_by_color_ranges(img, color_ranges):
    result = np.zeros_like(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    for color_range in color_ranges:
        color_bottom = color_range[0]
        color_top = color_range[1]
        
        if color_bottom[0] == color_bottom[1] == color_bottom[2] and color_top[0] == color_top[1] == color_top[2]:
            mask = binary_gray_mask(gray_img, color_range)
        else:
            mask = binary_hsv_mask(hsv_img, color_range)
            
        masked_img = binary_mask_apply(img, mask)
        result = cv2.addWeighted(masked_img, 1.0, result, 1.0, 0.0)

    return result

def color_threshold(img, white_value):
    white = [[white_value, white_value, white_value], [255, 255, 255]]
    yellow = [[80, 90, 90], [120, 255, 255]]
    
    return filter_by_color_ranges(img, [white, yellow])


def draw_lanes(height, width, left_x, right_x, intersect_pt):
    line_img = np.zeros((height, width, 3), dtype=np.uint8)

    if left_x != 0 and intersect_pt != None:
        cv2.line(line_img, (left_x, height), (intersect_pt[0], intersect_pt[1]), [255, 0, 0], 5)

    if right_x != 0 and intersect_pt != None:
        cv2.line(line_img, (intersect_pt[0], intersect_pt[1]), (right_x, height), [0, 255, 0], 5)
        
    return line_img

def hough_lines(image, white_value):
    if white_value < 150:
        return None

    image_masked = color_threshold(image, white_value)
    image_edges = edges(image_masked)
    houghed_lns = cv2.HoughLinesP(image_edges, 2, np.pi / 180, 50, np.array([]), 20, 100)
    
    if houghed_lns is None:
        return hough_lines(image, white_value - 5)
    
    return houghed_lns

def left_right_lines(lines):
    lines_all_left = []
    lines_all_right = []
    slopes_left = []
    slopes_right = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            
            if slope > 0:
                lines_all_right.append(line)
                slopes_right.append(slope)
            else:
                lines_all_left.append(line)
                slopes_left.append(slope)
                
    filtered_left_lns = filter_lines_outliers(lines_all_left, slopes_left, True)
    filtered_right_lns = filter_lines_outliers(lines_all_right, slopes_right, False)
    
    return filtered_left_lns, filtered_right_lns

def filter_lines_outliers(lines, slopes, is_left, min_slope = 0.5, max_slope = 0.9):
    if len(lines) < 2:
        return lines
    
    lines_no_outliers = []
    slopes_no_outliers = []
    
    for i, line in enumerate(lines):
        slope = slopes[i]
        
        if min_slope < abs(slope) < max_slope:
            lines_no_outliers.append(line)
            slopes_no_outliers.append(slope)

    slope_median = np.median(slopes_no_outliers)
    slope_std_deviation = np.std(slopes_no_outliers)
    filtered_lines = []
    
    for i, line in enumerate(lines_no_outliers):
        slope = slopes_no_outliers[i]
        intercepts = np.median(line)

        if slope_median - 2 * slope_std_deviation < slope < slope_median + 2 * slope_std_deviation:
            filtered_lines.append(line)

    return filtered_lines

def median(lines, prev_ms, prev_bs):
    if prev_ms is None:
        prev_ms = []
        prev_bs = []
    
    xs = []
    ys = []
    xs_med = []
    ys_med = []
    m = 0
    b = 0

    for line in lines:
        for x1, y1, x2, y2 in line:
            xs += [x1, x2]
            ys += [y1, y2]
    
    if len(xs) > 2 and len(ys) > 2:
        m, b = np.polyfit(xs, ys, 1)
        m, b, r_value_left, p_value_left, std_err = linregress(xs, ys)

        if len(prev_ms) > 0:
            prev_ms.append(m)
            prev_bs.append(b)
        else:
            return np.poly1d([m, b])
    
    if len(prev_ms) > 0:
        return np.poly1d([np.average(prev_ms), np.average(prev_bs)])
    else:
        return None

def intersect(f_a, f_b, bottom_y):
    if f_a is None or f_b is None:
        return None
    
    equation = f_a.coeffs - f_b.coeffs
    x = -equation[1] / equation[0]
    y = np.poly1d(f_a.coeffs)(x)
    x, y = map(int, [x, y])

    return [x, y]

def detect(image, white_value, prev_left_ms, prev_left_bs, prev_right_ms, prev_right_bs):
    houghed_lns = hough_lines(image, white_value)
    
    if houghed_lns is None:
        return [None, None, None, None]
    
    filtered_left_lns, filtered_right_lns = left_right_lines(houghed_lns)
    median_left_f = median(filtered_left_lns, prev_left_ms, prev_left_bs)
    median_right_f = median(filtered_right_lns, prev_right_ms, prev_right_bs)
    
    if median_left_f is None or median_right_f is None:
        return detect(image, white_value - 5, prev_left_ms, prev_left_bs, prev_right_ms, prev_right_bs)
    else:
        return [filtered_left_lns, filtered_right_lns, median_left_f, median_right_f]

def image_pipeline(image, prev_left_ms = [], prev_left_bs = [], prev_right_ms = [], 
                   prev_right_bs = [], bottom_offset = 0, should_plot = False):
    height = image.shape[0]
    width = image.shape[1]
    
    image_cropped = crop(image, bottom_offset)
    image_blured = biliteral(image_cropped)
    image_unsharp = unsharp_mask(image_cropped, image_blured)
    image_clached = clache(image_unsharp)
    
    left_lns, right_lns, median_left_f, median_right_f = detect(
        image_clached, 220, prev_left_ms, prev_left_bs, prev_right_ms, prev_right_bs)
    
    if median_left_f is None and median_right_f is None:
        return [image, None, None]
    
    intersect_pt = intersect(median_left_f, median_right_f, height)
    left_x = int((median_left_f - height).roots)
    right_x = int((median_right_f - height).roots)

    if intersect_pt is None:
        return [image, median_left_f, median_right_f, 320, 450]
    
    lanes = draw_lanes(height, width, left_x, right_x, intersect_pt)
    return [weighted_img(lanes, image), median_left_f, median_right_f, intersect_pt[0], intersect_pt[1]]

class Lane(object):

    def __init__(self, bottom = 0, buffer = 1):
        self.left_fs = []
        self.right_fs = []
        self.center_x = None
        self.center_y = None
        self.bottom_offset = bottom
        self.buffer_size = buffer
        print("Lane initialized", bottom, buffer)
        
    def process_frame(self, frame):
        left_ms = []
        right_ms = []
        left_bs = []
        right_bs = []
        
        for i in range(len(self.left_fs)):
            m, b = self.left_fs[-i].coeffs
            left_ms.append(m)
            left_bs.append(b)
        
        for i in range(len(self.right_fs)):
            m, b = self.right_fs[-i].coeffs
            right_ms.append(m)
            right_bs.append(b)
        
        result, left_f, right_f, self.center_x, self.center_y = image_pipeline(frame, left_ms, left_bs, right_ms, right_bs, should_plot = False, bottom_offset = self.bottom_offset)
   
        if left_f is not None:
            self.left_fs.append(left_f)
            
            while len(self.left_fs) > self.buffer_size:
                self.left_fs.pop(0)
                
        if right_f is not None:
            self.right_fs.append(right_f)
            
            while len(self.right_fs) > self.buffer_size:
                self.right_fs.pop(0)
        return result

def json_to_string(speed, angle):
    json_object = {'speed': speed,
    'angle': angle,
    }

    json_string = json.dumps(json_object)
    print(json_string)
    return json_string

def send_drive_command(speed, angle):
    message = json_to_string(speed, angle)
    arr = bytes(message, 'ascii')
    sock.sendall(arr)

port = 9999
ip = str(sys.argv[1])

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((ip , port))
print("Connected to ", ip, ":", port)

folder = "../theFxUITCar_Data/Snapshots"
filename = "fx_UIT_Car.png"
path = os.path.join(folder, filename)
frame = cv2.imread(path)
lane = Lane(bottom=20)

error_count = 0
while True:
    frame = cv2.imread(path)
    if not frame is None:
        image = lane.process_frame(frame)
        cv2.imshow("Car", image)
        angle = math.degrees(math.atan(abs(lane.center_x - image.shape[1] / 2)/(480- lane.center_y -20)))

        if lane.center_x < image.shape[1] / 2:
            angle = angle * (-1)
        speed = 45
       
        if abs(angle) > 3:
            speed = 4
        if abs(angle) > 9:
            speed = 2
        if abs(angle) > 10.5:
            speed = 0.5
        if abs(angle) > 13:
            speed = 0
            
        if (abs(angle) > 5.8) and abs(angle) < 6:
            error_count = error_count+1
        else:
            error_count = 0

        if error_count >= 25:
            error_count = 0
            send_drive_command(50, -45)
            time.sleep(1)
        else:
            send_drive_command(speed, angle)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

