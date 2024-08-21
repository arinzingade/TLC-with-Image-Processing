import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def load_and_preprocess_image(img):
    cropped_img = crop_image(img, 0.10, 0.10, 0.05, 0.05)
    resized_img = cv2.resize(cropped_img, (256, 500))
    gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return resized_img, blurred_image

def crop_image(image, left_pct, right_pct, top_pct, bottom_pct):
    height, width = image.shape[:2]
    left = int(width * left_pct)
    right = int(width * right_pct)
    top = int(height * top_pct)
    bottom = int(height * bottom_pct)
    cropped_image = image[top:height-bottom, left:width-right]
    return cropped_image

def compute_gradients(blurred_image):
    gradient_x = cv2.Scharr(blurred_image, cv2.CV_64F, 1, 0)
    gradient_y = cv2.Scharr(blurred_image, cv2.CV_64F, 0, 1)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    return gradient_magnitude

def find_contours(gradient_magnitude, threshold, min_area_threshold):
    high_contrast_areas = gradient_magnitude > threshold
    contours, _ = cv2.findContours(high_contrast_areas.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            rectangles.append([x, y, x + w, y + h])
    return np.array(rectangles)

def non_max_suppression(rectangles, overlap_thresh):
    if len(rectangles) == 0:
        return []

    pick = []
    x1 = rectangles[:, 0]
    y1 = rectangles[:, 1]
    x2 = rectangles[:, 2]
    y2 = rectangles[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        for pos in range(0, last):
            j = idxs[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            overlap = float(w * h) / area[j]

            if overlap > overlap_thresh:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return rectangles[pick]

def draw_rectangles(image, rectangles, min_required_area, max_aspect_ratio):
    for (x, y, x2, y2) in rectangles:
        aspect_ratio = (x2 - x) / (y2 - y)
        area = (x2 - x) * (y2 - y)
        if aspect_ratio <= max_aspect_ratio and area > min_required_area:
            cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
            center_x = (x + x2) // 2
            center_y = (y + y2) // 2
            cv2.circle(image, (center_x, center_y), 1, (0, 0, 255), -1)
            y_normalized = 1 - center_y / 500
            draw_text(image, x, y, x2, y2, y_normalized)

def draw_text(image, x, y, x2, y2, y_normalized):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("arial.ttf", 15) 
    text = f'{y_normalized:.2f}'
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_x = x + (x2 - x - (text_bbox[2] - text_bbox[0])) // 2
    text_y = y - (text_bbox[3] - text_bbox[1]) - 15
    draw.text((text_x, text_y), text, font=font, fill=(255, 0, 0))
    image[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def RFValueCalc(img):
    resized_img, blurred_image = load_and_preprocess_image(img)
    gradient_magnitude = compute_gradients(blurred_image)
    initial_threshold = 50
    initial_min_area_threshold = 200
    min_required_rectangles = 7
    min_required_area = 250
    max_aspect_ratio = 3

    num_rectangles = 0
    while num_rectangles < min_required_rectangles:
        rectangles = find_contours(gradient_magnitude, initial_threshold, initial_min_area_threshold)
        rectangles = non_max_suppression(rectangles, 0.2)
        draw_rectangles(resized_img, rectangles, min_required_area, max_aspect_ratio)
        num_rectangles = len(rectangles)
        initial_min_area_threshold -= 100

    return resized_img

