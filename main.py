
from detection.DocScan import stack_img_generator
from RFCalc import RFValueCalc
import cv2

def main(image_path):
    img = cv2.imread(image_path)
    stacked_img, warped_img = stack_img_generator(img)
    cv2.imwrite("outputs/stacked.jpeg", stacked_img)

    if warped_img is None:
        rf_valued_img = RFValueCalc(img)
    else:
        rf_valued_img = RFValueCalc(warped_img)
    cv2.imwrite("outputs/rf_valued_img.jpeg", rf_valued_img)

if __name__ == "__main__":
    main("images/hello6.jpg")