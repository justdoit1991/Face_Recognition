# import module
import cv2

def img_resize(img, x, y, w, h):

    # extract face and resize
    y_min = int(y - (h * 0.05))
    y_max = int(y + h + (h * 0.05))
    x_min = int(x - (w * 0.05))
    x_max = int(x + w + (w * 0.05))
    extract_img = img[y_min:y_max, x_min:x_max, ::-1]  # (height, width, RGB dimension)
    
    # adjust image size
    if extract_img.size >= 160*160*3:
        infer_img = cv2.resize(extract_img, (160,160), interpolation = cv2.INTER_AREA)
    else:
        infer_img = cv2.resize(extract_img, (160,160), interpolation = cv2.INTER_CUBIC)

    return infer_img
