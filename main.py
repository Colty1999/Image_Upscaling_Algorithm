import numpy as np
import cv2
from cv2 import dnn_superres

# Read image
# image = cv2.imread('./input/test1.jpg', cv2.IMREAD_UNCHANGED)
image = cv2.imread("./upscaled/upscaled.png", cv2.IMREAD_UNCHANGED)


def upscale(image):
    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()
    # Read the desired model
    path = "EDSR_x3.pb"
    sr.readModel(path)
    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("edsr", 3)

    # Upscale the image
    return sr.upsample(image)


def crop(image):
    height, width, channels = image.shape
    crop_size = 50
    return (image[crop_size: height - crop_size, crop_size: width - crop_size])


def remove_background(image):
    final_images = []
    # kernel = np.array([[0, -1, 0],
    #                    [-1, 5, -1],
    #                    [0, -1, 0]])
    # image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)

    v = np.median(image)
    sigma = 0.5
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    print(lower, upper)

    canny = cv2.Canny(image, lower, upper)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Find contours; use proper return value with respect to OpenCV version
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Filter contours with sufficient areas; create binary mask from them
    cnts = [c for c in cnts if cv2.contourArea(c) > 10000]
    mask = np.zeros_like(canny)
    mask = cv2.drawContours(mask, np.array(
        cnts, dtype=object), -1, 255, cv2.FILLED)

    # Iterate all contours...
    for i, c in enumerate(cnts):

        # Get bounding rectangle of contour and min/max coordinates
        rect = cv2.boundingRect(c)
        (x1, y1) = rect[:2]
        x2 = x1 + rect[2]
        y2 = y1 + rect[3]

        # Get image section
        crop_image = image[y1:y2, x1:x2]

        # Get mask section and cut possible neighbouring contours
        crop_mask = mask[y1:y2, x1:x2].copy()
        cnts = cv2.findContours(
            crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        c = max(cnts, key=cv2.contourArea)
        crop_mask[:, :] = 0
        cv2.drawContours(crop_mask, [c], -1, 255, cv2.FILLED)

        # Create image with transparent background
        transparent = np.zeros((rect[3], rect[2], 4), np.uint8)
        transparent[:, :, :3] = cv2.bitwise_and(crop_image, crop_image,
                                                mask=crop_mask)
        transparent[:, :, 3] = crop_mask
        final_images.append(transparent)
    return (final_images)


def denoise(image):
    return (cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21))


def treshold_bg_removal(image):
    # threshold on white
    # Define lower and uppper limits
    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])

    # Create mask to only select black
    thresh = cv2.inRange(image, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    # apply mask to image
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


crop_image = False
upscale_image = False
denoise_image = False
remove_background_from_image = True

if crop_image:
    image = crop(image)
if upscale_image:
    image = upscale(image)
# if denoise_image:
#     image = denoise(image)
if remove_background_from_image:
    final_images = remove_background(image)

    for i in range(len(final_images)):
        cv2.imwrite("./outcome/transparent_" +
                    str(i) + ".png", final_images[i])

    # image = treshold_bg_removal(image)
    # cv2.imwrite("./outcome/transparent.png", image)
else:
    cv2.imwrite("./upscaled/upscaled.png", image)
