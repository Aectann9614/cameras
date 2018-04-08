import cv2 as cv
import numpy as np

# Utils
def get_centroid(rect):
    """
    Return center of rectangle.
    :param rect: tuple(x, y, width, height).
    :return: tuple(x, y)
    """
    x, y, w, h = rect
    return (x + int(w / 2), y + int(h / 2))

def create_mask(size, polygon):
    mask = np.zeros(size, dtype=np.uint8)
    cv.fillPoly(mask, polygon, (255,))
    return mask

def check_masks(masks, point):
    """
    Check that point is located inside at least one mask from masks.
    :param masks: numpy array - mask.
    :param point: tuple(x, y)
    :return: boolean
    """
    for mask in masks:
        if mask[point[1]][point[0]] == 255:
            return True

    return False

# Adjusters frames
def resize(frame, size):
    """
    Resize image.
    :param frame: source image frame
    :param size: Size of new frame. Tuple - (width, height)
    :return: Reduced to size frame
    """
    return cv.resize(frame, size, 0, 0, cv.INTER_AREA)

def motion_detection(subtractor, learning_rate, frame):
    """
    Finding motion on the frame.
    :param subtractor: Opencv realisation of motion subtractor.
    :param learning_rate: Rate of learning subtractor to background model.
    :param frame: Color frame.
    :return: Mask of motion.
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    fgmask = subtractor.apply(gray, None, learning_rate)
    fgmask[fgmask < 240] = 0
    return fgmask

class MorphologyExecutor(object):
    """
    Executor of sequence morphology operations.
    """
    def __init__(self):
        self.__operations = []

    def add(self, operation):
        """
        Addition operation for execution
        :param operation: tuple(morphology`s code, kernel, iterations) of morphology params
        :return:
        """
        self.__operations.append(lambda frame: cv.morphologyEx(frame, operation[0], operation[1], operation[2]))

    def __call__(self, frame):
        filtering = frame
        for operation in self.__operations:
            filtering = operation(filtering)

        return filtering

# Frame processors
def find_contours(frame, filter, non_masks):
    """
    Find contours which correspond filter and isn`t located inside masks.
    :param frame: Frame
    :param filter: tuple(min area, min width, min height)
    :param non_masks:
    :return:
    """
    matches = []
    min_area, min_width, min_height = filter

    _, contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)

    for contour in contours:
        rect = cv.boundingRect(contour)
        center = get_centroid(rect)

        if (rect[2] < min_width) or (rect[3] < min_height) or check_masks(non_masks, center):
            continue

        matches.append(contour)

    return contour