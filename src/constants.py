from enum import Enum

BACKGROUND_CLASS = 0
BACKGROUND_RGB_COLOR = [0, 0, 0]

ARTERY_CLASS = 1
ARTERY_RGB_COLOR = [255, 0, 0]

VEIN_CLASS = 2
VEIN_RGB_COLOR = [0, 0, 255]

UNCERTAINTY_CLASS = 3
UNCERTAINTY_RGB_COLOR = [255, 255, 255]

PAD = 10


class TypeClassification(Enum):
    NN = 0
    ASP = 1
    EU = 2
    GT = 3