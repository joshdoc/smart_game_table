####################################################################################################
# sgt_types.py | Authors: mcprisk, dcalco, joshdoc, neelnv                                         #
#                                                                                                  #
# This file contains all of the shared types needed for the smart game table games.                #
#                                                                                                  #
####################################################################################################

####################################################################################################
# IMPORTS                                                                                          #
####################################################################################################

from dataclasses import dataclass
from enum import Enum

import numpy as np

####################################################################################################
# TYPES                                                                                            #
####################################################################################################


class Loop_Result_t(Enum):
    CONTINUE = 1
    EXIT = 2


@dataclass
class Centroid:
    xpos: int
    ypos: int
    contour_hull: np.ndarray


@dataclass
class DetectedCentroids:
    fingers: list[Centroid]
    cds: list[Centroid]
    escape: bool
