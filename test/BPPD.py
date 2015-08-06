import cv, cv2
import sys
from matplotlib import pyplot as plt
import numpy

CAMERA_INDEX = 0
HUE_MIN = 25
HUE_MAX = 120
PIXEL_WIDTH = 640
PIXEL_HEIGHT = 480
THRESHOLD_PERCENTILE = 95

try:
    f = sys.argv[1]
    bgr = cv2.imread(f)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue_min = HUE_MIN
    hue_max = HUE_MAX
    sat_min = hsv[:,:,1].mean()
    sat_max = 255
    val_min = numpy.percentile(hsv[:,:,2], 15)
    val_max = numpy.percentile(hsv[:,:,2], 85)
    threshold_min = numpy.array([hue_min, sat_min, val_min], numpy.uint8)
    threshold_max = numpy.array([hue_max, sat_max, val_max], numpy.uint8)
    mask = cv2.inRange(hsv, threshold_min, threshold_max)
    kernel = numpy.ones((5,5), numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    column_sum = mask.sum(axis=0) # vertical summation
    threshold = numpy.percentile(column_sum, THRESHOLD_PERCENTILE)
    probable = numpy.nonzero(column_sum >= threshold) # returns 1 length tuble
    num_probable = len(probable[0])
    centroid = int(numpy.median(probable[0]))
    egi = numpy.dstack((mask, mask, mask))
    bgr[:,centroid,:] = 255
    egi[:,centroid,:] = 255
    hsv[:,centroid,:] = 255
    output = numpy.hstack((bgr, hsv, egi))
    cv2.imshow('', output)
    cv2.waitKey(0)
except Exception as e:
    print str(e)
