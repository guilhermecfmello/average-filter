from matplotlib.image import imread
from Filter import Filter
import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw
import numpy as np
import scipy.misc as smp
import sys


# Arguments tratment function
def getArgs(args, param):
    try:
        argValue = False
        for a in args:
            if(a == '-h'):
                print('============= Help Menu =============')
                print("Exec format: python main.py [args] | python main.py -h for help")
                print("Commands:")
                print('-i "imageName" || -f "filterName"')
                print('Available filters: "avg" or "pAvg"')
                # print('-f format of image output, can be: "cmyk" or "hsi')
                return False
        for i in range(1, len(args)):
            # print("args len: " + str(len(args)))
            # print("i: " + str(i))
            if(args[i] == param):
                argValue = args[i+1]
        if(argValue == False): raise Exception("Param " + str(param) + " couldn't be found")
        return argValue
    except Exception as e:
        print("Exec format: python main.py [args] | python main.py -h for help.\nError: " + str(e))
        exit()    


# Todo: Normalize, Accumulation, equalizer

if __name__ == '__main__':
    imgName = getArgs(sys.argv, "-i")
    if imgName == False:
      exit()
    filterName = getArgs(sys.argv, "-f")
    if filterName == False:
      exit()
    filt = Filter(imgName, filterName)
    if filterName == 'avg':
      filt.AverageFilter()
    elif filterName == 'pAvg':
      filt.AveragePondFilter()
    # eq = Equalizer(imgName)
    # eq.imgNormalizer()


