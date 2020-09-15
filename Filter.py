from matplotlib.image import imread
import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw
import numpy as np
import scipy.misc as smp

RGB_MAX = 255
DIM = 3

class Filter:
  def __init__(self,imgName, filterName):
    self.imgFolder = "images/"
    self.imgName = imgName
    self.filterName = filterName
    self.file = self.imgFolder + self.imgName
  
  def AverageFilter(self):
    img = imread(self.file)
    x, y, z = np.shape(img)
    newImg = np.zeros((x,y,z), dtype=np.uint16)
    for i in range(np.shape(img)[0]):
      for j in range(np.shape(img)[1]):
        windows = self.__getWindow(i, j, img)
        newPixel = self.__averageWindow(windows)
        newImg[i][j][0] = newPixel[0]
        newImg[i][j][1] = newPixel[1]
        newImg[i][j][2] = newPixel[2]
    
    imgPlot = plt.imshow(newImg)
    plt.show()
    return newImg
  
  def laPlaceFilter(self):
    img = imread(self.file)
    x, y, z = np.shape(img)
    newImg = np.zeros((x,y,z), dtype=np.uint16)
    for i in range(np.shape(img)[0]):
      for j in range(np.shape(img)[1]):
        windows = self.__getWindow(i, j, img)
        newPixel = self.__sobelWindow(windows)
        newImg[i][j][0] = newPixel[0]
        newImg[i][j][1] = newPixel[1]
        newImg[i][j][2] = newPixel[2]
    imgPlot = plt.imshow(newImg)
    plt.show()
    return newImg
  
  def sobelFilter(self):
    img = imread(self.file)
    x, y, z = np.shape(img)
    newImg = np.zeros((x,y,z), dtype=np.uint16)
    for i in range(np.shape(img)[0]):
      for j in range(np.shape(img)[1]):
        windows = self.__getWindow(i, j, img)
        newPixel = self.__sobelWindow(windows)
        newImg[i][j][0] = newPixel[0]
        newImg[i][j][1] = newPixel[1]
        newImg[i][j][2] = newPixel[2]
    imgPlot = plt.imshow(newImg)
    plt.show()
    return newImg

  def AveragePondFilter(self):
    img = imread(self.file)
    x, y, z = np.shape(img)
    newImg = np.zeros((x,y,z), dtype=np.uint16)
    for i in range(np.shape(img)[0]):
      for j in range(np.shape(img)[1]):
        windows = self.__getWindow(i, j, img)
        newPixel = self.__averagePondWindow(windows)
        newImg[i][j][0] = newPixel[0]
        newImg[i][j][1] = newPixel[1]
        newImg[i][j][2] = newPixel[2]
    imgPlot = plt.imshow(newImg)
    plt.show()
    return newImg

  def __sobelWindow(self, window):
    avgH = np.ones(DIM, dtype=np.uint8)
    avgV = np.ones(DIM, dtype=np.uint8)
    maskH = np.array([-1,-2,-1,0,0,0,1,2,1])
    maskV = np.array([-1,0,1,-2,0,2,-1,0,1])
    for i in range(DIM):
      tempH = window[i].reshape((1,-1)) * maskH
      tempV = window[i].reshape((1,-1)) * maskV
      avgH[i] = np.sum(tempH, dtype=np.uint8)/8
      avgV[i] = np.sum(tempV, dtype=np.uint8)/8
    return avgH + avgV

  def __laplaceWindow(self, window):
    avg = np.ones(DIM, dtype=np.uint8)
    mask = np.array([0,-1,0,-1,4,-1,0,-1,0])
    
    for i in range(DIM):
      temp = window[i].reshape((1,-1)) * mask
      avg[i] = np.sum(window[i], dtype=np.uint8)/8
    return avg

  def __averagePondWindow(self, window):
    avg = np.ones(DIM)
    for i in range(DIM):
      window[i][1][1] = window[i][1][1] + window[i][1][1]
      avg[i] = window[i].mean(dtype=np.uint16)
    return avg

  def __averageWindow(self, window):
    avg = np.ones(DIM)
    for i in range(DIM):
      avg[i] = window[i].mean(dtype=np.uint16)
    return avg

  def __getWindow(self, i, j, img):
    shape = img.shape
    mask = np.ones((DIM,DIM))
    matrix = np.ones((DIM,DIM,DIM))
    # print(mask)
    # exit()
    for k in range(3):
      if i == 0:
        if j == 0:
          mask[0][0] = img[1][1][k]
          mask[0][1] = img[1][0][k]
          mask[0][2] = img[1][1][k]
          mask[1][0] = img[0][1][k]
          mask[2][0] = img[1][1][k]
          mask[1][1] = img[0][0][k]
          mask[1][2] = img[0][1][k]
          mask[2][1] = img[1][0][k]
          mask[2][2] = img[1][1][k]
        elif j > 0 and j < shape[1] - 1:
          mask[0][0] = img[0][j-1][k]
          mask[0][1] = img[0][j][k]
          mask[0][2] = img[0][j+1][k]
          for m in range(1,3):
            p = 0
            for n in range(j-1,j+2):
              # if k == 1: print('['+str(m-1)+']['+str(n)+']')
              mask[m][p] = img[m-1][n][k]
              p = p + 1
        else:
          o = 0
          for m in range(1,3):
            p = j - 1
            for n in range(0,2):
              mask[m][n] = img[o][p][k]
              p = p + 1
            o = o + 1

          mask[0][0] = mask[2][0]
          mask[0][1] = mask[2][1]
          mask[0][2] = mask[2][0]
          mask[1][2] = mask[1][0]
          mask[2][2] = mask[2][0]
      elif i > 0 and i < shape[0] - 1:
        if j == 0:
          o = i - 1
          for m in range(0,3):
            p = 0
            for n in range(1,3):
              mask[m][n] = img[o][p][k]
              p = p + 1
            o = o + 1
          mask[0][0] = mask[0][2]
          mask[1][0] = mask[1][2]
          mask[2][0] = mask[2][2]
        elif j > 0 and j < shape[1] - 1:
          o = i - 1
          for m in range(0,3):
            p = j - 1
            for n in range(0,3):
              mask[m][n] = img[o][p][k]
              p = p + 1
            o = o + 1
        else:
          o = i - 1
          for m in range(0,3):
            p = j - 1
            for n in range(0,2):
              mask[m][n] = img[o][p][k]
              p = p + 1
            o = o + 1
      else:
        if j == 0:
          o = i - 1
          for m in range(0,2):
            p = j
            for n in range(1,3):
              mask[m][n] = img[o][p][k]
              p = p + 1
            o = o + 1
          mask[0][0] = mask[0][2]
          mask[1][0] = mask[1][2]
          mask[2][0] = mask[0][2]
          mask[2][1] = mask[0][1]
          mask[2][2] = mask[0][2]
        
        elif j > 0 and j < shape[1] - 1:
          o = i - 1
          for m in range(0,2):
            p = j - 1
            for n in range(0,3):
              mask[m][n] = img[o][p][k]
              p = p + 1
            o = o + 1
          mask[2][:] = mask[0][:]
        else:
          o = i - 1
          for m in range(0,2):
            p = j - 1
            for n in range(0,2):
              mask[m][n] = img[o][p][k]
              p = p + 1
            o = o + 1
          mask[0][2] = mask[0][0]
          mask[1][2] = mask[1][0]
          mask[2][0] = mask[0][0]
          mask[2][1] = mask[0][1]
          mask[2][2] = mask[0][0]
      matrix[k] = mask
    return matrix[0], matrix[1], matrix[2]