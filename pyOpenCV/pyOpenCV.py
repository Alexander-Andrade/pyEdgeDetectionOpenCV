import numpy as np
import cv2 as cv

def cannyEdgeDetection():
    #0 - default camera
    cap = cv.VideoCapture(0)
    while True:
        #get frame
        ret,frame = cap.read()
        if not ret:
            continue
        #convert to grayscale
        pureFrame = frame.copy()
        img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        img = cv.medianBlur(img,5)
        #img = cv.GaussianBlur(img,(7,7), 1.5, 1.5);
        '''
        arg0 = frame - input image
        arg1 = 100 - minVal
        arg2 = 200 - maxVal, threshold values, any edges with intensity gradient more than maxVal are sure to
        be edges and those below minVal sure to be non-edges,so discarded
        arg3 - aperture-size - the size of Sobel kernel used for find image gradients (3 - by default)
        arg4 - L2gradient which specifies the equation for finding gradient magnitude (by default = false, which means that
        it uses function: Edge_gradient (G) = |Gx| + |Gy| )
        states: 5x5 Gaussian filter
        '''
        edges = cv.Canny(img,50,300,None,3,False)
        '''
        arg0 = source image
        arg1 = countour retrieval mode
        arg2 = contour approximation method
        return contours = python list of all the contours in the image. Each individual contour is a Numpy array of
        (x,y) coordinates of boundary points of the object
        '''
        #as findContours() modifies the source image, let's make a copy of correct img
        img,contours,hierarchy = cv.findContours(edges,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        '''
        arg0 = img
        arg1 = contours - python list
        arg2 = index of contours (for drawing indiv cotour), to draw all contours pass -1
        arg3... color, thickness etc.
        '''
        contourImg = cv.drawContours(pureFrame,contours,-1,(0,255,0),3)

        #cv.imshow('original',frame)
        cv.imshow('canny',edges)
        cv.imshow('contours',contourImg)
        #quit code = 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    #when everything is done, release the capture
    cap.release()
    cv.destroyAllWindows()

def thresholdEdgeDetection():
    cap = cv.VideoCapture(0)
    while True:
        #get frame
        ret,frame = cap.read()
        if not ret:
            continue
        #removing salt-and-pepper noise.
        '''
        cv2.medianBlur(src, ksize[, dst]) → dst
        Parameters:	
        src – input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
        dst – destination array of the same size and type as src.
        ksize – aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ...
        '''
        '''
        Gaussian filtering is highly effective in removing Gaussian noise from the image.
        cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) → dst
        Parameters:	
        src – input image; the image can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
        dst – output image of the same size and type as src.
        ksize – Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero’s and then they are computed from sigma* .
        sigmaX – Gaussian kernel standard deviation in X direction.
        sigmaY – Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height , respectively (see getGaussianKernel() for details); to fully control the result regardless of possible future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.
        borderType – pixel extrapolation method (see borderInterpolate() for details).
        '''
        #cv.imshow('fr',frame)
        pureFrame = frame.copy()
        img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        #cv.imshow('grayscale',img)
        img = cv.medianBlur(img,5)
        img = cv.GaussianBlur(img,(5,5),0)

        #cv.imshow('median',img)
        '''
        Parameters:	
        src – Source 8-bit single-channel image.
        dst – Destination image of the same size and the same type as src .
        maxValue – Non-zero value assigned to the pixels for which the condition is satisfied. See the details below.
        adaptiveMethod – Adaptive thresholding algorithm to use, ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C . See the details below.
        thresholdType – Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV .
        blockSize – Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
        C – Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
        '''
        thresh1 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        cv.imshow('gaussian',thresh1)
        #thresh2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
        #cv.imshow('mean',thresh2)

        img,contours,hierarchy = cv.findContours(thresh1,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        contourImg = cv.drawContours(pureFrame,contours,-1,(0,255,0),3)
        cv.imshow('contours',contourImg)
        #quit code = 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    #when everything is done, release the capture
    cap.release()
    cv.destroyAllWindows()

def colorObjectDetection():
    cap = cv.VideoCapture(0)
    while True:
        #get frame
        ret,frame = cap.read()
        if not ret:
            continue
        pureFrame = frame.copy()
        #convert from RGB to HSV
        img = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

if __name__ == "__main__":
    #cannyEdgeDetection()
    thresholdEdgeDetection()
    
