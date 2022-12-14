'''
1. convert the frame to grayscale
2. detect the face outline
3. predict the face landmarks
4. return the landmarks
'''
import numpy as np
import cv2
import dlib
from imutils import face_utils
from dotenv import load_dotenv
import os

load_dotenv()

model_path = os.environ.get('MODEL_PATH')

predictor = dlib.shape_predictor(model_path)
detector = dlib.get_frontal_face_detector()

BLOWUP_FACTOR = 1 # Resizes image before doing the algorithm. Changing to 2 makes things really slow. So nevermind on this.
RELEVANT_DIST_FOR_CORNER_GRADIENTS = 8*BLOWUP_FACTOR
dilationWidth = 1+2*BLOWUP_FACTOR #must be an odd number
dilationHeight = 1+2*BLOWUP_FACTOR #must be an odd number
dilationKernel = np.ones((dilationHeight,dilationWidth),'uint8')

target_landmarks = [28 ,29, 30, 31, 37,38,39,40, 43,44,45,46 ]
target_landmarks = [a - 1 for a in target_landmarks]   #python indexing starts from zero

# Estimates the probability that the given cx,cy is the pupil center, by taking
# (its vector to each gradient location) dot (the gradient vector)
# only uses gradients which are near the peak of a histogram of distance
# cx and cy may be integers or floating point.
def phiWithHist(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords, IRIS_RADIUS):
    vecx = gradXcoords-cx
    vecy = gradYcoords-cy
    lengthsSquared = np.square(vecx)+np.square(vecy)
    # bin the distances between 1 and IRIS_RADIUS. We'll discard all others.
    binWidth = 1 
    numBins =  int(np.ceil((IRIS_RADIUS-1)/binWidth))
    bins = [(1+binWidth*index)**2 for index in range(numBins+1)] #express bin edges in terms of length squared
    hist = np.histogram(lengthsSquared, bins)[0]
    maxBin = hist.argmax()
    slop = binWidth
    valid = (lengthsSquared > max(1,bins[maxBin]-slop)) &  (lengthsSquared < bins[maxBin+1]+slop) 
    dotProd = np.multiply(vecx,gradDX)+np.multiply(vecy,gradDY)
    valid = valid & (dotProd > 0) # only use vectors in the same direction (i.e. the dark-to-light transition direction is away from us. The good gradients look like that.)
    dotProd = np.square(dotProd[valid]) # dot products squared
    dotProd = np.divide(dotProd,lengthsSquared[valid]) #make normalized squared dot products
    dotProd = np.square(dotProd) # squaring puts an even higher weight on values close to 1
    return np.sum(dotProd) # this is equivalent to normalizing vecx and vecy, because it takes dotProduct^2 / length^2

#Takes as input an eye gray images and   Returns (cy,cx) of the pupil center. 
def getPupilCenter(gray, getRawProbabilityImage=False):

    gray = gray.astype('float32')
    if BLOWUP_FACTOR != 1:
        gray = cv2.resize(gray, (0,0), fx=BLOWUP_FACTOR, fy=BLOWUP_FACTOR, interpolation=cv2.INTER_LINEAR)

    IRIS_RADIUS = gray.shape[0]*.75/2 #conservative-large estimate of iris radius 
    dxn = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3) 
    dyn = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    magnitudeSquared = np.square(dxn)+np.square(dyn)

    # ########### Pupil finding
    magThreshold = magnitudeSquared.mean()*.6 #only retain high-magnitude gradients. <-- VITAL TUNABLE PARAMETER
                    # The value of this threshold is critical for good performance.

    # form a bool array, unrolled columnwise, which can index into the image.
    # we will only use gradients whose magnitude is above the threshold, and
    # (optionally) where the gradient direction meets characteristics such as being more horizontal than vertical.
    gradsTouse = (magnitudeSquared>magThreshold) & (np.abs(4*dxn)>np.abs(dyn))
    lengths = np.sqrt(magnitudeSquared[gradsTouse]) #this converts us to double format
    gradDX = np.divide(dxn[gradsTouse],lengths) #unrolled columnwise
    gradDY = np.divide(dyn[gradsTouse],lengths)


    isDark = gray< (gray.mean()*.8)  #<-- TUNABLE PARAMETER
    global dilationKernel
    isDark = cv2.dilate(isDark.astype('uint8'), dilationKernel) #dilate so reflection goes dark too


    gradXcoords =np.tile( np.arange(dxn.shape[1]), [dxn.shape[0], 1])[gradsTouse] # build arrays holding the original x,y position of each gradient in the list.
    gradYcoords =np.tile( np.arange(dxn.shape[0]), [dxn.shape[1], 1]).T[gradsTouse] # These lines are probably an optimization target for later.
    minXForPupil = 0 #int(dxn.shape[1]*.3)

    #histogram method
    centers = np.array([[phiWithHist(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords, IRIS_RADIUS) if isDark[cy][cx] else 0 for cx in range(minXForPupil,dxn.shape[1])] for cy in range(dxn.shape[0])]).astype('float32')
    
    maxInd = centers.argmax()
    (pupilCy,pupilCx) = np.unravel_index(maxInd, centers.shape)
    pupilCx += minXForPupil
    pupilCy /= BLOWUP_FACTOR
    pupilCx /= BLOWUP_FACTOR

    return (pupilCy, pupilCx)

def update(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    print('detecting faces')
    if len(rects) == 1:
        print('One face detected..')
        for i, rect in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # geting the right eye's pupil position:
            rex = shape[36,0]
            rey = shape[37,1]       
            rexx = shape[39,0]
            reyy = shape[40,1]    
         
            reye= gray[rey-5:reyy+5, rex-5:rexx+5]
            (rcy,rcx) = getPupilCenter(reye)
            right_pupil_pos = (round(rcx+rex), round(rcy+rey))

            # geting the left eye's pupil position:
            lex = shape[42,0]
            ley = shape[43,1]       
            lexx = shape[45,0]
            leyy = shape[46,1]    
         
            reye= gray[ley-5:leyy+5, lex-5:lexx+5]
            (lcy,lcx) = getPupilCenter(reye)
            left_pupil_pos = (round(lcx+lex), round(lcy+ley))

            #check the data point             
            pupil_asarray = np.zeros((2,2))
            pupil_asarray[0,:] = left_pupil_pos
            pupil_asarray[1,:] = right_pupil_pos
            current_data = np.concatenate((shape[target_landmarks,:],pupil_asarray))
            current_data =  np.reshape(current_data,(-1,))
        return current_data
    else:
        return None  