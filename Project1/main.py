import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage as sk
import skimage.io as skio
from os import listdir
from os.path import isfile, join

"""
This function takes in two matricies and calculated their NCC summed.
"""
def ncc(matA, matB):
    sum = 0
    aNorm = np.divide(matA, np.linalg.norm(matA, axis=0).T)
    bNorm = np.divide(matB, np.linalg.norm(matB, axis=0).T)
    sum += np.sum(np.einsum('ij, ij -> i', aNorm, bNorm))

    aNorm = np.divide(matA, np.linalg.norm(matA, axis=1)[:, None])
    bNorm = np.divide(matB, np.linalg.norm(matB, axis=1)[:, None])
    sum += np.sum(np.einsum('ij, ij -> j', aNorm, bNorm))
    return sum

"""
Takes in x and y shift, matrix. It will shift the matrix by the amount specified (cut off the extra parts), and optionally zero pads to the
same dimensions as the original matrix.
"""
def shift_frame(x_shift, y_shift, matA, zeroPad = False):
    if x_shift < 0:
        if y_shift < 0:
            return np.pad(matA[:y_shift, :x_shift], ((0, -y_shift), (0, -x_shift)), mode="constant") if zeroPad else matA[:y_shift, :x_shift]
        else:
            return np.pad(matA[y_shift: ,:x_shift], ((y_shift, 0), (0, -x_shift)), mode="constant") if zeroPad else matA[y_shift: ,:x_shift]
    else:
        if y_shift < 0:
            return np.pad(matA[:y_shift ,x_shift:], ((0, -y_shift), (x_shift, 0)), mode="constant") if zeroPad else matA[:y_shift ,x_shift:]
        else:
            return np.pad(matA[y_shift: ,x_shift:], ((y_shift, 0), (x_shift, 0)), mode="constant") if zeroPad else matA[y_shift: ,x_shift:]

"""
Finds the shift needed to align the layerFrame to the BaseFrame.
Pyramid Layer: number of times to scale down the image by 0.5x

Returns a list of shifts that increasingly increases NCC score.
"""  
def find_shift(baseFrame, layerFrame, pyramidLayer):

    #Scaled down the image, and applied the function recursively
    if (pyramidLayer > 1):
        shiftRange = find_shift(sk.transform.rescale(baseFrame, 0.5, anti_aliasing = True), sk.transform.rescale(layerFrame, 0.5, anti_aliasing = True), pyramidLayer - 1)
    shift = []
    maxCorr = 0

    #Calculates the range of pixles to check.
    if (pyramidLayer > 1):
        highXRange = shiftRange[-1][0] * 2 + max(np.floor(baseFrame.shape[1] / 2 * 0.005).astype(int) + 1, 5)
        lowXRange = shiftRange[-1][0] * 2 - max(np.floor(baseFrame.shape[1] / 2 * 0.005).astype(int) + 1, 5)
        highYRange = shiftRange[-1][1] * 2 + max(np.floor(baseFrame.shape[0] / 2 * 0.005).astype(int) + 1, 5)
        lowYRange = shiftRange[-1][1] * 2 - max(np.floor(baseFrame.shape[0] / 2 * 0.005).astype(int) + 1, 5)
    else:
        highXRange = max(np.floor(baseFrame.shape[1] * 0.04).astype(int), 25)
        lowXRange = -highXRange
        highYRange = max(np.floor(baseFrame.shape[0] * 0.04).astype(int), 25)
        lowYRange = -highYRange
    
    #Shifts image, and calcualtes the NCC
    for y in range(lowYRange, highYRange):
        yShifted = np.roll(layerFrame, y, axis=0)
        for x in range(lowXRange, highXRange):
            xShifted = np.roll(yShifted, x, axis=1)
            layerFrameCropped = shift_frame(x, y, xShifted)
            baseFrameCropped = shift_frame(x, y, baseFrame)
            corrScore = ncc(baseFrameCropped, layerFrameCropped)
            if corrScore > maxCorr:
                maxCorr = corrScore
                shift.append([x,y])
    return shift

"""
Function that takes in file and colourizes it.
"""
def pipeline(imname):
    # read in the image
    im = skio.imread(f'./Project1/data/{imname}', cv2.IMREAD_GRAYSCALE)

    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)
    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    #Crop out border
    v_border = int(r.shape[0]*0.1)
    h_border = int(r.shape[1]*0.1)
    r_cropped = sk.util.img_as_float(r[v_border:-v_border, h_border:-h_border])
    b_cropped = sk.util.img_as_float(b[v_border:-v_border, h_border:-h_border])
    g_cropped = sk.util.img_as_float(g[v_border:-v_border, h_border:-h_border])
    rgb = np.array([r_cropped, g_cropped, b_cropped])

    #Adjust blackpoint and contrast, applies normalization
    for i in range(0, 3):
        img_adj = rgb[i]
        blackpoint = np.percentile(img_adj, 1)
        img_adj = np.clip(img_adj - blackpoint, 0, 1).astype(np.float64)
        
        img_adj = 1 - img_adj
        img_adj = 1.2 * img_adj
        img_adj = 1 - img_adj

        rgb[i] = cv2.normalize(img_adj, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    r_cropped = rgb[0]
    g_cropped = rgb[1]
    b_cropped = rgb[2]

    #Finding shifts needed to align images
    shift = []
    shift.append(find_shift(b_cropped, g_cropped, 4))
    shift.append(find_shift(b_cropped, r_cropped, 4))
    finalShift = np.array([shift[0][-1], shift[1][-1]])

    #Enable to print shifts needed.
    print(finalShift)

    #Shifts frames according to shifts calculated
    b = b_cropped
    g = shift_frame(finalShift[0, 0], finalShift[0, 1], np.roll(g_cropped, (finalShift[0, 0], finalShift[0, 1]), axis=(1, 0)), zeroPad=True)
    r = shift_frame(finalShift[1, 0], finalShift[1, 1], np.roll(r_cropped, (finalShift[1, 0], finalShift[1, 1]), axis=(1, 0)), zeroPad=True)
    bgr = np.array([r,g,b])

    #Calculates and crops images so that the region remaining is fully colourized.
    maxShifts = np.clip(np.max(finalShift, axis=0), a_min = 0, a_max = None)
    minShifts = np.clip(np.max(finalShift, axis=0), a_min = None, a_max = 0)
    if minShifts[0] < 0:
        if minShifts[1] < 0:
            bgr = bgr[:, maxShifts[1]:minShifts[1], maxShifts[0]:minShifts[0]]
        else:
            bgr = bgr[:, maxShifts[1]:, maxShifts[0]:minShifts[0]]
    else:
        if minShifts[1] < 0:
            bgr = bgr[:, maxShifts[1]:minShifts[1], maxShifts[0]:]
        else:
            bgr = bgr[:, maxShifts[1]:, maxShifts[0]:]
    
    colorized = (np.stack((bgr[0],bgr[1],bgr[2]), axis=2))
    colorized = cv2.normalize(colorized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    #Enable to show/save results
    # plt.axis('off')
    # plt.imshow(colorized)
    # plt.savefig(f'./Project1/Writeup/ImgData/{imname[:-4]}_colourized.jpg', bbox_inches='tight', pad_inches=0)

"""
Iterates through files and colourizes each one. Does not successfuly colourize Emir.
"""
def main():
    files = [f for f in listdir("./Project1/data") if isfile(join("./Project1/data", f))]
    for f in files:
        print(f)
        pipeline(f)

if __name__ == "__main__":
    main()