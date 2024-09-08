import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage as sk
import skimage.io as skio
from os import listdir
from os.path import isfile, join

def ncc(matA, matB):
    sum = 0
    aNorm = np.divide(matA, np.linalg.norm(matA, axis=0).T)
    bNorm = np.divide(matB, np.linalg.norm(matB, axis=0).T)
    # print(aNorm.shape)
    sum += np.sum(np.einsum('ij, ij -> i', aNorm, bNorm))
    # for y in range(0, matA.shape[0]):
    #     matAVec = aNorm[y, :]
    #     matBVec = aNorm[y, :]
    #     sum += np.dot(np.divide(matAVec, aNorm), np.divide(matBVec, bNorm, where= bNorm!=0))
    aNorm = np.divide(matA, np.linalg.norm(matA, axis=1)[:, None])
    bNorm = np.divide(matB, np.linalg.norm(matB, axis=1)[:, None])
    # print(aNorm.shape)
    sum += np.sum(np.einsum('ij, ij -> j', aNorm, bNorm))
    # for x in range(0, matA.shape[1]):
    #     matAVec = matA[:, x]
    #     matBVec = matB[:, x]
    #     aNorm = np.linalg.norm(matAVec)
    #     bNorm = np.linalg.norm(matBVec)
    #     sum += np.dot(np.divide(matAVec, aNorm), np.divide(matBVec, bNorm, where= bNorm!=0))
    return sum
    # return np.corrCoef(matA, matB)

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
        
def find_shift(baseFrame, layerFrame, pyramidLayer):
    if (pyramidLayer > 1):
        shiftRange = find_shift(sk.transform.rescale(baseFrame, 0.5, anti_aliasing = True), sk.transform.rescale(layerFrame, 0.5, anti_aliasing = True), pyramidLayer - 1)
    shift = []
    # bestFrame = []
    maxCorr = 0

    print("Pyramid layer/img shape", pyramidLayer, baseFrame.shape)
    if (pyramidLayer > 1):
        # print("shiftRange: ", shiftRange[-2])
        highXRange = shiftRange[-1][0] * 2 + max(np.floor(baseFrame.shape[1] / 2 * 0.005).astype(int) + 1, 5)
        lowXRange = shiftRange[-1][0] * 2 - max(np.floor(baseFrame.shape[1] / 2 * 0.005).astype(int) + 1, 5)
        highYRange = shiftRange[-1][1] * 2 + max(np.floor(baseFrame.shape[0] / 2 * 0.005).astype(int) + 1, 5)
        lowYRange = shiftRange[-1][1] * 2 - max(np.floor(baseFrame.shape[0] / 2 * 0.005).astype(int) + 1, 5)
    else:
        highXRange = max(np.floor(baseFrame.shape[1] * 0.04).astype(int), 25)
        lowXRange = -highXRange
        highYRange = max(np.floor(baseFrame.shape[0] * 0.04).astype(int), 25)
        lowYRange = -highYRange
    # print(f"roll range x/y: {lowXRange} - {highXRange-1}, {lowYRange} - {highYRange-1}")
    
    for y in range(lowYRange, highYRange):
        yShifted = np.roll(layerFrame, y, axis=0)
        for x in range(lowXRange, highXRange):
            xShifted = np.roll(yShifted, x, axis=1)
            layerFrameCropped = shift_frame(x, y, xShifted)
            baseFrameCropped = shift_frame(x, y, baseFrame)
            corrScore = ncc(baseFrameCropped, layerFrameCropped)
            # print(y,x)
            # print(corrScore)
            if corrScore > maxCorr:
                # print(corrScore, y, x)
                maxCorr = corrScore
                # print(corrScore)
                # bestFrame = [baseFrameCropped, layerFrameCropped]
                shift.append([x,y])
             
    # matchedFrame = np.stack((bestFrame[0], np.empty_like(bestFrame[0]), np.empty_like(bestFrame[0])), axis=2)   
    # matchedFrame = cv2.normalize(np.sum(bestFrame, axis=0), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # fig, axs = plt.subplots(1, 2, figsize=(20, 30))
    # axs[0].imshow(bestFrame[0], cmap='gray', vmin=0, vmax=1)
    # axs[1].imshow(bestFrame[1], cmap='gray', vmin=0, vmax=1)
    # plt.show()
    return shift

def pipeline(imname):
    # read in the image
    im = skio.imread(f'./Project1/data/{imname}', cv2.IMREAD_GRAYSCALE)
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    v_border = int(r.shape[0]*0.1)
    h_border = int(r.shape[1]*0.1)
    r_cropped = sk.util.img_as_float(r[v_border:-v_border, h_border:-h_border])
    b_cropped = sk.util.img_as_float(b[v_border:-v_border, h_border:-h_border])
    g_cropped = sk.util.img_as_float(g[v_border:-v_border, h_border:-h_border])
    rgb = np.array([r_cropped, g_cropped, b_cropped])

    for i in range(0, 3):
        img_adj = rgb[i]
        blackpoint = np.percentile(img_adj, 1)
        # print(blackpoint)
        img_adj = np.clip(img_adj - blackpoint, 0, 1).astype(np.float64)
        
        img_adj = 1 - img_adj
        img_adj = 1.2 * img_adj
        img_adj = 1 - img_adj

        rgb[i] = cv2.normalize(img_adj, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    r_cropped = rgb[0]
    g_cropped = rgb[1]
    b_cropped = rgb[2]

    shift = []
    shift.append(find_shift(b_cropped, g_cropped, 4))
    shift.append(find_shift(b_cropped, r_cropped, 4))
    
        
    finalShift = np.array([shift[0][-1], shift[1][-1]])


    b = b_cropped
    g = shift_frame(finalShift[0, 0], finalShift[0, 1], np.roll(g_cropped, (finalShift[0, 0], finalShift[0, 1]), axis=(1, 0)), zeroPad=True)
    r = shift_frame(finalShift[1, 0], finalShift[1, 1], np.roll(r_cropped, (finalShift[1, 0], finalShift[1, 1]), axis=(1, 0)), zeroPad=True)
    # print(r.shape, g.shape, b.shape)

    bgr = np.array([r,g,b])
    colorized= np.stack((r,g,b), axis=2)

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
    plt.axis('off')
    plt.imshow(colorized)
    plt.savefig(f'./Project1/Writeup/ImgData/{imname[:-4]}_colourized.jpg', bbox_inches='tight', pad_inches=0)
def main():
    files = [f for f in listdir("./Project1/data") if isfile(join("./Project1/data", f))]
    for f in files:
        print(f)
        pipeline(f)
    # pipeline('church.tif')

if __name__ == "__main__":
    main()