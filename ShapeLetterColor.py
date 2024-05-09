import cv2
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt

#img = cv2.imread(f"./images/Green.png")

# cv2.imwrite("./images/hue.png", pix[:,:,0])
# mask = cv2.Mat(pix[:,:,1])
# mask = np.ones((260,250,1), dtype="uint8") * 255
# mask = pix[:,:,1]
# hist = cv2.calcHist([pix], [mask], None, [180],[0,180])



# y, x, k = pix.shape
# pix = pix[int(y*0.1):(y-int(y*0.1)), int(x*0.1):(x-int(x*0.1))]
# cv2.imshow("image", pix[:,:,1])
# cv2.waitKey(0), mask)
# cv2.waitKey(0)

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

def isNotGrayFuncGen(_v):
    def isNotGray(h,s,v):
        if (s/255.)**2*3+(v/255.-_v)**2*1 <= 0.15**2:
            return 0
        else:
            return 1.0
    return isNotGray
    
def isNotColorfuncgen(h):
    hue_range = (15/360)**2
    sat_range = (255*1//10, 255)
    val_range = (255*1//10, 255)

    def funcgen(h_,s_,v_):
        diff = (((h_*2-h)%360+180)%360-180)
        # print(((((h_*2-h)%360+180)%360-180)/360.)**2/hue_range, sat_range[0] <= s_ <= sat_range[1], val_range[0] <= v_ <= val_range[1], sep="\t")
        if (diff/360.)**2/hue_range <= 1**2 and sat_range[0] <= s_ <= sat_range[1] and val_range[0] <= v_ <= val_range[1]:
            return 0
        else:
            return 1.0
    return funcgen

def isNotValfuncgen(v):
    # assuming the shape can't be gray
    if v > 255/2:
        def funcgen(h_,s_,v_):
            if s_ <= 40 and abs(v_/255.-v/255.) < 20/255:
                return 0
            else:
                return 1.0
    else:
        def funcgen(h_,s_,v_):
            if s_ <= 40 and abs(v_/255.-v/255.) < 20/255:
                return 0
            else:
                return 1.0
    return funcgen

def maskColor(pix, func, erode_factor=0.1, dilate_factor=0.1, coutour_type=cv2.RETR_EXTERNAL):
    y, x, _ = pix.shape
    mask = np.asarray([[0.0 for _ in range(x)] for _ in range(y)])
    for i in range(y):
        for j in range(x):
            if i < y // 10 or y * 9 // 10 < i:
                continue
            if j < x // 10 or x * 9 // 10 < j:
                continue
            h,s,v = pix[i][j]
            mask[i][j] = func(h,s,v)

    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    if erode_factor > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(erode_factor*(x+y)/4),int(erode_factor*(x+y)/4)))
        mask = cv2.erode(mask, kernel, iterations=1)
    if dilate_factor > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(dilate_factor*(x+y)/2),int(dilate_factor*(x+y)/2)))
        mask = cv2.dilate(mask, kernel, iterations=1)
    if erode_factor > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(erode_factor*(x+y)/4),int(erode_factor*(x+y)/4)))
        mask = cv2.erode(mask, kernel, iterations=1)

    mask = np.asarray(mask*255, dtype=np.uint8)
    contours, hier = cv2.findContours(mask,coutour_type,cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros((y,x,3), np.uint8)

    # cnts = []
    # for cnt in contours:
    #     if x*y*min_size < cv2.contourArea(cnt):
    #         cnts.append(cnt)
    cv2.drawContours(mask,contours,-1,(255, 255, 255),thickness=cv2.FILLED)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY, dstCn=1)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    return mask

distribution = [(-2, 1/4), (-1, 1/3), (0, 1/2), (1, 1/3), (2, 1/4)]
def getPeakHue(pix, mask):
    y, x, _ = pix.shape
    hist = [0.0 for _ in range(180)]
    for i in range(y):
        for j in range(x):
            h = pix[i][j][0]
            if mask[i][j] < 128:
                continue

            for k,d in distribution:
                hist[(h+k)%len(hist)] += d
    shape_hue = np.argmax(hist)*2
    return shape_hue
def getPeakSat(pix, mask):
    y, x, _ = pix.shape
    hist = [0.0 for _ in range(256//4)]
    for i in range(y):
        for j in range(x):
            s = pix[i][j][1]
            if mask[i][j] < 128:
                continue

            for k,d in distribution:
                hist[clamp(s//4+k,0,len(hist)-1)] += d
    shape_sat = np.argmax(hist)*4
    return shape_sat

def getPeakVal(pix, mask):
    y, x, _ = pix.shape
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    hist = [0.0 for _ in range(256//4)]
    for i in range(y):
        for j in range(x):
            if i < y // 10 or y * 9 // 10 < i:
                continue
            if j < x // 10 or x * 9 // 10 < j:
                continue
            if mask[i][j] < 128:
                continue
            v = pix[i][j][2]
            if 255*2//5 <= v <= 255*4//5:
                continue
            for k,d in distribution:
                hist[clamp(v//4+k,0,len(hist)-1)] += d
    shape_val = np.argmax(hist)*4
    return shape_val

def find_gray_val(pix):
    # Gray is the average of the outer boundary of the image
    x = pix.shape[0]
    y = pix.shape[1]
    total = 0
    num = 0
    for i in range(x):
        total += pix[i][0][2]
        total += pix[i][y-1][2]
        num += 2
    
    for i in range(y):
        total += pix[0][i][2]
        total += pix[x-1][i][2]
        num += 2

    return total/num

def getShapeLetterHSV(img):
    pix = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2HSV)
    # pix[:,:,2] = cv2.equalizeHist(pix[:,:,2])
    img_output = cv2.cvtColor(pix, cv2.COLOR_HSV2BGR)

    y, x, _ = pix.shape
    # cv2.imshow("image", img_output)

    gray = find_gray_val(pix)
    print(f"{gray/255=}")
    mask = maskColor(pix, isNotGrayFuncGen(gray/255.), erode_factor=0.2, dilate_factor=0.2)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    mask2 = None
    if mask.mean() <= 255 * 20 // 100:
        shape_val = getPeakVal(pix, ~mask)
        avgShapeColor = (0, 0, shape_val)
        maskLetter = maskColor(pix, isNotGrayFuncGen(gray), erode_factor=0.1, dilate_factor=0.1)
    else:
        shape_sat = getPeakSat(pix, mask)
        print(f"{shape_sat=}")
        if shape_sat >= 40:
            shape_hue = getPeakHue(pix, mask)
            print(f"{shape_hue=}")
            mask2 = maskColor(pix, isNotColorfuncgen(shape_hue), erode_factor=0.05, dilate_factor=0.05, coutour_type=cv2.RETR_LIST)
            # cv2.waitKey(0)
        else:
            shape_val = getPeakVal(pix, mask)
            # print(f"{shape_val=}")
            mask2 = maskColor(pix, isNotValfuncgen(shape_val), erode_factor=0.05, dilate_factor=0.05, coutour_type=cv2.RETR_LIST)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(0.10*(x+y)/2),int(0.10*(x+y)/2)))
        mask = cv2.erode(mask, kernel, iterations=1)
        maskShape = mask & ~mask2

        # temp =  np.ma.masked_array(img_output,np.broadcast_to(~maskShape[...,np.newaxis], img_output.shape), fill_value=0)
        # cv2.imshow("maskShape",np.ma.filled(temp, 0))
        # cv2.imshow("mask", maskShape)
        # cv2.waitKey(0)
        maShape = np.ma.masked_array(pix,np.broadcast_to(~maskShape[...,np.newaxis], pix.shape))
        avgShapeColor = np.ma.median(maShape, (0,1)).data
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(0.01*(x+y)/2),int(0.01*(x+y)/2)))
        # mask2 = cv2.erode(mask2, kernel, iterations=1)
        maskLetter = mask & mask2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(0.03*(x+y)/2),int(0.03*(x+y)/2)))
    maskLetter = cv2.erode(maskLetter, kernel)
    # cv2.imshow("mask", maskLetter)
    # cv2.waitKey(0)

    maLetter = np.ma.masked_array(pix,np.broadcast_to(~maskLetter[...,np.newaxis], pix.shape))
    avgLetterColor = np.ma.median(maLetter, (0,1)).data
    # letter_hue = get_peak_hue(pix, maskLetter)
    if mask2 is not None:
        return avgShapeColor, avgLetterColor, None, mask & mask2
    else:
        return avgShapeColor, avgLetterColor, None, maskLetter


def HSV2Color(h,s,v):
    # Finds the most similar color to the input
    if h == 0.0 and s == 0.0 and v == 0.0:
        return "not found"
    if (255 - v)/3 + s < 255*15//100:
        return "white"
    elif v < 255*30//100 or v < 255*55//100 and s < 255*20//100:
        return "black"
    elif s < 255*20//100 and v < 255*80//100 or s < 255*10/100:
        return "gray"
    elif h<=14//2 or 320//2 <= h:
        if 10//2 <= h <=14//2 and s < 255*5//10 and v > 255 * 7 // 10:
            return "orange"
        return "red"
    elif h<=44//2:
        if s > 255*5//10 and v < 255 * 7 // 10:
            return "brown"
        return "orange"
    elif h<=78//2:
        return "yellow"
    elif h<=170//2:
        return "green"
    elif h<=250//2:
        # Might have some issues here, because "light indigo" appears to be purple, but when the color is more saturated it becomes blue 
        return "blue"
    elif h<=320//2:
        return "purple"
    else:
        return f"Color Not Available: HSV={h,s,v}"

def getShapeLetterColor(img):

    avgShapeHSV, avgLetterHSV, _ , _= getShapeLetterHSV(img)
    shapeCol, letterCol = HSV2Color(*avgShapeHSV), HSV2Color(*avgLetterHSV)
    print(f"Shape: {avgShapeHSV}={shapeCol}\tLetter: {avgLetterHSV}={letterCol}")
    return shapeCol, letterCol

if __name__ == '__main__':
    img = cv2.imread(f"./images/DJI_0491_6.jpg")
    avgShapeColor, avgLetterColor = getShapeLetterColor(cv2.resize(img,(80,80)))
    print(f"{avgShapeColor=}\n{avgLetterColor=}")

# cv2.imshow("mask", mask)
# cv2.waitKey(0)
# plt.plot(hist, color='b')
# plt.xlabel("Value (Hue)")
# plt.ylabel("Number of pixels")
# plt.show()