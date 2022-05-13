import sys
import getopt
import numpy as np
import math
import cmath
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
import cv2
import time
import random
import copy

# Default mode is 1
mode = 1
image = cv2.imread('moonlanding.png', cv2.IMREAD_GRAYSCALE)

dictionnaryK = {}
dictionnary = {}

xkdictionnary = {}

hit = 0
miss = 0

negexp = (-1j)*2*np.pi
posexp = (1j)*2*np.pi

# Method to parse input to update global variables


def opts():
    listOfGlobals = globals()

    options, nonOptions = getopt.getopt(sys.argv[1:], "m:i:")

    for(opt, arg) in options:
        if(opt == "-m"):
            if int(arg) < 1 or int(arg) > 4:
                print("ERROR\tInvalid mode. Expected mode between 1 and 4")
                exit()
            listOfGlobals['mode'] = int(arg)
        if(opt == "-i"):
            listOfGlobals['image'] = cv2.imread(arg, cv2.IMREAD_GRAYSCALE)


def resize_arr(array):
    N = array.size
    if N & (N-1) != 0:

        pad = int(np.power(2, np.floor(np.log2(N)) + 1) - N)
        print(pad, np.floor(pad/2), pad-np.floor(pad/2))
        array = np.pad(array, (int(np.floor(pad/2)),
                               int(pad-np.floor(pad/2))), 'constant')
    return array


def pad_image(image):
    num_rows = len(image)
    num_cols = len(image[0])
    cols_pad = 0
    rows_pad = 0

    if num_cols & (num_cols-1) != 0:
        cols_pad = int(np.power(2, np.floor(np.log2(num_cols)) + 1) - num_cols)
    if num_rows & (num_rows-1) != 0:
        rows_pad = int(np.power(2, np.floor(np.log2(num_rows)) + 1) - num_rows)

    left = np.floor(rows_pad/2)
    right = rows_pad-np.floor(rows_pad/2)
    top = np.floor(cols_pad/2)
    bottom = cols_pad-np.floor(cols_pad/2)

    padded = np.pad(image, [(int(left), int(right)), (int(top), int(bottom))])

    return padded


def dft_naive(array):
    N = array.size
    sum_arr = []

    for k in range(N):
        sum = base_dft(array, k)

        sum_arr.append(sum)

    return np.array(sum_arr)

def dft2_naive(array):
    N = len(array)
    M = len(array[0])

    narr = np.arange(N)
    marr = np.arange(M)
    const = (-1j)*2*np.pi

    
    result = np.empty((int(N), int(M)), dtype=complex)

    for l in range(N):
        for k in range(M):
            sum = 0 
            for n in range(N):
                exps = np.exp(const*(k*marr/M + l*n/N))
                res = np.dot(array[n,:], exps)
                sum = sum + res

            result[l][k] = sum

    return result


def base_inv_dft(array, k):
    N = len(array)
    const = posexp*k/N
    narr = np.arange(N)
    exps = np.exp(const*narr)

    return np.dot(array, exps)


def inv_dft_naive(array):
    N = array.size
    sum_arr = []

    for k in range(N):
        sum = base_inv_dft(array, k)
        sum_arr.append(sum.real/N)

    return np.array(sum_arr)


def split_inv_fft(array, k, index, level):
    if(len(array) <= 1):
        evenOddResults[(index, level, 0)] = base_inv_dft(array, 0)
        return
    evenArray = array[0::2]
    oddArray = array[1::2]
    newk = k % len(evenArray)
    oddResult = evenOddResults.get((index+len(evenArray), level+1, newk))
    if(oddResult == None):
        split_inv_fft(oddArray, newk, index+len(evenArray), level+1)
        oddResult = evenOddResults.get((index+len(evenArray), level+1, newk))

    evenResult = evenOddResults.get((index, level+1, newk))
    if(evenResult == None):
        split_inv_fft(evenArray, newk, index, level+1)
        evenResult = evenOddResults.get((index, level+1, newk))

    exp = dictionnaryK.get((len(array), k, 1))
    if(exp == None):
        smallEXP = dictionnary.get((len(array), 1))
        if(smallEXP == None):
            smallEXP = np.exp(posexp/len(array))
            dictionnary[(len(array), 1)] = smallEXP
        exp = np.power(smallEXP, k)
        dictionnaryK[(len(array), k, 1)] = exp

    evenOddResults[(index, level, k)] = evenResult+oddResult*exp


def inv_fft(array):
    N = len(array)
    evenOddResults.clear()
    dictionnaryK.clear()
    dictionnary.clear()
    shape = np.shape(array)
    result = np.empty([shape[0]], dtype=complex)
    for k in range(0, len(array)):
        split_inv_fft(array, k, 0, 0)
        result[k] = np.flip(evenOddResults[(0, 0, k)]/N)

    return np.array(result)


def inv_fft2(array):
    shape = np.shape(array)
    intermediateResult = np.empty([shape[0], shape[1]], dtype=complex)
    result = np.empty([shape[0], shape[1]], dtype=complex)

    for x in range(0, len(array[:, 0])):
        intermediateResult[x, :] = inv_fft(array[x, :])

    for x in range(0, len(array[0, :])):
        result[:, x] = inv_fft(intermediateResult[:, x])

    return result

evenOddResults = {}


def attemptedDiv(array, k, index, level):
    if(len(array) <= 1):  # Base case-do naive implementation
        evenOddResults[(index, level, 0)] = base_dft(array, 0)
        return
    evenArray = array[0::2]
    oddArray = array[1::2]
    newk = k % len(evenArray)  # Reduce k for smaller arrays if necessary
    # Get the odd result if it is in the dictionnary. If not, compute it
    oddResult = evenOddResults.get((index+len(evenArray), level+1, newk))
    if(oddResult == None):
        attemptedDiv(oddArray, newk, index+len(evenArray), level+1)
        oddResult = evenOddResults.get((index+len(evenArray), level+1, newk))

    # Get the even result if it is in the dictionnary. If not, compute it.
    evenResult = evenOddResults.get((index, level+1, newk))
    if(evenResult == None):
        attemptedDiv(evenArray, newk, index, level+1)
        evenResult = evenOddResults.get((index, level+1, newk))

    # The exp operation results are stored in a dictionnary to save on compute time.
    exp = dictionnaryK.get((len(array), k, 1))
    if(exp == None):
        smallEXP = dictionnary.get((len(array), 1))
        if(smallEXP == None):
            smallEXP = np.exp(negexp/len(array))
            dictionnary[(len(array), 1)] = smallEXP
        exp = np.power(smallEXP, k)
        dictionnaryK[(len(array), k, 1)] = exp

    evenOddResults[(index, level, k)] = evenResult+oddResult*exp


def base_dft(array, k):
    N = len(array)
    const = negexp*k/N
    narr = np.arange(N)
    exps = np.exp(const*narr)

    return np.dot(array, exps)


def split_fft(array, k, numsplit):
    if len(array) == 2 or numsplit == 4:
        return base_dft(array, k)

    even_arr = array[0::2]
    odd_arr = array[1::2]

    odd_res = split_fft(odd_arr, k, numsplit+1)
    even_res = split_fft(even_arr, k, numsplit+1)
    exp = np.exp(negexp*k/len(array))

    return even_res+odd_res*exp


def fft(array):
    evenOddResults.clear()
    dictionnaryK.clear()
    dictionnary.clear()
    mid = math.ceil(len(array)/2.0)
    shape = np.shape(array)
    result = np.empty([shape[0]], dtype=complex)
    for x in range(0, len(array)):
        attemptedDiv(array, x, 0, 0)
        result[x] = evenOddResults[(0, 0, x)]
    return np.array(result)


def fft2(array):
    shape = np.shape(array)
    intermediateResult = np.empty([shape[0], shape[1]], dtype=complex)
    result = np.empty([shape[0], shape[1]], dtype=complex)

    for x in range(0, len(array[:, 0])):

        tempResult = fft(array[x, :])
        intermediateResult[x, :] = tempResult

    for x in range(0, len(array[0, :])):

        tempResult = fft(intermediateResult[:, x])
        result[:, x] = (tempResult)
    return np.array(result)


def test1d(array):

    print("INITIAL ARRAY")
    print(array)
    print("PADDED ARRAY")
    array = resize_arr(array)
    print(array)

    print("EXPECTED np.fft")
    expected = np.fft.fft(array)
    print(expected)

    print("DFT")
    starTime = time.perf_counter()
    calculated = dft_naive(np.float32(array))
    print(time.perf_counter()-starTime)
    print(calculated)

    print("FFT")
    starTime = time.perf_counter()
    resfft = fft(np.float32(array))
    print(time.perf_counter()-starTime)
    print(resfft)

    print("INVERSE")
    inverse = inv_dft_naive(calculated)
    print(inverse)

    print("INVERSE FFT")
    inversefft = inv_fft(calculated)
    print(inversefft)

    print("INVERSE FFT NP")
    inversefft = np.fft.ifft(calculated)
    print(inversefft)


def test2d(array):
    print("INITIAL ARRAY")
    print(array)
    print("PADDED ARRAY")
    array = resize_arr(array)
    print(array)

    print("Expected 2d fft")
    expected = np.fft.fft2(array)
    print(expected)

    print("2D FFT")
    starTime = time.perf_counter()
    resfft2 = fft2(array)
    print(time.perf_counter()-starTime)
    print(resfft2)

    print("2D INVERSE")
    print(inv_fft2(fft2))


def denoise(array):
    prop = 0.05
    num_rows = len(array)
    num_cols = len(array[0])

    array[int(num_rows*prop):int(num_rows-num_rows*prop)] = 0
    array[:, int(num_cols*prop):int(num_cols-num_cols*prop)] = 0

    return array, prop


def mode1():
    array = pad_image(image)

    starTime = time.perf_counter()
    resfft = fft2(np.float32(array))

    # Used to compare our implementation in the first experiment
    # resfft = np.fft.fft2(np.float32(array))

    print("FFT2 time:", time.perf_counter()-starTime, "seconds")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax1.title.set_text('Original Image')
    ax2.imshow(np.log(abs(resfft)))
    ax2.title.set_text('FFT Image')
    plt.show()


def mode2():
    array = pad_image(image)

    starTime = time.perf_counter()
    resfft = fft2(np.float32(array))
    print("FFT2 time:", time.perf_counter()-starTime, "seconds")

    denoisedfft, prop = denoise(resfft)

    print("Number of non-zeros:", len(image) *
          len(image[0])*prop, "Fraction of original:", prop)

    
    denoisedimage = inv_fft2(denoisedfft)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax1.title.set_text('Original Image')
    ax2.imshow(denoisedimage.real)
    ax2.title.set_text('Denoised Image')
    plt.show()


def mode3():

    array = pad_image(image)

    fft = fft2(np.float32(array))
    zerosix = copy.deepcopy(fft)
    onesix = copy.deepcopy(fft)
    twosix = copy.deepcopy(fft)
    threesix = copy.deepcopy(fft)
    foursix = copy.deepcopy(fft)
    fivesix = copy.deepcopy(fft)

    list = []
    for x in range(len(fft)):
        for y in range(len(fft[0])):
            list.append(fft[x][y])
    list.sort()

    compressionPercentage = 0
    bufferPercentage = 1-compressionPercentage
    lowCut = list[int(len(list)*0.5*bufferPercentage)]
    highCut = list[int(len(list)*(compressionPercentage+0.5*bufferPercentage))]

    nonzerozero = 0
    for x in range(len(zerosix)):
        for y in range(len(zerosix[0])):
            if(zerosix[x][y] > lowCut and zerosix[x][y] < highCut):
                zerosix[x][y] = 0
            else:
                nonzerozero += 1

    compressionPercentage = 1.0/6.0
    bufferPercentage = 1-compressionPercentage
    lowCut = list[int(len(list)*0.5*bufferPercentage)]
    highCut = list[int(len(list)*(compressionPercentage+0.5*bufferPercentage))]
    nonzeroone = 0
    for x in range(len(onesix)):
        for y in range(len(onesix[0])):
            if(onesix[x][y] > lowCut and onesix[x][y] < highCut):
                onesix[x][y] = 0
            else:
                nonzeroone += 1

    compressionPercentage = 2.0/6.0
    bufferPercentage = 1-compressionPercentage
    lowCut = list[int(len(list)*0.5*bufferPercentage)]
    highCut = list[int(len(list)*(compressionPercentage+0.5*bufferPercentage))]
    nonzerotwo = 0
    for x in range(len(twosix)):
        for y in range(len(twosix[0])):
            if(twosix[x][y] > lowCut and twosix[x][y] < highCut):
                twosix[x][y] = 0
            else:
                nonzerotwo += 1

    compressionPercentage = 3.0/6.0
    bufferPercentage = 1-compressionPercentage
    lowCut = list[int(len(list)*0.5*bufferPercentage)]
    highCut = list[int(len(list)*(compressionPercentage+0.5*bufferPercentage))]
    nonzerothree = 0
    for x in range(len(threesix)):
        for y in range(len(threesix[0])):
            if(threesix[x][y] > lowCut and threesix[x][y] < highCut):
                threesix[x][y] = 0
            else:
                nonzerothree += 1

    compressionPercentage = 4.0/6.0
    bufferPercentage = 1-compressionPercentage
    lowCut = list[int(len(list)*0.5*bufferPercentage)]
    highCut = list[int(len(list)*(compressionPercentage+0.5*bufferPercentage))]
    nonzerofour = 0
    for x in range(len(foursix)):
        for y in range(len(foursix[0])):
            if(foursix[x][y] > lowCut and foursix[x][y] < highCut):
                foursix[x][y] = 0
            else:
                nonzerofour += 1

    compressionPercentage = 5.0/6.0
    bufferPercentage = 1-compressionPercentage
    lowCut = list[int(len(list)*0.5*bufferPercentage)]
    highCut = list[int(len(list)*(compressionPercentage+0.5*bufferPercentage))]
    nonzerofive = 0
    for x in range(len(fivesix)):
        for y in range(len(fivesix[0])):
            if(fivesix[x][y] > lowCut and fivesix[x][y] < highCut):
                fivesix[x][y] = 0
            else:
                nonzerofive += 1

    ci0 = inv_fft2(zerosix)
    ci1 = inv_fft2(onesix)
    ci2 = inv_fft2(twosix)
    ci3 = inv_fft2(threesix)
    ci4 = inv_fft2(foursix)
    ci5 = inv_fft2(fivesix)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.imshow(ci0.real)
    ax2.imshow(ci1.real)
    ax3.imshow(ci2.real)
    ax4.imshow(ci3.real)
    ax5.imshow(ci4.real)
    ax6.imshow(ci5.real)
    ax1.title.set_text('0 compression'+str(nonzerozero))
    ax2.title.set_text('1/6 compression'+str(nonzeroone))
    ax3.title.set_text('2/6 compression'+str(nonzerotwo))
    ax4.title.set_text('3/6 compression'+str(nonzerothree))
    ax5.title.set_text('4/6 compression'+str(nonzerofour))
    ax6.title.set_text('5/6 compression'+str(nonzerofive))
    plt.show()


def mode3Experiments(compressionPercentage):
    array = pad_image(image)

    fft = fft2(np.float32(array))
    uniform = fft.copy()
    highPercentage = fft.copy()
    mixedPercentage = fft.copy()
    nonzeroU = 0
    for x in range(len(uniform)):
        for y in range(len(uniform[0])):
            if(random.random() < compressionPercentage):
                uniform[x][y] = 0
            else:
                nonzeroU = nonzeroU+1
    list = []
    for x in range(len(highPercentage)):
        for y in range(len(highPercentage[0])):
            list.append(highPercentage[x][y])
    list.sort()
    cutoff = list[int(len(list)*compressionPercentage)]
    nonZeroH = 0
    for x in range(len(highPercentage)):
        for y in range(len(highPercentage[0])):
            if(highPercentage[x][y] < cutoff):
                highPercentage[x][y] = 0
            else:
                nonZeroH += 1
    bufferPercentage = 1-compressionPercentage
    lowCut = list[int(len(list)*0.5*bufferPercentage)]
    highCut = list[int(len(list)*(compressionPercentage+0.5*bufferPercentage))]
    nonZeroM = 0
    for x in range(len(mixedPercentage)):
        for y in range(len(mixedPercentage[0])):
            if(mixedPercentage[x][y] > lowCut and mixedPercentage[x][y] < highCut):
                mixedPercentage[x][y] = 0
            else:
                nonZeroM += 1

    compressedImage = inv_fft2(uniform)
    cI2 = inv_fft2(highPercentage)
    cI4 = inv_fft2(mixedPercentage)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.imshow(image)

    ax1.title.set_text('Original Image')
    ax2.imshow(compressedImage.real)
    ax3.imshow(cI2.real)
    ax4.imshow(cI4.real)
    ax2.title.set_text('Randomnly compressed image'+str(nonzeroU))
    ax3.title.set_text("Keep high values"+str(nonZeroH))
    ax4.title.set_text("Reject middle values"+str(nonZeroM))
    plt.show()


def mode4():
    dftsizes = [32,64,128]
    fftsizes = [32,64,128,256,512,1024]
    exectimesfft = []
    exectimesdft = []
    avgfft = []
    varfft = []
    avgdft = []
    vardft = []

    num_runs = 10

    for i in range(len(dftsizes)):
        exectimesdft.append([])
        
        for run in range(num_runs):
            arr = np.random.uniform(0.0, 255.0, (dftsizes[i], dftsizes[i]))

            startTime = time.perf_counter()
            dft2_naive(arr)
            endTime = time.perf_counter()
            execTime = endTime - startTime
            exectimesdft[i].append(execTime)
        
        avgdft.append(np.mean(np.array(exectimesdft[i])))
        vardft.append(np.var(np.array(exectimesdft[i])))  
        print("Avg DFT", avgdft[i])
        print("Var DFT", vardft[i])

    for i in range(len(fftsizes)):
        exectimesfft.append([])
        
        for run in range(num_runs):
            arr = np.random.uniform(0.0, 255.0, (fftsizes[i], fftsizes[i]))
            startTime = time.perf_counter()
            fft2(arr)
            endTime = time.perf_counter()
            execTime = endTime - startTime

            exectimesfft[i].append(execTime)
        
        avgfft.append(np.mean(np.array(exectimesfft[i])))
        varfft.append(np.var(np.array(exectimesfft[i])))
        print("Avg FFT", avgfft[i])
        print("Var FFT", varfft[i])

    # fig, (ax1, ax2) = plt.subplots(2)
    # ax1.plot(sizes, avgfft, 'o-')
    # ax1.title.set_text('FFT2 execution time')
    
    # ax2.plot(sizes, avgdft, 'o-')
    # ax2.title.set_text('DFT2 execution time')
    # for i in range(len(sizes)):
    #     print("Adding error bars")
    #     ax1.errorbar(sizes[i], avgfft[i], yerr=varfft[i])
    #     ax2.errorbar(sizes[i], avgdft[i], yerr=vardft[i])

    plt.subplot(1,1,1)
    plt.plot(fftsizes, avgfft, color='b', label="fft")
    plt.plot(dftsizes, avgdft, color='g', label='dft')
    plt.xlabel("Size of row and col")
    plt.ylabel("Execution time (seconds)")
    plt.title("Execution time of fft and dft in 2-dimensions depending on the size of the square matrix")
    plt.legend()
    for i in range(len(dftsizes)):
        plt.errorbar(dftsizes[i], avgdft[i], yerr=2*vardft[i])
    for i in range(len(fftsizes)):
        plt.errorbar(fftsizes[i], avgfft[i], yerr=2*varfft[i])
        
    plt.show()
        

# We can change the sizes of the arrays to test multiple scenarios
def test():
    # opts()

    test_1d = np.random.uniform(low=0.0, high=255.0, size=8)
    test_2d = np.random.rand(512, 1024)
    test2d(test_2d)
    test1d(test_1d)


def main():
    opts()

    if mode == 1:
        mode1()
    if mode == 2:
        mode2()
    if mode == 3:
        mode3()
    if mode == 4:
        mode4()


main()