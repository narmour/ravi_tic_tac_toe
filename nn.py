import numpy as np
import struct
from matplotlib import pyplot
import matplotlib as mpl
import random


def getData():
    #get traini 
    with open('./train-labels.idx1-ubyte','rb') as fl:
        magic,num = struct.unpack(">II",fl.read(8))
        label = np.fromfile(fl,dtype=np.int8)
    with open('./train-images.idx3-ubyte','rb') as fi:
        magic, num, rows, cols = struct.unpack(">IIII",fi.read(16))
        image = np.fromfile(fi,dtype=np.uint8).reshape(len(label),rows,cols)
    get_image = lambda idx: (label[idx],image[idx])
    ret = []
    print("label: ",label)
    for i in range(len(label)):
        ret.append(get_image(i))
    return ret

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show() 


def extractFeatures(data):
    print
    subgrid_values = []
    for image in data:
        #print(image[1])
        #get all 7x7 subgrids in image[1]
        for s in range(16):
            values = []#one subgrid
            #print("subrgrid: ",s,"starts at   row: ",7 * int((s/4))," col: ",7* (s%4))
            for i in range(7):
                for j in range(7):
                    row = i + (7 * int((s/4)))
                    col = j + (7 * (s % 4))
                    #print("row: ",row," col: ",col)
                    values.append(image[1][row][col])
            subgrid_values.append(np.mean(values))
            #print(values,"\n\n")


    return subgrid_values

            

        


def main():
    data = getData()
    print(len(data))
    # 28 by 28 pixel grid
    # each feature is 7x7 subgrid
    training_data = extractFeatures(data[:54000])
    #show(data[0][1])
    print(training_data)

main()
