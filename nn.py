import numpy as np
import struct
from matplotlib import pyplot
import matplotlib as mpl
import random
from sklearn.neural_network import MLPClassifier


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
    res = []
    for image in data:
        subgrid_values = []
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
        res.append((subgrid_values,image[0]))


    return res

            

        


def main():
    data = getData()
    # 28 by 28 pixel grid
    # each feature is 7x7 subgrid
    training_data = extractFeatures(data[:54000])
    test_data = extractFeatures(data[54000:])

    #split training_data
    train_features = [t[0] for t in training_data]
    train_labels = [t[1] for t in training_data]

    #make nn
    c = MLPClassifier()
    c.fit(train_features,train_labels)

    #predict on test_data
    test_features = [t[0] for t in test_data]
    test_labels = [t[1] for t in test_data]
    res = c.predict(test_features)

    #check results
    err = 0
    for p in range(len(res)):
        if res[p] != test_labels[p]:
            err+=1
    print("RESULT: ",len(res) - err, "/",len(res), "CORRECT     ", int((float(len(res)) - float(err))/float(len(res))*100),"%")



    







main()
