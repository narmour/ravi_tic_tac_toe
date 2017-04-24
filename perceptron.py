import numpy as np
import struct
from matplotlib import pyplot
import matplotlib as mpl
import random



def getData():
    #get training set
    with open('./train-labels.idx1-ubyte','rb') as fl:
        magic,num = struct.unpack(">II",fl.read(8))
        label = np.fromfile(fl,dtype=np.int8)
    with open('./train-images.idx3-ubyte','rb') as fi:
        magic, num, rows, cols = struct.unpack(">IIII",fi.read(16))
        image = np.fromfile(fi,dtype=np.uint8).reshape(len(label),rows,cols)
    get_image = lambda idx: (label[idx],image[idx])
    ret = []
    for i in range(len(label)):
        ret.append(get_image(i))
    return ret



def pixelDensity(pixels):
    density = 0
    res = (pixels > 0).tolist()
    for r in res:
        for c in r:
            if c == True:
                density+=1

    return density/(28*28)


def symmetry(pixels):
    horiz_flipped = np.fliplr(pixels)
    horiz_mag = np.linalg.norm(horiz_flipped-pixels)
    horiz_mag /= horiz_mag/pixelDensity(pixels)

    vert_flipped = np.flipud(pixels)
    vert_mag = np.linalg.norm(vert_flipped-pixels)
    vert_mag /= vert_mag/pixelDensity(pixels)




    return ((1-horiz_mag),(1-vert_mag))

def intersections(pixels):
    #horizontal intersections
    min_h_int = float('inf')
    max_h_int= -float('inf')
    res = (pixels>0).tolist()
    for r in res:
        v = 0
        for c in r:
            if c:
                v+=1
        if v>0:
            min_h_int = min(min_h_int,v)
            max_h_int = max(max_h_int,v)

    #vertical intersections
    min_v_int = float('inf')
    max_v_int = -float('inf')
    for c in range(len(res)):
        v =0
        for r in range(len(res)):
            if res[r][c]:
                v+=1
    
        if v>0:
            min_v_int = min(min_v_int,v)
            max_v_int = max(max_v_int,v)



    return (min_h_int,max_h_int,min_v_int,max_v_int)





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



def trainWeights(one_training,five_training,onet,fivet,epochs):
    weights = [-1,0,0,0,0,0,0,0]
    training = one_training + five_training
    np.random.shuffle(training)
    best_err = float('inf')
    best_weights = []


    while epochs > 0:
        for t in training:
            label = t[8]
            res = np.dot(t[:8],weights)

            if res >0 and label !=1:
                weights = np.subtract(t[:8],weights).tolist()
            elif res <=0 and label!=5:
                weights = np.add(t[:8],weights).tolist()
        epochs-=1
        #pocket algorithm
        errors = error(weights,onet+fivet)
        if errors < best_err:
            best_err = errors
            best_weights = weights


    print("FINAL WEIGHTS")
    print(weights)

    print("final errors")
    print(best_err)
    return best_weights



def error(weights,test):
    e = 0
    np.random.shuffle(test)
    for t in test:
        label = t[8]
        res = np.dot(t[:8],weights)
        if res >0 and label !=1:
            e+=1
        elif res <=0 and label!=5:
            e+=1
    print(e)
    return e



def main():
    images = getData()

    one_training_samples = [i for i in images if i[0] ==1][:900]
    five_training_samples = [i for i in images if i[0] ==5][:900]

    one_test_samples = [i for i in images if i[0] ==1][900:1000]
    five_test_samples = [i for i in images if i[0] ==5][900:1000]


    

    #extract features to create training data
    one_training_features = []
    five_training_features = []
    for i in range(900):
        pd_1 = pixelDensity(one_training_samples[i][1])
        pd_5 = pixelDensity(five_training_samples[i][1])

        symmetries_1 = symmetry(one_training_samples[i][1])
        symmetries_5 = symmetry(five_training_samples[i][1])

        inter_1 = intersections(one_training_samples[i][1])
        inter_5 = intersections(five_training_samples[i][1])

        one_training_features.append([1,pd_1,symmetries_1[0],symmetries_1[1],inter_1[0],inter_1[1],
            inter_1[2],inter_1[3],one_training_samples[i][0]])

        five_training_features.append([1,pd_5,symmetries_5[0],symmetries_5[1],inter_5[0],inter_5[1],
            inter_5[2],inter_5[3],five_training_samples[i][0]])

        print(five_training_features[i])


    # extract features to create test data
    one_testing_features = []
    five_testing_features = []
    for i in range(100):
        pd_1 = pixelDensity(one_test_samples[i][1])
        pd_5 = pixelDensity(five_test_samples[i][1])

        symmetries_1 = symmetry(one_training_samples[i][1])
        symmetries_5 = symmetry(five_training_samples[i][1])

        inter_1 = intersections(one_test_samples[i][1])
        inter_5 = intersections(five_test_samples[i][1])


        one_testing_features.append([1,pd_1,symmetries_1[0],symmetries_1[1],inter_1[0],
            inter_1[1],inter_1[2],inter_1[3],one_test_samples[i][0]])

        five_testing_features.append([1,pd_5,symmetries_5[0],symmetries_5[1],inter_5[0],
            inter_5[1],inter_5[2],inter_5[3],five_test_samples[i][0]])
        #print(five_training_features[i][:6])


    weights = trainWeights(one_training_features,five_training_features,one_testing_features,five_testing_features,1000)
    #error(weights,five_testing_features+one_testing_features)





    
    



main()
