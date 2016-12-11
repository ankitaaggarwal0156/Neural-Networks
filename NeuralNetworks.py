import re
import numpy as np
from numpy import random
#from bigfloat  import *
np.seterr(all='ignore')
alpha =0.1
eL=1.0
Wl1=[]
Wl2=[]
sigL=-1.0
for i in range (96000):
    Wl1.append(random.uniform(-0.1, 0.1))
for i in range (100):
    Wl2.append(random.uniform(-0.1, 0.1))
def read_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer1 = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer1).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer1,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def sigmoid(z): 
    #e = bigfloat.exp(-z,bigfloat.precision(100))
    e =  np.exp(np.longfloat(-z))
    return 1 / (1 + e)

def layer_init(image1, label1):
    global  alpha, eL, Wl1, Wl2, sigL
    deltaL=0.0
    deltaL1 = np.zeros(100)
    image = np.array(image1)
    Wl1 = np.array(Wl1)
    Wl2 = np.array(Wl2)
    outL1 = np.zeros(100)
    count =0
    while(count <1000):
        
        outp=0.0
        count = count +1
        k=0
        for i in range(100):
            p=0.0
            for j in range(960):
                p = p+ image[j] * Wl1[k]
                k=k+1
            o = sigmoid(p)
            outL1[i]= o
            outp= outp + o* Wl2[i]
   
        sigL = sigmoid(outp)
        eL =errorCal(sigL, label1)
        #print "el = ", eL
        if(eL>=-0.04 and eL <=0.04 ):
            break
        #print "sigL", sigL
        
        e= errorCal(sigL, label1)
        deltaL= Delta1(e,sigL)
        outL1 = np.array(outL1)
        for i in range(100): 
            q= Wl2[i] + outL1[i]*deltaL*alpha
            Wl2[i] = q
            
        k=0
        u=0.0
        n=0.0
        for i in range(100):
            u = u+ Wl2[i]*deltaL
    
        for d in range(100):
            n=outL1[d]* (1-outL1[d])
            for i in range(960):   
                de=d+ image[i] *n *u
            deltaL1[d] =de
        
        k=0
        for d in range(100):
            for i in range(960):
                Wl1[i] = Wl1[k] + alpha * image[i] * deltaL1[d] 
                k=k+1
        
   
    return sigL

def errorCal(z, label):
    E =  label -z 
    return E

def Delta1(e,s):
    global alpha
    d =  e* (s - s**2)
    return d


def calMain():
    global label1
    total = 0.0
    correct = 0.0
    with open("./downgesture_train.list") as f:
        for training_image in f.readlines():
            training_image = training_image.strip()
            image1=[]
            im=[]
            im = read_pgm("./"+training_image, byteorder='<')
            for i in range (len(im)):
                for j in range (len (im[0])): 
                    if(im[i][j] >0):
                        image1.append(float(im[i][j]))
                    else:
                        image1.append(0)
                    if 'down' in training_image:
                        label1 = 1
                    else:
                        label1 = 0 
            sig = layer_init(image1, label1) 
            if(sig<0.0001):
                sig = 0.0
            print "Trained on {} with prediction {}" .format(training_image,sig)
            
    print " \n---------------\ntest cases\n \n" 
    
    with open("./downgesture_test.list") as f:
        for test_image in f.readlines():
            total += 1
            test_image = test_image.strip()
            image1=[]
            im=[]
            im = read_pgm("./"+test_image, byteorder='<')
            for i in range (len(im)):
                for j in range (len (im[0])): 
                    if(im[i][j] >0):
                        image1.append(float(im[i][j]))
                    else:
                        image1.append(0)
            b= predictTest(image1)
            if(b<0.2):
                b=0  
            if(b>0.3):
                b=1
            print('{}: {}'.format(test_image, b))
            if (b != 0) == ('down' in test_image):
                correct += 1
    print('correct rate: {}'.format(correct*100 / total))    

            
    
 
def predictTest(image1):
    global Wl1, Wl2
    eL=2.0
    image = np.array(image1)
    k=0
    outp=0.0
    p=0.0
    outL1=np.zeros(100)
    outL1 = np.array(outL1)
    for i in range(100):
        p=0.0
        for j in range(960):
            p = p+ image[j] * Wl1[k]
            k=k+1
        o = sigmoid(p)
        outL1[i]=o
        outp= outp + o* Wl2[i]
   
    sigL = sigmoid(outp)
    eL =errorCal(sigL, label1)   
    return sigL

calMain()