
from scipy.cluster.vq import vq
def findmode(list):#this function is described above but returns the most occuring number in a list.
    dictformode={}
    max=0
    maxvalue=0
    for i in (list):
        if i in dictformode:
            dictformode[i]+=1
        else:
            dictformode[i]=1
            
    for value,counter in dictformode.items():
        if(counter>max):
            max=counter
            maxvalue=value
    return maxvalue

def findvaluefork(k,codebook2,X_train,Y_train):
    newdict={} #creates a new dictionary
    for i in range(X_train.shape[0]): #X_train.shape[0] gets the number of rows.
        encoding2,_ = vq(X_train.iloc[i:i+1,],codebook2)
        if encoding2[0] in newdict:
            newdict[encoding2[0]].append(Y_train.iloc[i])
        else:
            newdict[encoding2[0]]=[]
            newdict[encoding2[0]].append(Y_train.iloc[i])
    for key,value in newdict.items():
        newdict[key]=findmode(value)
    return newdict

