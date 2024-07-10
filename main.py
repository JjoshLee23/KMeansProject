import pandas as pd
from Kvalue import *
from scipy.cluster.vq import kmeans, kmeans2
from scipy.cluster.vq import vq
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import whiten
from sklearn.model_selection import train_test_split
#imported all of the necessary libaries

data = pd.read_csv('student-mat.csv',sep=";")
data=data.drop(columns=['G1','G2'])

data['sex']=data['sex'].apply(lambda x: 1 if x=='M' else 0)
data['activities']=data['activities'].apply(lambda x: 1 if x=='yes' else 0)
data['romantic']=data['romantic'].apply(lambda x: 1 if x=='yes' else 0)
features=['sex','age','studytime','failures','activities','romantic','absences']
subset=data[features]
subset=whiten(subset)


subset= pd.DataFrame(subset)


target=data['G3']

X_train,X_test,Y_train,Y_test=train_test_split(subset,target)

accuracyscores=[] #list of accuracy scores
for k in range(1,21):
    prediction=[] #list storing the predictions for the x test set
    correct=0
    label=0
    codebook,_=kmeans2(X_train,k,minit='++') #creates a codebook that stores the centroids found on the training data based on the # of clusters
    dictionary=findvaluefork(k,codebook,X_train,Y_train)#utilitzes the findvalueork function that returns a dictionary.
    encoding3,_=vq(X_test,codebook) #assigns the X_test data to certain clusters using the clusters computed with the training data.
    for key, value in dictionary.items():
        for cluster in encoding3:
            if(key==cluster):
                label=dictionary[key]
                prediction.append(label)
    for j in range(len(Y_test)):
            if prediction[j] == Y_test.iloc[j]: #I counted each time the prediction would be true to the actual values.
                correct += 1
    print("The accuracy score using",k,"clusters:", (correct / len(Y_test)))
    accuracyscores.append(correct/len(Y_test)) #appends each accuracy score for later use in the graph
    
#Visualization
kvalues=np.linspace(1,20,20)
plt.xticks(np.arange(1, 21, step=1))
plt.plot(kvalues,accuracyscores)
plt.plot(kvalues,accuracyscores,'o')
plt.xlabel("K-Values(Number of Clusters)")
plt.ylabel("Accuracy scores")
plt.show()

