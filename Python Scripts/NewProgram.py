import cv2
import numpy as np
import matplotlib.pyplot as plt
img_size=50
m=60
classes=3
inlayer=np.zeros((m,img_size*img_size))
for i in range(1,m+1):
    path='C:/Users/Jhon/Desktop/ingreso_canvas/'+str(i)+'.jpg'
    img=cv2.imread(path)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2=cv2.resize(gray,(img_size,img_size))
    img1=gray2.flatten()[np.newaxis]
    inlayer[i-1,:]=img1

inlayer=inlayer.astype(float)/255
print inlayer.shape
print inlayer[59,:]

outlayer=np.zeros((m,classes))
outlayer[0:20,:]=np.array([1,0,0])
outlayer[20:40,:]=np.array([0,1,0])
outlayer[40:60,:]=np.array([0,0,1])


(m,n)=inlayer.shape
hidden_layer_size=10
output_layer_size=outlayer.shape[1]
X=np.ones((m,n+1))
X[:,1:n+1]=inlayer

weight_1=np.random.rand(hidden_layer_size,n+1)*2.4-1.2
weight_2=np.random.rand(output_layer_size,hidden_layer_size+1)*2.4-1.2
alpha=0.1
epochs=2000
J=np.zeros((epochs))
#forward propagation
for j in range(0,epochs):
    cost_iter=0
    for i in range(0,m):
        a_1=X[i,:]
        Z_1=1/(1+np.exp(-(np.dot(a_1,weight_1.T))))
        a_2=np.ones((Z_1.shape[0]+1))
        a_2[1:hidden_layer_size+1]=Z_1
        Z_2=1/(1+np.exp(-(np.dot(a_2,weight_2.T))))
        #backpropagation
        e=outlayer[i,:]-Z_2
        grad_k=Z_2*(1-Z_2)*e
        #delta_2=alpha*a_2*grad_k
        delta_2=alpha*np.dot(grad_k[np.newaxis].T,a_2[np.newaxis])
        #grad_j=Z_1*(1-Z_1)*(np.dot(grad_k[np.newaxis],weight_2[0,1:][np.newaxis]))
        grad_j=Z_1*(1-Z_1)*(np.dot(grad_k[np.newaxis],weight_2[:,1:]))
        delta_1=alpha*np.dot(grad_j.T,a_1[np.newaxis])
        weight_1=weight_1+delta_1
        weight_2=weight_2+delta_2
        cost_iter=cost_iter+np.sum(np.square(e));
    J[j]=cost_iter
        

pred=np.zeros(outlayer.shape)
for i in range(0,m):
        a_1=X[i,:]
        Z_1=1/(1+np.exp(-(np.dot(a_1,weight_1.T))))
        a_2=np.ones((Z_1.shape[0]+1))
        a_2[1:hidden_layer_size+1]=Z_1
        pred[i]=1/(1+np.exp(-(np.dot(a_2,weight_2.T))))
print pred
fig, ax = plt.subplots()
t = np.arange(0.0, epochs, 1.0)
ax.set(xlabel='epochs', ylabel='Error',
       title='Neural network')
ax.plot(t, J)
ax.grid()
#fig.savefig("NN.png")
plt.show()


