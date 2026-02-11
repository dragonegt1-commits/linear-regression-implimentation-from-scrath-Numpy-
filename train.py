import numpy as np
#Data
x= np.array([[1,2],
			[2,3],
			[3,4]])
y= np.array([[5],
			 [7],
			 [9]])
w= np.random.randn(2,1) #weight
b= 0.0 #bias
lr= 0.01 #learning rate
epoches= 20
n= len(x) # Used for MSE and bias gradient

for epoch in range(epoches):
	#Prediction
	y_prediction = x@w + b
	#Loss(Mean squared loss)
	mse= np.mean((y-y_prediction)**2)
	#Gradient
	dw = (-2/n)*x.T @ (y-y_prediction) #Weight gradient
	db = (-2/n)*np.sum(y-y_prediction)
	#Update
	w-=lr*dw
	b-=lr*db
	print(f'Epoch {epoch+1}: Loss= {mse:.4f}')





