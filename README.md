# linear-regression-implimentation-from-scrath-Numpy-
# What this project is about 
This project impliment linear regression from scracth using numpy to understand gradient descent optimization and parameter learning without high-level ML libraries.

# Maths used
1)Model equation: y_prediction = XW + b
where X is the input feature.
W is the weight.
b is the bias.

2) Loss fuction: MSE = (1/n)sum(y-y_prediction)**2
3) Gradient: dW = (2/n)*X.T@(y_prediction-y)**2 and db= (2/n)sum(y_prediction - y)

# Details 
1) Shape of X = (3,2)
2) Shape of W = (2,1)
3) lr used is 0.01
4) at lr = 0.1 the model diverges
5) Number of epoches used is 20

