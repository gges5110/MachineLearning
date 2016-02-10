# Assignment 1

This project is for classifying handwritten digits for the MNIST database. The following table explains the functions of each file. To run the code, please put your training and testing set under the same folder as hw1.m. To save the pictures, you will need to give the directory where you want to save the picture.


| File name	            | Function |
| :-------------------- | :------- |
| hw1.m	                | A script that calculates the accuracy of KNN classifier on predicting handwritten digits. It first computes the principal components by the hw1FindEigendigits function and then predict by using fitcknn function provided by Matlab. |
| hw1FindEigendigits.m  |	This function is for calculating the principal components for the training set. It returns a mean column vector and a matrix that contains k eigenvectors of the covariance matrix of A. |
| loadMNISTImages.m	    | Read images from MNIST. It takes the filename and number of data to be read as its arguments. |
| loadMNISTLabels.m	    | Read labels from MNIST. It takes the filename and number of data to be read as its arguments. |
| plot_diagrams.m	      | Plot the result of accuracy against different amount of training points and eigenvectors (features). |
| reconstruct.m	        | Plot the digit picture after reconstruction with different principal components included. |
| savingPicture.m	      | A function for saving plots. |

