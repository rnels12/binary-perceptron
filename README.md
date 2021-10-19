A simple implementation of the perceptron algorithm based on the following lecture: https://www.youtube.com/watch?v=wl7gVvI-HuY&list=PLl8OlHZGYOQ7bkVbuRthEsaLr7bONzbXS&index=5.
The current implementation classifies a digit, in this case 1, from other digits, i.e. binary classification.
However, in principle, the impelementation is extendable to multiclass classification.
The algorithm is coded in perceptron.py and the main function separates the data into two sets, training and test sets, about 80%-20%, respectively.
To classify a different digit, change the value of the variable "digit" in main.py.
The code will output the accuracy of the model measured against the test data set.
The data set, i.e. train.csv, was taken taken from Kaggle.