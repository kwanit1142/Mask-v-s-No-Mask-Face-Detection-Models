# Mask v/s No-Mask Face Detection using Machine Learning Models

## About

![Screenshot (587)](https://user-images.githubusercontent.com/54277039/139411055-e4d385d0-97ba-43d9-b0b7-835e78e17648.png)

We have made our custom dataset consisting of 10,470 images with equal proportions of masked and no-masked people and segregated the same from `self-built-masked-face-recognition-dataset`. Since the masked data was less, image augmentation was incorporated to create an equal proportion of both the classes. We used various classification algorithms and compared their results in this report. We applied the same on the group photos given in the `Real-World-Masked-Face-Dataset-master` and detected the human faces using inbuilt-Cascade classifier (Adaboost based model). The dataset has been split into train and test with test size of 0.5

## Dependencies

1. OpenCV
2. Scikit-Learn
3. Numpy
4. Pandas
5. Scipy
6. Skimage
7. Matplotlib
8. Seaborn
9. Os

## Models

1. `Multilayer Perceptron` :- MLP is a feedforward Neural Network which uses backpropagation to update weights and improve the results. MLP was applied on the training dataset with custom architecture (first hidden layer consisted of 100 neurons, second hidden layer consisted of 50 neurons and third with 25 neurons, accompanied by ReLU activation function with tolerance value of 10e(-6)).

2. `Random Forest Classifier` :- Random Forest Classifiers use boosting ensemble methods to train upon various decision trees and produce aggregated results.It is one of the most used machine learning algorithms. It was applied on the training dataset with n_estimators as 100.

3. `Decision tree classifier` :- A Decision Tree is a simple representation for classifying examples. It is a Supervised Machine Learning where the data is continuously split according to a certain parameter. It was applied on the training dataset with default configuration.

4. `Logistic Regression` :- Logistic Regression model is widely used for binary classification and hence is well suited for classification into mask vs no mask with parameter n_jobs= -1 so that all the processors in CPU are used to speed up the code.

5. `Gaussian Naive Bayes` :- GNB is a type of Naive Bayes classifier that assumes that the distribution of data is gaussian and classifies data based on this assumption. It was applied on the training dataset with default configuration.

6. `KNN (k - nearest neighbors)` :- KNN are supervised algorithms which classify on the basis of distance from similar points.Here k is the number of nearest neighbors to be considered in the majority voting process with parameters n_neighbor=3 and n_jobs= -1 so that all the processors in CPU are used to speed up the code.

7. `Support Vector Machine` :- In SVM , data points are plotted into n-dimensional graphs which are then classified by drawing hyperplanes. The Gaussian Radial Basis Function was used as the kernel and `c` was taken as 30.

## Results

The models implemented were evaluated using techniques like :- 

1. Classification Report (Precision , Recall , F1-Score and Support) and Confusion Matrix , Accuracy Score and Cross-Validation Scores.

![Screenshot (588)](https://user-images.githubusercontent.com/54277039/139411075-e834ccde-347f-4a97-a06c-c3cef4ef54cd.png)

2. ROC plots

![Screenshot (1285)](https://user-images.githubusercontent.com/54277039/174298233-f1a96353-3d88-406e-986e-915236c3559e.png)
![Screenshot (1286)](https://user-images.githubusercontent.com/54277039/174298239-c9697117-19d0-4eb6-b077-64cfa73fd75b.png)
![Screenshot (1287)](https://user-images.githubusercontent.com/54277039/174298249-db81d861-560a-4232-99e4-0e61bb315728.png)

## Implementation on Group Photos and Conclusion

![Screenshot (589)](https://user-images.githubusercontent.com/54277039/139411091-67302bc3-aafc-4303-8ab9-57807ded75b5.png)

The table shows that all classifiers had accuracies in the range of `60-70%`. Out of all the models, we saw that `SVM and MLP` outperformed others with the accuracy of `71%` because in MLP, every weight associated with perceptrons captures every feature of an image and in SVM, we saw that gaussian performed better than others (linear, polynomial and sigmoid) because it normalises the data distribution which enabled us to deal with outliers. The Gaussian Naive Bayes did not perform well when compared to the others since in naive bayes, it is assumed that all the features are independent which was not the case.

## References

1. https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
2. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
3. https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
4. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
5. https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
6. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
7. https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
8. https://www.pyimagesearch.com/2021/04/12/opencv-haar-cascades/
9. https://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.random_noise
10. https://towardsdatascience.com/data-augmentation-techniques-in-python-f216ef5eed69

# Contributors

1. [Soumya Vaish](https://github.com/Saumya0206)
2. [Kwanit Gupta (me)](https://github.com/kwanit1142)
