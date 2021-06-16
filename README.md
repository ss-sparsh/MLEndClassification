# MLEndClassification
Simple ML Classification Task using SVN, KNN, MLP, RFC, Naive Bayes Classifier, Logistic Regression. A deep MLP is also used for the second classification task

### Basic Solution
Intonation Prediction

In this project I have built a model that predicts the intonation of a short audio segment.
The audio dataset was prepared by the batch of MSc Big Data Science at QMUL.

The trainingMLEND.csv consits of 20k rows and 4 columns. Each row corresponds to one of the items in our dataset, and each item is described by four attributes.
1) File ID (audio file)
2) Numeral
3) Participand ID
4) Intonation

Here I have extracted 10 features from the audio signal namely 
1) Power
2) Pitch mean
3) Pitch std
4) Voiced flag
5) Onset
6) MFCC
7) Zero Crossing Rate
8) Spectral Centroid
9) Spectral Rolloff
10) Root Mean Square
    
using librosa library.

I build 6 models which are

1) SVM
2) RandomForest
3) KNN
4) Naive Bayes
5) Logistic Regression
6) MuliLayer Perceptron Classifier

Grid Search is used to find the best parameters of each of the models.
The models take the 10 features as input and try to classify them into the the 4 intonations.

Finally, Getting the highest Accuracy on the Random Forest Classifier model with the accuracy of 54.5% on the validation data with normalised predictors.

### Advanced Solution
Next, I decided to develop a model that only identifies single digits as the accuracy achieved  was much higher than that for a model that would classify the audio into all the numerical classes.

The Approach For Single Digit identification

1. The all 20000 audio files are iterated through and the 10 features are extracted as well as the 10 label classes.
2. The numpy array values are saved on the drive
3. NaN values are removed
4. The numpy arrays are converted to a dataframe for easy of encoding and preprocessing
5. The dataframe is transformed using a Standard Scaler
6. PCA is performed to extract 2 and 4 sets of vectors.
7. Data Visualized to analyse class distribution, correlation and relations between features.
8. Data is split between training and testing set
9. A Deep MLP using keras is trained


#### THE DEEP LEARNING MODEL

I have used keras to develop a deep connected Multilayer Perceptron. It has 6 densely connected layers and 3 dropout layers. The activation function used is Leaky Rectified Linear Unit.
The hyper parameters batch size and epoch was updated based on repeated training. Batch size 16,32,64,128 were used ,with epochs 250,100,70,50 were used
Best results were obtained with batch size = 16 and epochs =250
But in this configuration the model overfit and had a training accuracy of 1.00 and validation accuracy of 0.24
In order to overcome this dropout layers and weight constraints were used, the final training accuracy was 0.3955 and validation accuracy of 0.3684.
