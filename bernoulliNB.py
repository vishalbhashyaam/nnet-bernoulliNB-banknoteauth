import numpy as np 
import pandas as pd 
import data_cleaning
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB as BNB
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix as cm 
from sklearn.metrics import ConfusionMatrixDisplay as cmd


def plot_data(train_accuracy,test_accuracy):

    
    plt.figure(figsize=(8, 6))
    plt.bar(['Training', 'Testing'], [train_accuracy, test_accuracy], color=['blue', 'green'])
    plt.title('Training and Testing Accuracies')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.show()


def bernoulliNB(train_x,test_x,train_y,test_y):

    model = BNB()
    model.fit(train_x,train_y)
    prediction = model.predict(test_x)
    prediction_train = model.predict(train_x)
    accuracy_train = accuracy_score(prediction_train,train_y)
    accuracy= accuracy_score(prediction,test_y)

    plot_data(accuracy_train,accuracy)

    print(f"Model Accuracy BNB:{round(accuracy*100,3)}%")

    best_params = hyperparameter(train_x,test_x,train_y,test_y,1)

    model_1 = BNB(alpha=best_params)

    model_1.fit(train_x,train_y)

    model_1_prediction = model_1.predict(test_x)

    m1_accuracy = accuracy_score(model_1_prediction,test_y)

    print(f"Model Accuracy after hyperparameter tuning BNB:{round(m1_accuracy*100,3)}%")

   

    model_GNB = GNB()

    model_GNB.fit(train_x,train_y)

    model_GNB_prediction = model_GNB.predict(test_x)

    m2_accuracy = accuracy_score(model_GNB_prediction,test_y)

    print(f"Model Accuracy GNB:{round(m2_accuracy*100,3)}%")

    best_params = hyperparameter(train_x,test_x,train_y,test_y,2)

    model_GNB_2 = GNB(priors=best_params)

    model_GNB_2.fit(train_x,train_y)

    model_2_prediction = model_GNB_2.predict(test_x)

    m2_accuracy = accuracy_score(model_2_prediction,test_y)

    print(f"Model Accuracy GNB after tuning:{round(m2_accuracy*100,3)}%")

    display = cmd(cm(test_y,model_2_prediction))

    display.plot()
    plt.show()

def hyperparameter(train_x,test_x,train_y,test_y,model):

    if model== 1:

        alphas = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}
   

        grid_search = GridSearchCV(BNB(), alphas, cv=10,n_jobs=1, scoring='accuracy')

        grid_search.fit(train_x,train_y)

        best = grid_search.best_params_['alpha']
        

        return best
    
    elif model == 2:
        #param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
        param_grid = {'priors': [None, [0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]]}

        grid_search = GridSearchCV(GNB(), param_grid=param_grid, cv=10,n_jobs=1, scoring='accuracy')

        grid_search.fit(train_x,train_y)

        best = grid_search.best_params_['priors']
        
        return best


    

if __name__ == "__main__":
    
    data = data_cleaning.rand_data(data_cleaning.read_data())

    inputs = data_cleaning.scale_value(data_cleaning.input_array(data = data)) # Scaling the input value

    target = data_cleaning.target_array(data= data).reshape(1372)  # Assigning the target value and reshaping to 1-D 

    split = int(0.7*len(inputs))

    
    trainx,testx = inputs[:split],inputs[split:]
    trainy,testy = target[:split],target[split:]

  #  print(trainy.shape,testy.shape)



    bernoulliNB(train_x=trainx,test_x=testx,train_y=trainy,test_y=testy)