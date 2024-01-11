import numpy as np
import math
import scipy as sp
import pandas as pd
import data_cleaning
from sklearn.metrics import log_loss, accuracy_score , confusion_matrix
from sklearn.model_selection import KFold
import os
import warning_tensorflow
import random
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


# Calculation for logg loss
def split_data(input,target):

    split_size = int(input.shape[0]*0.7)

    train_x,test_x = input[:split_size],input[split_size:]
    train_y,test_y = target[:split_size],target[split_size:]

    return train_x,test_x,train_y,test_y

def plot_data(history,fold):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Testing Loss for Fold {fold+1}')
    plt.legend()
    plt.show()
    plt.savefig(f"FIGURES/traintestloss{fold+1}.png")
def nnet(train_x,test_x,train_y,test_y):
    tf.keras.utils.set_random_seed(1)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16,activation="relu",input_shape  = (train_x.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8,activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1,activation="sigmoid")
    ])
    # Plotting the model 
    tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
 
# Compilation of model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
              loss="binary_crossentropy",
              metrics=['accuracy'])

   
    prediction_history = model.fit(train_x,train_y,epochs=50,batch_size=4,verbose =0,validation_data=(test_x,test_y))
    weights =[]
    weights.append(model.layers[0].weights)


    train_predictions = model.predict(train_x,verbose=0)
    train_predictions_classes = (train_predictions > 0.5).astype(int)
    train_accuracy = accuracy_score(train_y, train_predictions_classes)
    
    test_predictions = model.predict(test_x,verbose=0)
    test_predictions_classes = (test_predictions > 0.5).astype(int)
    test_accuracy = accuracy_score(test_y, test_predictions_classes)

    print(tf.math.confusion_matrix(test_predictions,test_y))

    
    return train_accuracy, test_accuracy,prediction_history,weights

    #train_loss, train_acc = model.evaluate(train_x, train_y, verbose=1)
    #test_loss, test_acc = model.evaluate(test_x, test_y, verbose=1)

    
    #loss_rate = accuracy_score(train_y,prediction)


    #return prediction_history#train_loss,train_acc,test_loss,test_acc
if __name__ == "__main__":

    tf = warning_tensorflow.import_tensorflow()
    
    data = data_cleaning.rand_data(data_cleaning.read_data())

    inputs = data_cleaning.scale_value(data_cleaning.input_array(data = data)) # Scaling the input value

    target = data_cleaning.target_array(data= data).reshape(1372)  # Assigning the target value and reshaping to 1-D 

   

  #  trainx,testx,trainy,testy= split_data(input=inputs,target=target)

    #result = nnet(train_x=trainx,test_x=testx,train_y=trainy,test_y=testy)
    np.random.seed(123)
    #print(result)
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True)
    fold_train_accuracies = []
    fold_test_accuracies = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(inputs)):
        trainx, testx = inputs[train_index], inputs[test_index]
        trainy, testy = target[train_index], target[test_index]

        train_accuracy, test_accuracy ,history,weights= nnet(train_x=trainx, test_x=testx, train_y=trainy, test_y=testy)
        plot_data(history=history,fold=fold)
        print(f"Weights for 1st layer:{weights[0][0]}")
       
        
        fold_train_accuracies.append(train_accuracy)
        fold_test_accuracies.append(test_accuracy)

        print(f"Fold {fold + 1} - Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")

    avg_train_accuracy = np.mean(fold_train_accuracies)
    avg_test_accuracy = np.mean(fold_test_accuracies)
    print("\nAverage Train Accuracy:", avg_train_accuracy)
    print("Average Test Accuracy:", avg_test_accuracy)

    
    

