import pandas as pd 

def read_data():
    data = pd.read_csv("Data/data_banknote_authentication.txt", header=None ,names= ["Variance","Skewness","Curtosis","Image_Entropy","Class"])


    return data


def rand_data(data):
   data = data.sample(frac =1,random_state =1)


   return data.reset_index()




def input_array(data):
    inputs =data[['Variance', 'Skewness','Curtosis','Image_Entropy']].to_numpy()
   
    return inputs

def target_array(data):
   target = data[["Class"]].to_numpy()

   return target






if __name__== "__main__":

 data =  read_data()

 randomised_data = rand_data(data=data)
 
 inputs  = input_array(randomised_data)

 target = target_array(randomised_data)