import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open('D:/PRANAMI/Dataset/Diabetes/trained_model.sav', 'rb'))

#Input data
input_data = (8, 183, 64, 0, 0, 23.3, 0.672, 32)

#change the inputdata to numpy array
input_data_to_array = np.asarray(input_data)

#reshape the array for one instance
data_reshape = input_data_to_array.reshape(1,-1)

#prediction
pred = loaded_model.predict(data_reshape)
print(pred)

if pred[0] == 0 :
  print("The person is not diabetic")
else :
  print("The person is diabetic")