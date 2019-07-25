import pickle 

with open('text_classifier', 'rb') as training_model:  
    model = pickle.load(training_model)