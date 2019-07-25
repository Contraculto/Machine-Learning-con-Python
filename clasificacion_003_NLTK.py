"""
    Clasificación de textos
    Prueba usando Python + NLTK
    rodrigo@contraculto.com
"""

#   Requerimientos
import numpy as np
import re
import nltk
from sklearn.datasets import load_files

import pickle  
from nltk.corpus import stopwords


#   Cargar datos
movie_data = load_files(r"datasets/txt_sentoken")
X, y = movie_data.data, movie_data.target

# Limpieza de textos
#   Stemmer o Lematizador. Se puede cambiar
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()
#from nltk.stem import *
#porter = PorterStemmer()
#lancaster = LancasterStemmer()

documents = []

for sen in range(0, len(X)):  
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

"""
# Vectorizar: Bag of Words
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X = vectorizer.fit_transform(documents).toarray()

# pasar de BoW a TFID
from sklearn.feature_extraction.text import TfidfTransformer  
tfidfconverter = TfidfTransformer()  
X = tfidfconverter.fit_transform(X).toarray()  
"""
# Vectorizar directo a TFID sin BoW
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X = tfidfconverter.fit_transform(documents).toarray()

# Crear sets para training y testing
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Random forest
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
classifier.fit(X_train, y_train)
print(X_test)
y_pred = classifier.predict(X_test)
"""
print("-----")
print((X_test))
print("-----")
print(classifier.predict("hola conchatumadre"))
print("-----")
"""

# Guardar el modelo en un archivo
with open('modelo_NLTK', 'wb') as picklefile:  
    pickle.dump(classifier, picklefile)

# Evaluar (el funcionamiento d)el modelo
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))

documento = "Ayer pasé por tu casa y me tiraste una palta. Qué palta de respeto."
ejemplo = re.sub(r'\W', ' ', str(documento))
ejemplo = re.sub(r'\s+[a-zA-Z]\s+', ' ', ejemplo)
ejemplo = re.sub(r'\^[a-zA-Z]\s+', ' ', ejemplo)
ejemplo = re.sub(r'\s+', ' ', ejemplo, flags=re.I)
ejemplo = re.sub(r'^b\s+', '', ejemplo)
ejemplo = ejemplo.lower()
ejemplo = ejemplo.split()

ejemplo = [stemmer.lemmatize(word) for word in ejemplo]
ejemplo = ' '.join(ejemplo)
print("---")
print(documento)
print(ejemplo)
lol = classifier.predict([documento])
print(classifier.named_steps['classifier'].labels_.inverse_transform(lol))