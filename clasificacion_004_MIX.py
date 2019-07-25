"""
	Clasificación de textos con Python y scikit-learn
	rodrigo@contraculto.com
"""

#   Requerimientos
import pandas # pandas para hacer el dataframe y meter los arreglos con datos
from sklearn import preprocessing, model_selection # para preparar los datos
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # para crear count vectors y TF-IDF vectors con los textos
from sklearn import naive_bayes, linear_model, svm, ensemble # distintos modelos que podemos usar
from sklearn import metrics # para probar el funcionamiento de los modelos

#   Mensaje de inicio
print("\nPrueba de clasificación de textos con Python y scikit-learn")

#   *** 01 Cargar datos
#       Deberían ser una función
#       def cargar_datos(ruta, tipo(archivo o directorio), opcionales(separador para el archivo, tal vez encoding))

#	*** 01.01 Desde un archivo de texto
print("\nCargando datos desde archivo...")

datos_originales = open('datasets/corpus_reviews').read()
etiquetas, textos = [], []
for i, line in enumerate(datos_originales.split("\n")):
	content = line.split()
	etiquetas.append(content[0])
	textos.append(" ".join(content[1:]))

#	***	01.02 Desde una serie de directorios con archivos usando sklearn
#		Cada directorio es una etiqueta, el contenido de cada archivo es el texto de una instancia
#print("\nCargando datos desde directorio...")
#from sklearn.datasets import load_files
#datos_originales = load_files(r"datasets/txt_sentoken")
#etiquetas, textos = datos_originales.target, datos_originales.data

"""
#	02 Crear un DataFrame de Pandas para guardar los datos
datos_entrenamiento = pandas.DataFrame()
datos_entrenamiento['textos'] = textos
datos_entrenamiento['etiquetas'] = etiquetas
"""

#	***	02 Limpiar, estructurar y estandarizar los datos

print("\nLimpiando y estandarizando...")

#		02.01 Limpiar los textos

#		02.02 Dividir los datos en "entrenamiento" y "validación"
textos_entrenamiento, textos_validacion, etiquetas_entrenamiento, etiquetas_validacion = model_selection.train_test_split(textos, etiquetas)

#		02.03 Label encode las etiquetas: pasarlas a un valor numérico para que el modelo las pueda usar para clasificar
encoder = preprocessing.LabelEncoder()
etiquetas = encoder.fit_transform(etiquetas_entrenamiento)

#		02.04 Crear Vector Counts y vectores TF-IDF para los textos

#		CountVectorizer
count_vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vectorizer.fit(textos)

textos_entrenamiento_countvector = count_vectorizer.transform(textos_entrenamiento)
textos_validacion_countvector = count_vectorizer.transform(textos_validacion)

#		Vectores TF-IDF a nivel de palabras
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(textos)

textos_entrenamiento_tfidf =  tfidf_vect.transform(textos_entrenamiento)
textos_validacion_tfidf =  tfidf_vect.transform(textos_validacion)

#	Vectores TF-IDF a nivel de ngrams
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(textos)

textos_entrenamiento_tfidf_ngram =  tfidf_vect_ngram.transform(textos_entrenamiento)
textos_validacion_tfidf_ngram =  tfidf_vect_ngram.transform(textos_validacion)

#	Vectores TF-IDF a nivel de caracteres
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(textos)

textos_entrenamiento_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(textos_entrenamiento) 
textos_validacion_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(textos_validacion)

#   *** 05 Crear y ajustar modelos
#       Entrenar distintos algoritmos de clasificación y porbar cada uno contra el set de validación

print("\nEntrenando modelos...")

clasificador = {}
predicciones = {}

#	05.01 Naive Bayes con Count Vectors
print("\nNaive Bayes con Count Vectors")
# entrenamiento
clasificador["naive_bayes_cv"] = naive_bayes.MultinomialNB()
clasificador["naive_bayes_cv"].fit(textos_entrenamiento_countvector, etiquetas_entrenamiento)
# prueba de predicción
predicciones["naive_bayes_cv"] = clasificador["naive_bayes_cv"].predict(textos_validacion_countvector)
#print(predicciones["naive_bayes_cv"])
print(metrics.accuracy_score(predicciones["naive_bayes_cv"], etiquetas_validacion))


#	05.02 Naive Bayes con Word Level TF IDF Vectors
print("\nNaive Bayes con Word Level TF IDF Vectors")
# entrenamiento
clasificador["naive_bayes_tfidf"] = naive_bayes.MultinomialNB()
clasificador["naive_bayes_tfidf"].fit(textos_entrenamiento_tfidf, etiquetas_entrenamiento)
# prueba de predicción
predicciones["naive_bayes_tfidf"] = clasificador["naive_bayes_tfidf"].predict(textos_validacion_tfidf)
#print(predicciones["naive_bayes_tfidf"])
print(metrics.accuracy_score(predicciones["naive_bayes_tfidf"], etiquetas_validacion))


#	05.03 Naive Bayes con Ngram Level TF IDF Vectors
print("\nNaive Bayes con Ngram Level TF IDF Vectors")
# entrenamiento
clasificador["naive_bayes_tfidf_ngram"] = naive_bayes.MultinomialNB()
clasificador["naive_bayes_tfidf_ngram"].fit(textos_entrenamiento_tfidf_ngram, etiquetas_entrenamiento)
# prueba de predicción
predicciones["naive_bayes_tfidf_ngram"] = clasificador["naive_bayes_tfidf_ngram"].predict(textos_validacion_tfidf_ngram)
#print(predicciones["naive_bayes_tfidf_ngram"])
print(metrics.accuracy_score(predicciones["naive_bayes_tfidf_ngram"], etiquetas_validacion))

#	05.04 Naive Bayes con Character Level TF IDF Vectors
print("\nNaive Bayes con Character Level TF IDF Vectors")
# entrenamiento
clasificador["naive_bayes_tfidf_ngram_chars"] = naive_bayes.MultinomialNB()
clasificador["naive_bayes_tfidf_ngram_chars"].fit(textos_entrenamiento_tfidf_ngram_chars, etiquetas_entrenamiento)
# prueba de predicción
predicciones["naive_bayes_tfidf_ngram_chars"] = clasificador["naive_bayes_tfidf_ngram_chars"].predict(textos_validacion_tfidf_ngram_chars)
#print(predicciones["naive_bayes_tfidf_ngram_chars"])
print(metrics.accuracy_score(predicciones["naive_bayes_tfidf_ngram_chars"], etiquetas_validacion))

#	***	***

#	05.05 Linear classifier con Count Vectors
print("\nLinear classifier con Count Vectors")
# entrenamiento
clasificador["linear_model_cv"] = linear_model.LogisticRegression(solver='lbfgs',max_iter=2000)
clasificador["linear_model_cv"].fit(textos_entrenamiento_countvector, etiquetas_entrenamiento)
# prueba de predicción
predicciones["linear_model_cv"] = clasificador["linear_model_cv"].predict(textos_validacion_countvector)
#print(predicciones["linear_model_cv"])
print(metrics.accuracy_score(predicciones["linear_model_cv"], etiquetas_validacion))


#	05.06 Linear classifier con Word Level TF IDF Vectors
print("\nLinear classifier con Word Level TF IDF Vectors")
# entrenamiento
clasificador["linear_model_tfidf"] = linear_model.LogisticRegression(solver='lbfgs',max_iter=2000)
clasificador["linear_model_tfidf"].fit(textos_entrenamiento_tfidf, etiquetas_entrenamiento)
# prueba de predicción
predicciones["linear_model_tfidf"] = clasificador["linear_model_tfidf"].predict(textos_validacion_tfidf)
#print(predicciones["linear_model_tfidf"])
print(metrics.accuracy_score(predicciones["linear_model_tfidf"], etiquetas_validacion))


#	05.07 Linear classifier con Ngram Level TF IDF Vectors
print("\nLinear classifier con Ngram Level TF IDF Vectors")
# entrenamiento
clasificador["linear_model_tfidf_ngram"] = linear_model.LogisticRegression(solver='lbfgs',max_iter=2000)
clasificador["linear_model_tfidf_ngram"].fit(textos_entrenamiento_tfidf_ngram, etiquetas_entrenamiento)
# prueba de predicción
predicciones["linear_model_tfidf_ngram"] = clasificador["linear_model_tfidf_ngram"].predict(textos_validacion_tfidf_ngram)
#print(predicciones["linear_model_tfidf_ngram"])
print(metrics.accuracy_score(predicciones["linear_model_tfidf_ngram"], etiquetas_validacion))


#	05.08 Linear classifier con Character Level TF IDF Vectors
print("\nLinear classifier con Character Level TF IDF Vectors")
# entrenamiento
clasificador["linear_model_tfidf_ngram_chars"] = linear_model.LogisticRegression(solver='lbfgs',max_iter=2000)
clasificador["linear_model_tfidf_ngram_chars"].fit(textos_entrenamiento_tfidf_ngram_chars, etiquetas_entrenamiento)
# prueba de predicción
predicciones["linear_model_tfidf_ngram_chars"] = clasificador["linear_model_tfidf_ngram_chars"].predict(textos_validacion_tfidf_ngram_chars)
#print(predicciones["linear_model_tfidf_ngram_chars"])
print(metrics.accuracy_score(predicciones["linear_model_tfidf_ngram_chars"], etiquetas_validacion))

#	***	***

#	05.05 Support Vector Machine con Count Vectors
print("\nSupport Vector Machine con Count Vectors")
# entrenamiento
clasificador["svm_cv"] = svm.SVC(gamma='scale')
clasificador["svm_cv"].fit(textos_entrenamiento_countvector, etiquetas_entrenamiento)
# prueba de predicción
predicciones["svm_cv"] = clasificador["svm_cv"].predict(textos_validacion_countvector)
#print(predicciones["svm_cv"])
print(metrics.accuracy_score(predicciones["svm_cv"], etiquetas_validacion))


#	05.06 Support Vector Machine con Word Level TF IDF Vectors
print("\nSupport Vector Machine con Word Level TF IDF Vectors")
# entrenamiento
clasificador["svm_tfidf"] = svm.SVC(gamma='scale')
clasificador["svm_tfidf"].fit(textos_entrenamiento_tfidf, etiquetas_entrenamiento)
# prueba de predicción
predicciones["svm_tfidf"] = clasificador["svm_tfidf"].predict(textos_validacion_tfidf)
#print(predicciones["svm_tfidf"])
print(metrics.accuracy_score(predicciones["svm_tfidf"], etiquetas_validacion))


#	05.07 Support Vector Machine con Ngram Level TF IDF Vectors
print("\nSupport Vector Machine con Ngram Level TF IDF Vectors")
# entrenamiento
clasificador["svm_tfidf_ngram"] = svm.SVC(gamma='scale')
clasificador["svm_tfidf_ngram"].fit(textos_entrenamiento_tfidf_ngram, etiquetas_entrenamiento)
# prueba de predicción
predicciones["svm_tfidf_ngram"] = clasificador["svm_tfidf_ngram"].predict(textos_validacion_tfidf_ngram)
#print(predicciones["svm_tfidf_ngram"])
print(metrics.accuracy_score(predicciones["svm_tfidf_ngram"], etiquetas_validacion))


#	05.08 Support Vector Machine con Character Level TF IDF Vectors
print("\nSupport Vector Machine con Character Level TF IDF Vectors")
# entrenamiento
clasificador["svm_tfidf_ngram_chars"] = svm.SVC(gamma='scale')
clasificador["svm_tfidf_ngram_chars"].fit(textos_entrenamiento_tfidf_ngram_chars, etiquetas_entrenamiento)
# prueba de predicción
predicciones["svm_tfidf_ngram_chars"] = clasificador["svm_tfidf_ngram_chars"].predict(textos_validacion_tfidf_ngram_chars)
#print(predicciones["svm_tfidf_ngram_chars"])
print(metrics.accuracy_score(predicciones["svm_tfidf_ngram_chars"], etiquetas_validacion))


#	05.09 Random Forest con Count Vectors
print("\nRandom Forest con Count Vectors")
# entrenamiento
clasificador["random_forest_cv"] = ensemble.RandomForestClassifier(n_estimators=100)
clasificador["random_forest_cv"].fit(textos_entrenamiento_countvector, etiquetas_entrenamiento)
# prueba de predicción
predicciones["random_forest_cv"] = clasificador["random_forest_cv"].predict(textos_validacion_countvector)
#print(predicciones["random_forest_cv"])
print(metrics.accuracy_score(predicciones["random_forest_cv"], etiquetas_validacion))


#	05.10 Random Forest con Word Level TF IDF Vectors
print("\nRandom Forest con Word Level TF IDF Vectors")
# entrenamiento
clasificador["random_forest_tfidf"] = ensemble.RandomForestClassifier(n_estimators=100)
clasificador["random_forest_tfidf"].fit(textos_entrenamiento_tfidf, etiquetas_entrenamiento)
# prueba de predicción
predicciones["random_forest_tfidf"] = clasificador["random_forest_tfidf"].predict(textos_validacion_tfidf)
#print(predicciones["random_forest_tfidf"])
print(metrics.accuracy_score(predicciones["random_forest_tfidf"], etiquetas_validacion))


#	05.11 Random Forest con Ngram Level TF IDF Vectors
print("\nRandom Forest con Ngram Level TF IDF Vectors")
# entrenamiento
clasificador["random_forest_tfidf_ngram"] = ensemble.RandomForestClassifier(n_estimators=100)
clasificador["random_forest_tfidf_ngram"].fit(textos_entrenamiento_tfidf_ngram, etiquetas_entrenamiento)
# prueba de predicción
predicciones["random_forest_tfidf_ngram"] = clasificador["random_forest_tfidf_ngram"].predict(textos_validacion_tfidf_ngram)
#print(predicciones["random_forest_tfidf_ngram"])
print(metrics.accuracy_score(predicciones["random_forest_tfidf_ngram"], etiquetas_validacion))


#	05.12 Random Forest con Character Level TF IDF Vectors
print("\nRandom Forest con Character Level TF IDF Vectors")
# entrenamiento
clasificador["random_forest_tfidf_ngram_chars"] = ensemble.RandomForestClassifier(n_estimators=100)
clasificador["random_forest_tfidf_ngram_chars"].fit(textos_entrenamiento_tfidf_ngram_chars, etiquetas_entrenamiento)
# prueba de predicción
predicciones["random_forest_tfidf_ngram_chars"] = clasificador["random_forest_tfidf_ngram_chars"].predict(textos_validacion_tfidf_ngram_chars)
#print(predicciones["random_forest_tfidf_ngram_chars"])
print(metrics.accuracy_score(predicciones["random_forest_tfidf_ngram_chars"], etiquetas_validacion))