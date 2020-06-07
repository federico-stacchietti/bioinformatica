from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import pandas as pd
from multiprocessing import cpu_count

#---------------------------------------

from sklearn.impute import KNNImputer

def imputation(epigenomes:pd.DataFrame)->pd.DataFrame: #DA RIVEDERE?
    #if detect_NaN(epigenomes) > 0:
    epigenomes = pd.DataFrame(
        KNNImputer(n_neighbors=5).fit_transform(epigenomes.values),
        columns=epigenomes.columns,
        index=epigenomes.index
    )
    return epigenomes

#------------------------------------

def get_features_filter(X, y)->BorutaPy:

    forest = RandomForestClassifier(n_jobs=cpu_count(), class_weight='balanced', max_depth=5)

    boruta_selector = BorutaPy(
        forest,
        n_estimators='auto',
        verbose=2,
        alpha=0.05, # p_value
        max_iter=200,           #10, # In practice one would run at least 100-200 times
        #da testare il numero di iterazioni
        random_state=42
    )
    boruta_selector.fit(X.values, y)#.values.ravel())
    return boruta_selector

#main

# data2 = [[100000, 1, 2, 3], [20, 21, 19, 18], [1, 1, 1, 200000], [20, 44, 60, 1000], [20, 45, 70, 1020]]
# df2 = pd.DataFrame(data2, columns=['Name', 'Age', 'Job', 'Date'])
#
# labels = pd.DataFrame([1,0,1,0,1])

#-----------------------------


#Materiale Test



#epigenoma

epigenoma = pd.read_csv ('/home/willy/HEK293.csv')

epigenoma = imputation(epigenoma)

print(type(epigenoma))



#etichette

etichette_file = pd.read_csv('/home/willy/promoters.bed')

for region,x in etichette_file.items():
    if region == 'HEK293':
        etichette = x.values.ravel()
print(type(etichette))





#fefefef

#Chiamo la funzione
X_filtered = get_features_filter(epigenoma, etichette).transform(epigenoma.values)
print(X_filtered)
















#FLAVIO
# # define random forest classifier, with utilising all cores and
# # sampling in proportion to y labels
# forest = RandomForestClassifier(n_jobs=cpu_count(), class_weight='balanced', max_depth=5)
#
# # define Boruta feature selection method
# feat_selector = BorutaPy(forest, n_estimators='auto',
#                          verbose=2,
#                          alpha=0.05,  # p_value
#                          max_iter=10,  # In practice one would run at least 100-200 times
#                          random_state=42
#                          )
#
# # find all relevant features
# feat_selector.fit(epigenoma.values, etichette)
#
# # check selected features
# feat_selector.support_
#
# # check ranking of features
# feat_selector.ranking_
#
# # call transform() on X to filter it down to selected features
# X_filtered = feat_selector.transform(epigenoma.values)
# print(X_filtered)

