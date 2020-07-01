from bioinformatica.source.models.libraries import *
from multiprocessing import cpu_count
from bioinformatica.source.commons import *

'''
This functions allows to return a dictionary containing all the models to be trained. It returns a dictionary: every key
is a string that represents a machine learning algorithm, the value for each key is a list that containes one or more models
for a particular algorithm. Every model is represented as a list of the values: a string for the name of the model and a 
tuple containing the parameters to build and train a model.
Neural networks use a tuple of three elements: the paramters to construct the network, compiling parameters and training paramters:
Exemple of use:

models = {
        'RandomForest': [
            ['random_forest_1',
             dict(
                 n_estimators=20,
                 max_depth=5,
                 criterion='gini',
                 n_jobs=cpu_count()
             )],
             
             ['random_forest_2',
             dict(
                 n_estimators=20,
                 max_depth=5,
                 criterion='gini',
                 n_jobs=cpu_count()
             )],
             
             ['random_forest_3',
             dict(
                 n_estimators=20,
                 max_depth=5,
                 criterion='gini',
                 n_jobs=cpu_count()
             )]

        ],

        'NN':

             [['FFNN_1',
              (
                  ([
                       Input(shape=(298,)),
                       Dense(32, activation='relu'),
                       Dense(16, activation='relu'),
                       Dense(1, activation='sigmoid')
                   ], 'FFNN'),

                  dict(
                      optimizer='nadam',
                      loss='binary_crossentropy'
                  ),

                  dict(
                      epochs=10,
                      batch_size=1024,
                      validation_split=0.1,
                      shuffle=True,
                      verbose=True,
                      callbacks=[
                          EarlyStopping(monitor='val_loss', mode='min'),
                      ]
                  )

              )],

             ['FFNN_2',
              (
                  ([
                       Input(shape=(298,)),
                       Dense(32, activation='relu'),
                       Dense(16, activation='relu'),
                       Dense(1, activation='sigmoid')
                   ], 'FFNN'),

                  dict(
                      optimizer='nadam',
                      loss='binary_crossentropy'
                  ),

                  dict(
                      epochs=10,
                      batch_size=1024,
                      validation_split=0.1,
                      shuffle=True,
                      verbose=True,
                      callbacks=[
                          EarlyStopping(monitor='val_loss', mode='min'),
                      ]
                  )

              )]
             ]  
    }
'''


def define_models() -> Dict[str, List]:
    models = {

        'NN':

            [
             ['FFNN_2',
              (
                  ([
                       Input(shape=(200, 4)),
                       Reshape((800, 1)),
                       Flatten(),
                       # Input(shape=(298,)),
                       Dense(32, activation='relu'),
                       Dense(16, activation='relu'),
                       Dense(1, activation='sigmoid')
                   ], 'FFNN'),

                  dict(
                      optimizer='nadam',
                      loss='binary_crossentropy'
                  ),

                  dict(
                      epochs=10,
                      batch_size=1024,
                      validation_split=0.1,
                      shuffle=True,
                      verbose=True,
                      callbacks=[
                          EarlyStopping(monitor='val_loss', mode='min'),
                      ]
                  )

              )]
             ]

    }

    defined_models = {}
    for algorithm in models:
        defined_models[algorithm] = []
        for model in models.get(algorithm):
            defined_models.get(algorithm).append(model)
    return defined_models
