from multiprocessing import cpu_count
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping


def define_models():
    models = {
        'RandomForest': [
            ['random_forest_1',
             dict(
                 n_estimators=20,
                 max_depth=5,
                 criterion='gini',
                 n_jobs=cpu_count()
             )],

        ],

        'NN':

             [['ffnn_1',
              (
                  ([
                       Input(shape=(298, )),
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
