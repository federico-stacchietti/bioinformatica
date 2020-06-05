from bioinformatica.source.models_builder.models.models_libraries import *


def define_models():

    alberi_decision = [


        dict(criterion="gini",
        max_depth=50,
        random_state=42,
        class_weight="balanced"),



        dict(criterion="gini",
             max_depth=40,
             random_state=42,
             class_weight="balanced"),




        dict(criterion="gini",
             max_depth=50,
             random_state=42,
             class_weight="balanced")
    ]

    NN = [

        (((
        Input(shape=(104,)),
        Dense(1, activation="sigmoid")
    ), "Perceptron"),
    dict(
        optimizer="nadam",
        loss="binary_crossentropy"
    ),
    dict(epochs=1000,
        batch_size=1024,
        validation_split=0.1,
        shuffle=True,
        verbose=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ktqdm(leave_outer=False)
        ])
        ),






        (((
              Input(shape=(300,)),
              Dense(1, activation="sigmoid")
          ), "Perceptron"),
         dict(
             optimizer="nadam",
             loss="binary_crossentropy"
         ),
         dict(epochs=500,
              batch_size=1024,
              validation_split=0.1,
              shuffle=True,
              verbose=False,
              callbacks=[
                  EarlyStopping(monitor="val_loss", mode="min", patience=50),
                  ktqdm(leave_outer=False)
              ])
        )

    ]

    algorithms = [('DecTree', alberi_decision), ('NN', NN)]


    models = {}

    for name, modelli_creati in algorithms:
        models[name] = []
        for model in modelli_creati:
            models.get(name).append(model)

    return models
