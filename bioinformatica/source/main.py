from bioinformatica.source.models_builder.experiments_builder.builder import *


if __name__ == '__main__':
    dicctt = {'a': [1, 2], 'b': [5, 6]}

    print([[x] + dicctt.get(x) for x in dicctt])