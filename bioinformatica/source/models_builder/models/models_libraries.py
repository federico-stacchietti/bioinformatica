from sklearn import svm as SVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from keras_tqdm import TQDMNotebookCallback as ktqdm
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation
from tensorflow.keras.layers import LSTM