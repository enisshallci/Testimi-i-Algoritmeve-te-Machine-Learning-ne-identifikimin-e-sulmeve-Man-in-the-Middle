#Support vector machines (SVMs): Kjo perdoret per klasifikim te lidhjeve #trafiku duke gjetur hiperplanin që ndan në maksimum trafikun normal #nga trafiku keqdashës.
import pandas as pd  # importon bibliotekën pandas
from sklearn.svm import SVC  # importon klasën SVC (support vector classifier) nga sklearn.svm
from sklearn.model_selection import train_test_split  # importon funksionin train_test_split nga sklearn.model_selection

# Në këtë rast po përdorim një skedar CSV për të ngarkuar të dhënat. Pandas mund të përdoret për të lexuar një skedar CSV duke përdorur funksionin read_csv
df = pd.read_csv('data.csv')

# Ndajmë të dhënat në "features" dhe "labels"
X = df.drop('label', axis=1)  # features
y = df['label']  # labels

# Ndajmë të dhënat në një "training set" dhe një "testing set"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Krijojmë një model të llojit "SVM"
model = SVC()

# Fitojmë modelin me të dhënat e "training set"-it
model.fit(X_train, y_train)

