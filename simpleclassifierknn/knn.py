import argparse
from imutils import paths
from preprocessor.preprocessor import Preprocessor
from dataloader.dataloader import Dataloader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", required=True, help="path to input dataset")
parser.add_argument("-k", "--neighbors", type=int, default=1, help="number of nearest neighbours for classification")
parser.add_argument("-j", "--jobs", type=int, default=-1, help="number of jobs for k-NN distance(-1 uses all available cores)")

args = vars(parser.parse_args())

image_paths = list(paths.list_images(args["dataset"]))
preprocessor = Preprocessor(64, 64)
loader = Dataloader(preprocessors=[preprocessor])
(data, labels) = loader.load(image_paths, verbose = 100)
#print("DATA: ", data.shape, labels)
data = data.reshape((data.shape[0], 12288))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))
