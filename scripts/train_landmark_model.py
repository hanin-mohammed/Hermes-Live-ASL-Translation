import csv
import joblib
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier


project_folder = Path(__file__).resolve().parent.parent
data_file = project_folder / "data" / "landmark_data.csv"
model_file = project_folder / "models" / "landmark_knn_model.pkl"

X = []
y = []

with open(data_file, "r") as file:
    reader = csv.reader(file)

    for row in reader:
        coordinates = row[0:63]
        label = row[63]

        number_coordinates = []

        for value in coordinates:
            number_coordinates.append(float(value))

        X.append(number_coordinates)
        y.append(label)

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X, y)

joblib.dump(model, model_file)

print("Training complete")
print("Total examples:", len(X))
print("Model saved to:", model_file)
