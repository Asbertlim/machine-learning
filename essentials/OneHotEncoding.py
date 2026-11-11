# One-Hot Encoding -> converts categorical values into separate binary columns containing either 1 or 0
# fit -> learns the available categories
# transform() -> converts categories into binary columns
# fit_transform() -> performs bothoperations
# retrieve by using encoder.get_feature_names_out()

# num -> keep numerical feautres as they are
# passtrhough -> don't transofrm these numerical columns
# cat -> apply one hot encoding to categorical features
# handle_unknown="ignore" -> if the model encounters an unseen category during training, it won't carsh

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

town_encoded = encoder.fit_transform(
    data[["town"]]
)

print(town_encoded.toarray())

# ------------------------------------------------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

numerical_features = [
    "floor_area_sqm",
    "lease_commence_date"
]

categorical_features = [
    "town",
    "flat_type",
    "storey_range"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)
