import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import RandomizedSearchCV
def main():
    raw_data = pd.read_csv("./data/car_data_raw.csv")
    X = raw_data.drop("class", axis=1)
    y = raw_data["class"]
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)
    car_preprocessor = make_column_transformer(
        (OrdinalEncoder(categories=[['low','med','high','vhigh']]), ['buying']),
        (OrdinalEncoder(categories=[['low','med','high','vhigh']]), ['maint']),
        (OrdinalEncoder(categories=[['2','3','4','5more']]), ['doors']),
        (OrdinalEncoder(categories=[['2','4','more']]), ['persons']),
        (OrdinalEncoder(categories=[['small','med','big']]), ['lug_boot']),
        (OrdinalEncoder(categories=[['low','med','high']]), ['safety']),
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    svc = SVC()
    print(X_train.shape)
    car_pipe = make_pipeline(car_preprocessor, svc)
    car_pipe.fit(X_train, y_train)
    score = car_pipe.score(X_test, y_test).round(3)
    print(f"The accuracy of our model is {score}")
if __name__ == '__main__':
    main()