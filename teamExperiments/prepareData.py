from sklearn import model_selection, preprocessing

def get_train_test_X_y(classifiedDataframe, y_col, std_cols = None, size = 0.3):
    y = classifiedDataframe[y_col]
    X = classifiedDataframe.drop(columns=y_col)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=size, random_state=42)
    cols = X.columns
    num_cols = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ]

    if std_cols:
        std = preprocessing.StandardScaler()
        X_train.loc[:, std_cols] = std.fit_transform(X_train[std_cols])
        X_test.loc[:, std_cols] = std.transform(X_test[std_cols])

    return X_train, X_test, y_train, y_test