np.random.seed(123)

X = news.drop(columns=['shares', 'is_popular'])
y = news['is_popular']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=123)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=123)

nb_model.fit(X_train, y_train)

knn.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state=123, max_depth=5)
dt_model.fit(X_train, y_train)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

