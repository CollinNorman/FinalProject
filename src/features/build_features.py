Q1 = news['shares'].quantile(0.15)
Q3 = news['shares'].quantile(0.85)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

news = news[(news['shares'] >= lower_bound) & (news['shares'] <= upper_bound)]

news.columns = news.columns.str.strip()

threshold = np.percentile(news['shares'], 50)

news['is_popular'] = news['shares'].apply(lambda x: 1 if x >= threshold else 0)

unique_counts = news['is_popular'].nunique()
print(news['is_popular'].value_counts())

k_v = range(1, 30)
knn_acc = []

for k in k_v:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_val_pred = knn.predict(X_val)
    accuracy = precision_score(y_val, y_val_pred)
    knn_acc.append(accuracy)

k_v = range(1, 30)
knn_acc = []

for k in k_v:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_val_pred = knn.predict(X_val)
    accuracy = recall_score(y_val, y_val_pred)
    knn_acc.append(accuracy)

k_v = range(1, 30)
knn_acc = []

for k in k_v:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_val_pred = knn.predict(X_val)
    accuracy = f1_score(y_val, y_val_pred)
    knn_acc.append(accuracy)

max_depth_values = range(1, 30)
val_accuracies = []

for depth in max_depth_values:
    dt_model = DecisionTreeClassifier(random_state=123, max_depth=depth)
    dt_model.fit(X_train, y_train)
    y_val_pred = dt_model.predict(X_val)
    accuracy = recall_score(y_val, y_val_pred)
    val_accuracies.append(accuracy)

max_depth_values = range(1, 30)
val_accuracies = []

for depth in max_depth_values:
    dt_model = DecisionTreeClassifier(random_state=123, max_depth=depth)
    dt_model.fit(X_train, y_train)
    y_val_pred = dt_model.predict(X_val)
    accuracy = f1_score(y_val, y_val_pred)
    val_accuracies.append(accuracy)

max_depth_values = range(1, 30)
val_accuracies = []

for depth in max_depth_values:
    dt_model = DecisionTreeClassifier(random_state=123, max_depth=depth)
    dt_model.fit(X_train, y_train)
    y_val_pred = dt_model.predict(X_val)
    accuracy = precision_score(y_val, y_val_pred)
    val_accuracies.append(accuracy)

knn = KNeighborsClassifier(n_neighbors=29)

conf_matrix = confusion_matrix(y_val, knn_val_pred)
tn, fp, fn, tp = conf_matrix.ravel()

knn_fpr = fp/(fp + tn) if (fp + tn)> 0 else 0
knn_specificity = tn/(tn + fp) if (tn + fp)> 0 else 0

labels = ['0', '1'] 

dt_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
dt_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

nb_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
nb_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

feature_importances = dt_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)
top_10_features = importance_df.head(10)

