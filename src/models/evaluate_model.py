y_val_pred = nb_model.predict(X_val)

knn_val_pred = knn.predict(X_val)
knn_test_pred = knn.predict(X_test)

knn_val_probs = knn.predict_proba(X_val)[:, 1]
knn_test_probs = knn.predict_proba(X_test)[:, 1]

knn_val_accuracy = accuracy_score(y_val, knn_val_pred)
knn_f1 =f1_score(y_val, knn_val_pred)
knn_precision = precision_score(y_val, knn_val_pred)
knn_recall = recall_score(y_val, knn_val_pred)
knn_val_auc = roc_auc_score(y_val, knn_val_probs)

dt_val_pred = dt_model.predict(X_val)
dt_test_pred = dt_model.predict(X_test)

dt_val_probs = dt_model.predict_proba(X_val)[:, 1]
dt_test_probs = dt_model.predict_proba(X_test)[:, 1]

dt_val_accuracy = accuracy_score(y_val, dt_val_pred)
dt_f1 =f1_score(y_val, dt_val_pred)
dt_precision = precision_score(y_val, dt_val_pred)
dt_recall = recall_score(y_val, dt_val_pred)
dt_val_auc = roc_auc_score(y_val, dt_val_probs)

conf_matrix = confusion_matrix(y_val, dt_val_pred)
tn, fp, fn, tp = conf_matrix.ravel()

nb_val_pred = nb_model.predict(X_val)
nb_test_pred = nb_model.predict(X_test)

nb_val_probs = nb_model.predict_proba(X_val)[:, 1]
nb_test_probs = nb_model.predict_proba(X_test)[:, 1]

nb_val_auc = roc_auc_score(y_val, nb_val_probs)

nb_val_accuracy = accuracy_score(y_val, nb_val_pred)
nb_f1 =f1_score(y_val, nb_val_pred)
nb_precision = precision_score(y_val, nb_val_pred)
nb_recall = recall_score(y_val, nb_val_pred)

conf_matrix = confusion_matrix(y_val, nb_val_pred)
tn, fp, fn, tp = conf_matrix.ravel()

dt_probs = dt_model.predict_proba(X_test)[:, 1]
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)
dt_auc = auc(dt_fpr, dt_tpr)

nb_probs = nb_model.predict_proba(X_test)[:, 1]
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)
nb_auc = auc(nb_fpr, nb_tpr)

knn_probs = knn.predict_proba(X_test)[:, 1]
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
knn_auc = auc(knn_fpr, knn_tpr)

models = ['Decision Tree', 'k-NN', 'Naive Bayes']
accuracy = [dt_val_accuracy, knn_val_accuracy, nb_val_accuracy]
f1 = [dt_f1, knn_f1, nb_f1]
precision = [dt_precision,knn_precision, nb_precision]
recall = [dt_recall, knn_recall, nb_recall]
auc_score = [dt_val_auc, knn_val_auc, nb_val_auc]
specificity = [dt_specificity, knn_specificity, nb_specificity]

models = ['Decision Tree', 'k-NN', 'Naive Bayes']
accuracy = [dt_val_accuracy, knn_val_accuracy, nb_val_accuracy]
f1 = [dt_f1, knn_f1, nb_f1]
precision = [dt_precision,knn_precision, nb_precision]
recall = [dt_recall, knn_recall, nb_recall]
auc_score = [dt_val_auc, knn_val_auc, nb_val_auc]