news.dtypes

news.isnull().sum()

non_numeric_features = news.select_dtypes(exclude=['int', 'float'])

nb_model = GaussianNB()

labels = ['0', '1'] 

x = np.arange(len(models))
width = 0.1

x = np.arange(len(models))
width = 0.1