import numpy as np
import pandas as pd

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found")
        return None
    except pd.errors.EmptyDataError:
        print("No data in file")
        return None
    except pd.errors.ParserError:
        print("Error parsing file")
        return None

def split_data(data, test_size=0.2):
    from sklearn.model_selection import train_test_split
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def evaluate_model(y_true, y_pred):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    return accuracy, report, matrix

def get_feature_importances(model):
    import matplotlib.pyplot as plt
    import seaborn as sns
    importances = model.feature_importances_
    feature_names = model.feature_names_in_
    plt.figure(figsize=(10, 8))
    sns.barplot(x=feature_names, y=importances)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.show()