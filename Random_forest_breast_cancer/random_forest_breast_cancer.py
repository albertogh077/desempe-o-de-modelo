# 1. Importar librerías necesarias
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    """Carga el dataset de cáncer de mama y lo devuelve como DataFrame + arrays."""
    cancer_data = load_breast_cancer()
    X = cancer_data.data
    y = cancer_data.target
    feature_names = cancer_data.feature_names

    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    print("Dataset shape:", X.shape)
    print(f"Classes: {np.unique(y)} (0=Malignant, 1=Benign)")
    print(f"Class distribution:\n{pd.Series(y).value_counts()}\n")

    return X, y, feature_names, df


def split_data(X, y, test_size=0.2, random_state=42):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}\n")

    return X_train, X_test, y_train, y_test


def build_model(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42):
    """Crea el clasificador Random Forest con la configuración indicada."""
    return RandomForestClassifier(
        n_estimators=n_estimators,         
        max_depth=max_depth,                
        min_samples_split=min_samples_split,  
        random_state=random_state,          
        n_jobs=-1                           
    )


def train_model(model, X_train, y_train):
    """Entrena el modelo con los datos de entrenamiento."""
    print("Training the Random Forest model...")
    model.fit(X_train, y_train)
    print("Model trained successfully!\n")
    return model


def evaluate_model(model, X_test, y_test):
    """Calcula predicciones, accuracy, classification report y matriz de confusión."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    return y_pred, accuracy, cm


def get_feature_importance(model, feature_names, top_n=10):
    """Devuelve las features ordenadas por importancia y muestra el top_n."""
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop {top_n} Most Important Features:")
    print(feature_importance.head(top_n))

    return feature_importance


def plot_results(cm, feature_importance, top_n=10, output_path='random_forest_breast_cancer_results.png'):
    """Genera y guarda las visualizaciones: matriz de confusión y feature importance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'],
                ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Feature Importance (Top N)
    top_features = feature_importance.head(top_n)
    axes[1].barh(top_features['feature'], top_features['importance'])
    axes[1].set_xlabel('Importance')
    axes[1].set_title(f'Top {top_n} Feature Importance')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGráficos guardados en: {output_path}")
    plt.show()


def main():
    X, y, feature_names, df = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model()
    model = train_model(model, X_train, y_train)

    y_pred, accuracy, cm = evaluate_model(model, X_test, y_test)
    feature_importance = get_feature_importance(model, feature_names)

    plot_results(cm, feature_importance)


if __name__ == '__main__':
    main()