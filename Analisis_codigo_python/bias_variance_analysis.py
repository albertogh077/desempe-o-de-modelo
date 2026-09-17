# Análisis de bias, varianza, ajuste y regularización sobre el Random Forest
# entrenado con el dataset de cáncer de mama (Breast Cancer Wisconsin).
# 1. Importar librerías necesarias
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, learning_curve, validation_curve, GridSearchCV, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from random_forest_breast_cancer import load_data

OUT_DIR = 'reports'
import os
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42


def split_train_val_test(X, y, val_size=0.2, test_size=0.2, random_state=RANDOM_STATE):
    """Divide en Train/Validation/Test (60/20/20) de forma estratificada."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=random_state, stratify=y
    )
    rel_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test_size, random_state=random_state, stratify=y_temp
    )
    print(f"Train: {X_train.shape[0]} ({X_train.shape[0]/len(X):.1%})")
    print(f"Validation: {X_val.shape[0]} ({X_val.shape[0]/len(X):.1%})")
    print(f"Test: {X_test.shape[0]} ({X_test.shape[0]/len(X):.1%})\n")
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(model, X, y):
    """Calcula el accuracy del modelo sobre un conjunto de datos dado."""
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)


def report_split_metrics(model, splits, label):
    """Evalúa el modelo en Train/Validation/Test y devuelve el accuracy de cada uno."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits
    acc_train = evaluate(model, X_train, y_train)
    acc_val = evaluate(model, X_val, y_val)
    acc_test = evaluate(model, X_test, y_test)
    print(f"[{label}] Train acc: {acc_train:.4f} | Val acc: {acc_val:.4f} | Test acc: {acc_test:.4f}")
    return {'train': acc_train, 'val': acc_val, 'test': acc_test}


def plot_learning_curve(model, X_train, y_train, output_path, title):
    """Genera la curva de aprendizaje (accuracy de train vs. CV según el tamaño de muestra) para diagnosticar bias/varianza."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8), scoring='accuracy',
        random_state=RANDOM_STATE
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_mean, 'o-', color='tab:blue', label='Entrenamiento')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='tab:blue')
    plt.plot(train_sizes, val_mean, 'o-', color='tab:orange', label='Validación cruzada (CV)')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='tab:orange')
    plt.xlabel('Número de muestras de entrenamiento')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.ylim(0.85, 1.01)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Curva de aprendizaje guardada en: {output_path}")
    return train_sizes, train_mean, val_mean


def plot_validation_curve_depth(X_train, y_train, output_path):
    """Genera la curva de validación variando max_depth para ubicar la zona de underfitting/overfitting."""
    depth_range = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None]
    depth_labels = [str(d) if d is not None else 'None' for d in depth_range]

    base_model = RandomForestClassifier(
        n_estimators=100, min_samples_split=2, random_state=RANDOM_STATE, n_jobs=-1
    )
    train_scores, val_scores = validation_curve(
        base_model, X_train, y_train, param_name='max_depth', param_range=depth_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    train_mean, val_mean = train_scores.mean(axis=1), val_scores.mean(axis=1)

    x = np.arange(len(depth_range))
    plt.figure(figsize=(7, 5))
    plt.plot(x, train_mean, 'o-', color='tab:blue', label='Entrenamiento')
    plt.plot(x, val_mean, 'o-', color='tab:orange', label='Validación cruzada (CV)')
    plt.xticks(x, depth_labels)
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.title('Curva de validación: profundidad del árbol (max_depth)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Curva de validación (max_depth) guardada en: {output_path}")
    return depth_labels, train_mean, val_mean


def tune_hyperparameters(X_train, y_train):
    """Busca la combinación de hiperparámetros de regularización con mejor accuracy en validación cruzada (5-fold, solo sobre Train)."""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7, 10],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2'],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"\nMejores hiperparámetros encontrados: {grid.best_params_}")
    print(f"Mejor accuracy promedio en CV (train): {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def plot_before_after_bars(before, after, output_path):
    """Grafica en barras el accuracy de Train/Validation/Test antes y después de regularizar."""
    labels = ['Train', 'Validation', 'Test']
    before_vals = [before['train'], before['val'], before['test']]
    after_vals = [after['train'], after['val'], after['test']]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(x - width/2, before_vals, width, label='Antes (baseline)', color='tab:red', alpha=0.8)
    plt.bar(x + width/2, after_vals, width, label='Después (regularizado)', color='tab:green', alpha=0.8)
    for i, v in enumerate(before_vals):
        plt.text(i - width/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate(after_vals):
        plt.text(i + width/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
    plt.xticks(x, labels)
    plt.ylabel('Accuracy')
    plt.ylim(0.85, 1.05)
    plt.title('Accuracy por conjunto: antes vs. después de regularizar')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparación antes/después guardada en: {output_path}")


def plot_confusion_matrices(cm_before, cm_after, output_path):
    """Grafica lado a lado las matrices de confusión de Test antes y después de regularizar."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cm, title in zip(axes, [cm_before, cm_after], ['Antes (baseline)', 'Después (regularizado)']):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Maligno', 'Benigno'], yticklabels=['Maligno', 'Benigno'], ax=ax)
        ax.set_title(f'Matriz de confusión (Test) - {title}')
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicho')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matrices de confusión guardadas en: {output_path}")


def main():
    """Orquesta el análisis completo: split train/val/test, diagnóstico del modelo baseline
    (bias, varianza y ajuste), búsqueda de hiperparámetros y comparación antes/después de regularizar."""
    X, y, feature_names, df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)
    splits = ((X_train, y_train), (X_val, y_val), (X_test, y_test))

    # 1. Modelo baseline: Random Forest sin límite de profundidad (mayor riesgo de sobreajuste)
    print("=" * 70)
    print("MODELO BASELINE (sin regularización de profundidad)")
    print("=" * 70)
    baseline_model = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1
    )
    baseline_model.fit(X_train, y_train)
    baseline_metrics = report_split_metrics(baseline_model, splits, 'Baseline')
    y_pred_test_before = baseline_model.predict(X_test)
    cm_before = confusion_matrix(y_test, y_pred_test_before)

    # 2. Diagnóstico de bias/varianza: curva de aprendizaje del baseline
    print("\nCurva de aprendizaje (baseline)...")
    plot_learning_curve(
        baseline_model, X_train, y_train,
        f'{OUT_DIR}/learning_curve_baseline.png',
        'Curva de aprendizaje - Modelo baseline (max_depth=None)'
    )

    # 3. Diagnóstico de ajuste (underfit/overfit): curva de validación sobre max_depth
    print("\nCurva de validación sobre max_depth...")
    depth_labels, depth_train, depth_val = plot_validation_curve_depth(
        X_train, y_train, f'{OUT_DIR}/validation_curve_max_depth.png'
    )

    # 4. Regularización: búsqueda de hiperparámetros por validación cruzada, solo sobre Train
    print("\n" + "=" * 70)
    print("BÚSQUEDA DE HIPERPARÁMETROS (GridSearchCV, 5-fold, sobre Train)")
    print("=" * 70)
    best_model, best_params, best_cv_score = tune_hyperparameters(X_train, y_train)

    # 5. Modelo regularizado: se reentrena con los mejores hiperparámetros y se evalúa igual que el baseline
    print("\n" + "=" * 70)
    print("MODELO REGULARIZADO (después de tuning)")
    print("=" * 70)
    best_model.fit(X_train, y_train)
    regularized_metrics = report_split_metrics(best_model, splits, 'Regularizado')
    y_pred_test_after = best_model.predict(X_test)
    cm_after = confusion_matrix(y_test, y_pred_test_after)

    print("\nCurva de aprendizaje (modelo regularizado)...")
    plot_learning_curve(
        best_model, X_train, y_train,
        f'{OUT_DIR}/learning_curve_regularized.png',
        'Curva de aprendizaje - Modelo regularizado'
    )

    # 6. Comparación antes/después: accuracy por conjunto y matrices de confusión en Test
    plot_before_after_bars(baseline_metrics, regularized_metrics, f'{OUT_DIR}/before_after_accuracy.png')
    plot_confusion_matrices(cm_before, cm_after, f'{OUT_DIR}/confusion_matrices_before_after.png')

    print("\nReporte de clasificación (Test) - Antes:")
    print(classification_report(y_test, y_pred_test_before, target_names=['Malignant', 'Benign']))
    print("\nReporte de clasificación (Test) - Después:")
    print(classification_report(y_test, y_pred_test_after, target_names=['Malignant', 'Benign']))

    # 7. Se guardan todos los resultados numéricos para poder redactar el reporte con datos reales
    results = {
        'baseline': {
            'params': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
            'metrics': baseline_metrics,
            'gap_train_test': baseline_metrics['train'] - baseline_metrics['test'],
            'confusion_matrix_test': cm_before.tolist(),
        },
        'regularized': {
            'params': best_params,
            'cv_score_train': best_cv_score,
            'metrics': regularized_metrics,
            'gap_train_test': regularized_metrics['train'] - regularized_metrics['test'],
            'confusion_matrix_test': cm_after.tolist(),
        },
        'validation_curve_max_depth': {
            'depth_labels': depth_labels,
            'train_scores': depth_train.tolist(),
            'val_scores': depth_val.tolist(),
        },
        'split_sizes': {
            'train': int(X_train.shape[0]),
            'val': int(X_val.shape[0]),
            'test': int(X_test.shape[0]),
        }
    }
    with open(f'{OUT_DIR}/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados numéricos guardados en: {OUT_DIR}/results.json")


if __name__ == '__main__':
    main()
