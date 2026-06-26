#Import des bibliothèques
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer

# -----------------------------------------------------------------------------
# 1. Fonction d’apprentissage et de test des algorithmes supervisés
# -----------------------------------------------------------------------------

def train_test_supervised(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "accuracy": acuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=1),
        "recall": recall_score(y_test, y_pred, pos_label=1)
    }

# -------------------------------------------------------------------
# 2. Fonction d’apprentissage et de test des algorithmes supervisés sur des données normalisé
# -------------------------------------------------------------------

def train_test_supervised_normalized(X_train, X_test, y_train, y_test, model):
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return train_test_supervised(X_train_norm, X_test_norm, y_train, y_test, model)

# -------------------------------------------------------------------
# 3.  Fonction d’apprentissage et de test des algorithmes supervisés sur des données incluant des nouvelles variables
# -------------------------------------------------------------------

def train_test_supervised_new_variables(X_train, X_test, y_train, y_test, model):
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_test_pca = pca.transform(X_test_norm)
    return train_test_supervised(X_train_pca, X_test_pca, y_train, y_test, model)

# -------------------------------------------------------------------
# 4. Fonction de sélection des variables les plus pertinentes
# -------------------------------------------------------------------

def select_optimal_variables(X_train, X_test, y_train, y_test, sorted_idx):
    model = MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1)
    scores=np.zeros(X_train.shape[1])
    for f in range(X_train.shape[1]):
        X1_f = X_train[:,sorted_idx[:f+1]]
        X2_f = X_test[:,sorted_idx[:f+1]]
        model.fit(X1_f,y_train)
        y_pred = model.predict(X2_f)
        scores[f] = precision_score(y_test, y_pred, pos_label=1)

    return scores

# -------------------------------------------------------------------
# 5. Fonction de recherche et de choix des meilleurs paramètres
# -------------------------------------------------------------------

def custom_metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    return (acc + prec) / 2

custom_scorer = make_scorer(custom_metric, greater_is_better=True)

def getParam(X_train, y_train):
    model = MLPClassifier(random_state=1)
    param = {"hidden_layer_sizes" : [(20,),(40,)]}
    grid = GridSearchCV( 
        estimator = model,
        param_grid = param,
        scoring=custom_scorer, 
        cv=5
    )
    grid.fit(X_train, y_train)
    return grid.best_params_

# -------------------------------------------------------------------
# 6.  Fonction de création du pipeline d’apprentissage
# -------------------------------------------------------------------

def generer_et_sauvegarder_pipeline(model, X, y, n_features, use_pca, nom_fichier):
    
    steps = []
    steps.append(('scaler', StandardScaler()))

    if use_pca:
        steps.append(('pca', PCA(n_components=3)))

    selecteur = SelectFromModel(
        estimator = RandomForestClassifier(n_estimators=300, random_state=1),
        max_features = n_features,
        threshold = -np.inf
    )
    steps.append(('selection', selecteur))
    steps.append(('model', model))

    pipeline = Pipeline(steps)
    pipeline.fit(X, y)
    with open(nom_fichier, 'wb') as f:
        pickle.dump(pipeline, f)
    return pipeline

# -------------------------------------------------------------------
# 7. Fonction principale
# -------------------------------------------------------------------

def pipeline_generation_train_test_split(X, y, n_features=8, use_pca=False, nom_fichier="credit_scoring.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=200, random_state=1)
    rf.fit(X_train_norm, y_train)
    sorted_idx = np.argsort(rf.feature_importances_)[::-1]

    optimal_idx = sorted_idx[:n_features]
    X_train_opt = X_train_norm[:, optimal_idx]

    best_params = getParam(X_train_opt, y_train)
    best_model = MLPClassifier(**best_params, random_state=1)

    pipeline = generer_et_sauvegarder_pipeline(
        model = best_model,
        X = X_train,
        y = y_train,
        n_features = n_features,
        use_pca = use_pca,
        nom_fichier = nom_fichier
    )
    return pipeline
    

# -------------------------------------------------------------------
# 8. Fonction pipeline_generation_cv (question 10)
# -------------------------------------------------------------------

def tune(X, y):
    param_grid = {
        "n_estimators": [100, 200, 300]
    }
    base_model = XGBClassifier(random_state=1)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,scoring="roc_auc",
        cv=5
    )
    grid.fit(X, y)
    return grid.best_params_


def pipeline_generation_cv(X, y, n_features = 8, use_pca = False, nom_fichier = "credit_scoring_cv.pkl"):
    best_model = XGBClassifier(n_estimators=200, random_state=1)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=200, random_state=1)
    rf.fit(X_norm, y)
    sorted_idx = np.argsort(rf.feature_importances_)[::-1]
    optimal_idx = sorted_idx[:n_features]
    X_opt = X_norm[:, optimal_idx]

    best_params = tune(X_opt, y)
    best_model = XGBClassifier(**best_params, random_state=1)

    pipeline = generer_et_sauvegarder_pipeline(
        model=best_model,
        X=X,
        y=y,
        n_features=n_features,
        use_pca=use_pca,
        nom_fichier=nom_fichier
    )
    return pipeline, "XGBoost", best_params
