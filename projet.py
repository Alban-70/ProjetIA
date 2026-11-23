
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ============================================================
# 1. CHARGEMENT DES DONNEES
# ============================================================

df = pd.read_csv("data/data.csv")

print("===== REPARTITION DES CLASSES =====")
print(df["fraudulent"].value_counts(normalize=True))

print("\n===== VALEURS MANQUANTES =====")
print(df.isna().mean().sort_values(ascending=False))

print("\n===== DOUBLONS =====")
print("Doublons (hors job_id) :", df.drop(columns="job_id").duplicated().sum())


# ============================================================
# 2. NETTOYAGE DES DONNEES
# ============================================================

df = df.drop_duplicates(
    subset=["title", "description", "company_profile", "requirements"]
)

print("\nNombre d'offres après suppression des doublons :", len(df))


# ============================================================
# 3. CREATION DU TEXTE A ANALYSER
# ============================================================

text_cols = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

df["texte"] = df[text_cols].fillna("").agg("\n".join, axis=1)


# ============================================================
# 4. SEPARATION DES DONNEES
# ============================================================

X = df["texte"]
y = df["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nNombre d'offres pour l'entraînement :", len(X_train))
print("Nombre d'offres pour les tests :", len(X_test))


# ============================================================
# 5. CHARGEMENT OU ENTRAINEMENT DU MODELE
# ============================================================

MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):

    print("\n===== CHARGEMENT DU MODELE =====")

    model = joblib.load(MODEL_PATH)

    print("Modèle chargé depuis", MODEL_PATH)

else:

    print("\n===== CREATION ET ENTRAINEMENT DU MODELE =====")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000
        )),

        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    print("Modèle entraîné et sauvegardé dans", MODEL_PATH)


# ============================================================
# 6. EVALUATION DU MODELE
# ============================================================

y_pred = model.predict(X_test)

print("\n===== RESULTATS DU MODELE =====")

print("Accuracy :", accuracy_score(y_test, y_pred))

print("\nRapport de classification :")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Offre légitime", "Offre frauduleuse"],
    zero_division=0
))

print("\nMatrice de confusion :")

print(confusion_matrix(y_test, y_pred))


# ============================================================
# 7. ANALYSE DES ERREURS
# ============================================================

print("\n===== ANALYSE DES ERREURS =====")

# Associer chaque prédiction à son indice d'origine
predictions = pd.Series(y_pred, index=y_test.index)

# Rechercher les offres frauduleuses classées comme légitimes
false_negatives = y_test.index[
    (y_test == 1) & (predictions == 0)
]

print("Nombre de fraudes non détectées :", len(false_negatives))


# ============================================================
# 8. AFFICHAGE DES 5 PREMIERES FRAUDES NON DETECTEES
# ============================================================

print("\n===== EXAMEN DES 5 PREMIERES OFFRES =====")

for index in false_negatives[:5]:

    offre = df.loc[index]

    print("\n" + "=" * 70)

    print("IDENTIFIANT :", offre["job_id"])

    print("\nTITRE :")
    print(offre["title"])

    print("\nENTREPRISE :")
    print(offre["company_profile"])

    print("\nDESCRIPTION :")
    print(str(offre["description"])[:500])

    print("\nEXIGENCES :")
    print(str(offre["requirements"])[:300])

    print("\nSALAIRE :")
    print(offre["salary_range"])

    print("\n" + "=" * 70)
