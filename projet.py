import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("data/data.csv")

# 1. Déséquilibre
print(df["fraudulent"].value_counts(normalize=True))

# 2. Valeurs manquantes et doublons
print(df.isna().mean().sort_values(ascending=False))
print("doublons (hors job_id):", df.drop(columns="job_id").duplicated().sum())
df = df.drop_duplicates(subset=["title", "description", "company_profile", "requirements"])

# 3. Texte complet de l'offre (ce que verra le LLM / le TF-IDF)
text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
df["texte"] = df[text_cols].fillna("").agg("\n".join, axis=1)

# 4. Split stratifié + baseline
X, y = df["texte"], df["fraudulent"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print(classification_report(y_test, dummy.predict(X_test), zero_division=0))