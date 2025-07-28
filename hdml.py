from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Baixa os dados diretamente do repositório UCI
heart_disease = fetch_ucirepo(id=45)

# 2. Cria o DataFrame unindo features e target
df = pd.concat([heart_disease.data.features, heart_disease.data.targets], axis=1)

# 3. Renomeia a coluna alvo para 'num' se necessário
if 'num' not in df.columns:
    df.columns = list(heart_disease.data.features.columns) + ['num']

# 4. Trata a variável-alvo: 0 = sem doença, 1 = com doença
df['num'] = df['num'].astype(int)
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# 5. Preenche valores ausentes
df['ca'] = df['ca'].fillna(0)
df['thal'] = df['thal'].fillna(3)

# 6. Define colunas categóricas e numéricas
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

# 7. Cria o pré-processador com padronização e OneHotEncoder
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
])

# 8. Separa X e y
X = df.drop(columns='num')
y = df['num']

# 9. Divide os dados com estratificação
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 10. Aplica o pré-processamento
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# 11. Treina Regressão Logística
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# 12. Treina Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 13. Avaliação
print("=== Regressão Logística ===")
print("Acurácia:", accuracy_score(y_test, y_pred_log))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred_log))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred_log))

print("\n=== Random Forest ===")
print("Acurácia:", accuracy_score(y_test, y_pred_rf))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred_rf))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred_rf))
