import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Charger le dataset
data = pd.read_csv("creditcard.csv")
X = data.drop("Class", axis=1)
y = data["Class"]

# Vérifier la distribution originale
print("Distribution originale :")
print(y.value_counts())

# Appliquer SMOTE
sm = SMOTE(sampling_strategy=1.0, random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Vérifier la distribution après SMOTE
print("\nDistribution après SMOTE :")
print(y_res.value_counts())

# Préparer un dataframe pour visualisation
data_smote = pd.concat([pd.DataFrame(X_res, columns=X.columns),
                        pd.Series(y_res, name="Class")], axis=1)

# Remplacer 0 et 1 par Normal et Fraude
data["Class"] = data["Class"].map({0: "Normal", 1: "Fraude"})
data_smote["Class"] = data_smote["Class"].map({0: "Normal", 1: "Fraude"})

# Afficher les deux distributions sur la même figure
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Distribution des classes avant et après SMOTE",
             fontsize=14, fontweight="bold")

# Avant SMOTE
sns.countplot(x="Class", data=data, ax=axes[0],
              hue="Class", palette={"Normal": "steelblue", "Fraude": "crimson"}, legend=True)
axes[0].set_title("Avant SMOTE")
axes[0].set_ylabel("Nombre de transactions")
axes[0].set_yscale("log")
axes[0].legend(title="Classe")
for p in axes[0].patches:
    if p.get_height() > 0:
        axes[0].text(p.get_x() + p.get_width() / 2, p.get_height() * 1.3,
                     f'{int(p.get_height()):,}', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')

# Après SMOTE
sns.countplot(x="Class", data=data_smote, ax=axes[1],
              hue="Class", palette={"Normal": "steelblue", "Fraude": "crimson"}, legend=True)
axes[1].set_title("Après SMOTE")
axes[1].set_ylabel("Nombre de transactions")
axes[1].set_yscale("log")
axes[1].legend(title="Classe")
for p in axes[1].patches:
    if p.get_height() > 0:
        axes[1].text(p.get_x() + p.get_width() / 2, p.get_height() * 1.3,
                     f'{int(p.get_height()):,}', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()