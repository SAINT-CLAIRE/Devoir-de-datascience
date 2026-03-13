#impoerter les bibliotheque
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#charger le dataset
data = pd.read_csv("creditcard.csv")
print(data.head())
#voir la redistrubution des classe
print(data["Class"].value_counts())
#separer les classe
fraud = data[data["Class"] == 1]
normal = data[data["Class"] == 0]
print("Fraudes :", len(fraud))
print("Normales :", len(normal))
#Voir la distribution des classes
print(data["Class"].value_counts())
#Appliquer Random Under Sampling
normal_sample = normal.sample(n=len(fraud), random_state=42)
# Créer le nouveau dataset équilibré
new_data = pd.concat([fraud, normal_sample])
new_data = new_data.sample(frac=1, random_state=42)
print(new_data["Class"].value_counts())
#nouvel arrangement
new_data = new_data.sample(frac=1, random_state=42)
print(new_data["Class"].value_counts())
# Remplacer 0 et 1 par Normal et Fraude
data["Class"] = data["Class"].map({0: "Normal", 1: "Fraude"})
new_data["Class"] = new_data["Class"].map({0: "Normal", 1: "Fraude"})
# Création des deux graphiques sur la même figure
fig, axes = plt.subplots(1, 2, figsize=(10,4))
fig.suptitle("Distribution des classes avant et après Random Under Sampling", fontsize=14,fontweight="bold")

# Graphique 1 : Avant

# Graphique 1 : Avant
sns.countplot(x="Class", data=data, ax=axes[0],
              hue="Class", palette={"Normal": "steelblue", "Fraude": "crimson"}, legend=True)
axes[0].set_title("Avant Under-Sampling")
axes[0].set_ylabel("Nombre de transactions")
axes[0].legend(title="Classe")

# Graphique 2 : Après
sns.countplot(x="Class", data=new_data, ax=axes[1],
              hue="Class", palette={"Normal": "steelblue", "Fraude": "crimson"}, legend=True)
axes[1].set_title("Après Under-Sampling")
axes[1].set_ylabel("Nombre de transactions")
axes[1].legend(title="Classe")

plt.tight_layout()
plt.savefig("undersampling.png", dpi=120, bbox_inches="tight")
plt.show()
