# 🌸 D2ClimAFLo-Pyr — Application Streamlit

## Installation

### 1. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Mettre le fichier CSV dans le même dossier
```
📁 dossier/
├── app.py
├── requirements.txt
├── README.md
└── df_complete_2000_2020.csv   ← ton fichier !
```

### 4. Lancer l'application
```bash
streamlit run app.py
```

### 5. Ouvrir dans le navigateur
```
http://localhost:8501
```

---

## Structure de l'application

| Page                    | Contenu                              |
|-------------------------|--------------------------------------|
| 🎭 Introduction         | Histoire + tableau sites + KPI       |
| 🗺️ Zone d'étude        | Carte interactive + altitude         |
| 📊 Stats descriptives   | Filtres variable/année/saison        |
| 📈 Évolution temporelle | Courbes NOHEDES vs Autres            |
| 🔬 ACP + CAH            | PCA + Dendrogramme + Clusters        |
| 🧪 Test d'hypothèse     | Wilcoxon + Boxplots + p-values       |
| 🏁 Conclusion           | Réponses + Perspectives              |

---

## Auteur
Amadou FOFANA · Stage D2ClimAFLo-Pyr · CEFREM · UPVD · 2026
