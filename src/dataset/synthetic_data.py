import os
import sys
import pandas as pd
import nlpaug.augmenter.word as naw

# Vérification des arguments
if len(sys.argv) != 3:
    print("Usage : python synthetic_data.py <chemin_dossier_csv> <chemin_sortie_csv>")
    sys.exit(1)

# Arguments
dossier_csv = sys.argv[1]
fichier_sortie = sys.argv[2]

# Chargement des données originales
def charger_donnees(dossier):
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.csv')]
    dfs = []
    for f in fichiers:
        path = os.path.join(dossier, f)
        df = pd.read_csv(path)
        if 'cleaned_comment' in df.columns and 'class' in df.columns:
            dfs.append(df[['cleaned_comment', 'class']])
        else:
            print(f"Fichier ignoré : {f}")
    return pd.concat(dfs, ignore_index=True)

# Préparer les données pour l’augmentation
def preparer_dataframe(df):
    df = df.dropna(subset=['cleaned_comment', 'class'])
    mapping = {'neg': 0, 'mid': 1, 'pos': 2}
    df['label'] = df['class'].map(mapping)
    df = df.dropna(subset=['label'])
    return df[['cleaned_comment', 'label']]

# Génération de textes augmentés par synonymie
def augmenter_donnees(df, nb_exemples=1000):
    augmenter = naw.SynonymAug(aug_src='wordnet')
    synth_texts = []
    synth_labels = []

    for i, (text, label) in enumerate(zip(df['cleaned_comment'], df['label'])):
        if i >= nb_exemples:
            break
        try:
            augmented = augmenter.augment(text)
            synth_texts.append(augmented)
            synth_labels.append(label)
        except Exception as e:
            print(f"Erreur sur exemple {i} : {e}")
            continue

    return pd.DataFrame({'cleaned_comment': synth_texts, 'label': synth_labels})

# Traitement principal
df_original = charger_donnees(dossier_csv)
df_prepared = preparer_dataframe(df_original)
df_synth = augmenter_donnees(df_prepared, nb_exemples=1000)
df_total = pd.concat([df_prepared, df_synth], ignore_index=True)
df_total.to_csv(fichier_sortie, index=False)
print(f"Fichier avec données augmentées enregistré à : {fichier_sortie}")