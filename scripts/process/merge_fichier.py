import pandas as pd
import os
import sys
from glob import glob
import re
from collections import defaultdict

if len(sys.argv) != 3:
    print("Usage : python3 merge_fichier.py <dossier_csv> <fichier_sortie.csv>")
    sys.exit(1)

input_dir = sys.argv[1]
output_path = sys.argv[2]

# Trouver tous les fichiers CSV dans le dossier
file_names = sorted(glob(os.path.join(input_dir, "*.csv")))

# Regrouper les fichiers par groupe selon le suffixe _1, _2, etc.
grouped_files = defaultdict(list)
pattern = re.compile(r"_(\d+)\.csv$")

for file in file_names:
    match = pattern.search(file)
    if match:
        group_id = match.group(1)
        grouped_files[group_id].append(file)
    else:
        print(f"Pas de numéro détecté : {file}")

# Fusionner les fichiers de chaque groupe
for group_id, files in grouped_files.items():
    dfs = []
    for file in files:
        print(f"Lecture du fichier {file} dans le groupe {group_id}")
        try:
            df = pd.read_csv(file, on_bad_lines='skip')  # skip lignes malformées
            df["source_file"] = os.path.basename(file)
            dfs.append(df)
        except Exception as e:
            print(f"Erreur lors de la lecture de {file} : {e}")

    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        # Si output_path est un dossier, créer un fichier dedans
        if os.path.isdir(output_path):
            output_file = os.path.join(output_path, f"merged_{group_id}.csv")
        else:
            output_file = output_path
        merged_df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Fichier fusionné enregistré : {output_file}")
    else:
        print(f"Aucun fichier valide à fusionner pour le groupe {group_id}")