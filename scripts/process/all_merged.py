import os
import argparse
import pandas as pd
from glob import glob

def merge_files(input_dir, output_file):
    files = sorted(glob(os.path.join(input_dir, "*")))

    if not files:
        print("Aucun fichier trouvé dans ce dossier.")
        return

    extension = os.path.splitext(files[0])[1].lower()

    if extension == ".csv":
        dfs = []
        for file in files:
            print(f"Lecture de {file}")
            try:
                df = pd.read_csv(file)
                df["source_file"] = os.path.basename(file)
                dfs.append(df)
            except Exception as e:
                print(f"⚠️ Erreur de lecture {file} : {e}")
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Fichier CSV fusionné sauvegardé dans : {output_file}")
    else:
        with open(output_file, "w", encoding="utf-8") as out_f:
            for file in files:
                print(f"Ajout de {file}")
                try:
                    with open(file, "r", encoding="utf-8") as in_f:
                        content = in_f.read()
                        out_f.write(f"\n===== {os.path.basename(file)} =====\n")
                        out_f.write(content)
                        out_f.write("\n")
                except Exception as e:
                    print(f"Erreur de lecture {file} : {e}")
        print(f"Fichiers fusionnés en texte brut dans : {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusionner tous les fichiers d’un dossier en un seul fichier.")
    parser.add_argument("input_dir", help="Dossier contenant les fichiers à fusionner")
    parser.add_argument("output_file", help="Nom du fichier de sortie")
    args = parser.parse_args()

    merge_files(args.input_dir, args.output_file)