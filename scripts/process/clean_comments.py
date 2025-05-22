import pandas as pd
import spacy
import unicodedata
import argparse 

# Commande: python clean_comments.py mid_commit.csv mid_commit_clean.csv

# Charger le modèle
nlp = spacy.load("fr_core_news_sm")

# Supprimer les caractères de contrôle unicode invisibles
def remove_carac_unicode(text):
    return ''.join(
        ch for ch in str(text)
        if unicodedata.category(ch)[0] != 'C'
    )

# Nettoyage du texte
def clean_comment(text):
    text = text.lower()
    # Nettoyer les caractères unicode invisibles, comme '￼'
    text = remove_carac_unicode(text)
    # Traiter le texte
    doc = nlp(text)
    # Nettoyer le texte, supprimer les mots vides, la ponct et les espaces, garder les lemmes
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    # Joindre les lemmes
    return " ".join(tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utilisé pour nettoyer les commentaires")

    parser.add_argument("csv_file", help="Le fichier CSV à traiter")
    parser.add_argument("output_csv", help="Fichier CSV de sortie") 
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_file)

     # Garder uniquement les colonnes 'cleaned_comment' et 'class' dans le nouveau fichier
    new_df = pd.DataFrame()
    new_df["cleaned_comment"] = df["Comment"].apply(clean_comment)
    new_df["class"] = df["Class"]

    new_df.to_csv(args.output_csv, index=False)