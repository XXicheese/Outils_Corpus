import os
import sys
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

# Vérification des arguments
if len(sys.argv) != 3:
    print("Usage : python synthetic_data.py <chemin_dossier_csv> <chemin_sortie_csv>")
    sys.exit(1)

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

# Chargement des modèles de traduction pour back-translation
def load_translation_models():
    # fr -> en
    tokenizer_fr_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
    model_fr_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
    # en -> fr
    tokenizer_en_fr = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
    model_en_fr = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
    return (tokenizer_fr_en, model_fr_en, tokenizer_en_fr, model_en_fr)

# Traduction avec modèle MarianMT
def translate(texts, tokenizer, model):
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**batch)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

# Augmentation par back-translation FR -> EN -> FR
def back_translate(text, tokenizer_fr_en, model_fr_en, tokenizer_en_fr, model_en_fr):
    try:
        en_text = translate([text], tokenizer_fr_en, model_fr_en)[0]
        fr_text = translate([en_text], tokenizer_en_fr, model_en_fr)[0]
        return fr_text
    except Exception as e:
        print(f"Erreur traduction pour texte : {text[:30]}... : {e}")
        return text  # retour du texte original si erreur

def augmenter_donnees(df, nb_exemples=1000):
    tokenizer_fr_en, model_fr_en, tokenizer_en_fr, model_en_fr = load_translation_models()
    synth_texts = []
    synth_labels = []

    for i, (text, label) in enumerate(zip(df['cleaned_comment'], df['label'])):
        if i >= nb_exemples:
            break
        augmented = back_translate(text, tokenizer_fr_en, model_fr_en, tokenizer_en_fr, model_en_fr)
        synth_texts.append(augmented)
        synth_labels.append(label)

    return pd.DataFrame({'cleaned_comment': synth_texts, 'label': synth_labels})

# Traitement principal
df_original = charger_donnees(dossier_csv)
df_prepared = preparer_dataframe(df_original)
df_synth = augmenter_donnees(df_prepared, nb_exemples=1000)
df_total = pd.concat([df_prepared, df_synth], ignore_index=True)
df_total.to_csv(fichier_sortie, index=False)
print(f"Fichier avec données augmentées enregistré à : {fichier_sortie}")