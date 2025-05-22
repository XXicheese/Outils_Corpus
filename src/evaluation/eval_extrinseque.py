import sys
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader
from datasets import Dataset

def predict_on_new_data(model, tokenizer, fichier_nouveau):
    df = pd.read_csv(fichier_nouveau)
    dataset = Dataset.from_pandas(df)

    # Fonction de tokenization
    def tokenize_function(examples):
        return tokenizer(examples['cleaned_comment'], truncation=True, padding='max_length', max_length=128)

    # Appliquer la tokenization en mode batch
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    loader = DataLoader(dataset, batch_size=8)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).tolist())

    df['predicted_label'] = preds
    df.to_csv('predictions_extrinseques.csv', index=False)
    print("Prédictions enregistrées dans predictions_extrinseques.csv")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python eval_extrinseque.py chemin_modele dossier_fichier_csv")
        sys.exit(1)

    chemin_model = sys.argv[1]
    fichier_nouveau = sys.argv[2]

    # Charger tokenizer et modèle depuis mon dossier
    tokenizer = BertTokenizer.from_pretrained(chemin_model)
    model = BertForSequenceClassification.from_pretrained(chemin_model)

    predict_on_new_data(model, tokenizer, fichier_nouveau)