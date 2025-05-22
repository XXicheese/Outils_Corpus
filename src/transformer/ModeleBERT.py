import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, f1_score
import numpy as np

# Vérifier la ligne de commande
if len(sys.argv) != 2:
    print("Usage : python3 ModeleBERT.py <chemin_du_fichier_csv>")
    sys.exit(1)

fichier_csv = sys.argv[1]

def charger_donnees(fichier):
    df = pd.read_csv(fichier)
    if 'cleaned_comment' in df.columns and 'label' in df.columns:
        return df[['cleaned_comment', 'label']].dropna(subset=['cleaned_comment', 'label'])
    else:
        print(f"Colonnes attendues manquantes dans le fichier : {fichier}")
        sys.exit(1)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenizer_fonction(examples):
    return tokenizer(examples['cleaned_comment'], truncation=True, padding='max_length', max_length=128)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    f1_macro = f1_score(labels, preds, average='macro')
    print("\n" + classification_report(labels, preds,
                                      target_names=["négatif", "neutre", "positif"],
                                      zero_division=0))
    return {"accuracy": accuracy_score(labels, preds), "f1_macro": f1_macro}

df = charger_donnees(fichier_csv)

print("Distribution des labels dans tout le dataset :")
print(df['label'].value_counts(normalize=True))

df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

print("\nDistribution labels dans train :")
print(df_train['label'].value_counts(normalize=True))
print("\nDistribution labels dans test :")
print(df_test['label'].value_counts(normalize=True))

dataset_train = Dataset.from_pandas(df_train)
dataset_test = Dataset.from_pandas(df_test)

dataset_train = dataset_train.map(tokenizer_fonction, batched=True)
dataset_test = dataset_test.map(tokenizer_fonction, batched=True)

dataset_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Chargement modèle
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

args = TrainingArguments(
    output_dir="./resultats",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,    
    learning_rate=2e-5,             
    weight_decay=0.01,
    dataloader_num_workers=0,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch"
)

# Entraîneur
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Entraînement + Évaluation
trainer.train()
trainer.evaluate()

# Sauvegarde du modèle
trainer.save_model("./results")