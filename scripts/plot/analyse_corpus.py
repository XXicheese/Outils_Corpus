import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import numpy as np
import matplotlib

# Détection colonne
def trouver_colonne_texte(df, noms_possibles=None):
    if noms_possibles is None:
        noms_possibles = ["cleaned_comment"]
    colonnes_df = [col.lower() for col in df.columns]
    for nom in noms_possibles:
        if nom.lower() in colonnes_df:
            index = colonnes_df.index(nom.lower())
            return df.columns[index]
    return None

# Fonction principale : lire le csv et appliquer des analyses textuelles
def analyser_corpus(dossier, visualisations=None):
    corpus_par_fichier = dict()
    corpus_par_classe = dict()

    for filename in os.listdir(dossier):
        if filename.endswith(".csv"):
            path = os.path.join(dossier, filename)
            df = pd.read_csv(path)

            colonne_texte = trouver_colonne_texte(df)
            if colonne_texte is None:
                print(f"Aucune colonne texte reconnue dans {filename} (colonnes trouvées: {df.columns.tolist()})")
                continue

            textes = df[colonne_texte].dropna().astype(str).tolist()
            corpus_par_fichier[filename] = textes

            if 'class' in df.columns:
                for classe, group_df in df.groupby('class'):
                    textes_classe = group_df[colonne_texte].dropna().astype(str).tolist()
                    corpus_par_classe.setdefault(classe, []).extend(textes_classe)

    total_textes = sum(len(texts) for texts in corpus_par_fichier.values())
    print(f"{total_textes} textes chargés dans {len(corpus_par_fichier)} fichiers.")

    if total_textes == 0:
        print("Erreur : aucun texte chargé. Vérifie le contenu de tes fichiers.")
        return

    corpus = [text for textes in corpus_par_fichier.values() for text in textes]

    # Visualisation possible, si aucune précision, on les active toutes par défaut
    if visualisations is None:
        visualisations = {"longueur", "mots_frequents", "zipf", "wordcloud"}
    else:
        visualisations = set(visualisations)

    os.makedirs("figures", exist_ok=True)

    # Histogramme : longeurs de texte par mots
    if "longueur" in visualisations:
        plt.figure(figsize=(12, 7))
        bins = 30
        colors = matplotlib.colormaps['tab10'].colors

        for i, (filename, textes) in enumerate(corpus_par_fichier.items()):
            text_lengths = [len(text.split()) for text in textes]
            plt.hist(text_lengths, bins=bins, alpha=0.5, label=filename, color=colors[i % len(colors)])

        plt.title("Distribution des longueurs des textes par fichier")
        plt.xlabel("Nombre de mots")
        plt.ylabel("Nombre de textes")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/hist_longueur_par_fichier.png")
        plt.close()
        print("Histogramme des longueurs sauvegardé dans 'figures/hist_longueur_par_fichier.png'")

    # Mots les plus fréquents: global et/ou par classe
    if "mots_frequents" in visualisations:
        all_words = " ".join(corpus).lower().split()
        word_freq = Counter(all_words)
        most_common = word_freq.most_common(30)

        plt.figure(figsize=(12, 6))
        if most_common:
            words, counts = zip(*most_common)
            plt.bar(words, counts, color='coral')
            plt.xticks(rotation=45)
        else:
            print("Pas de mots pour afficher le graphique des mots fréquents.")
        plt.title("30 mots les plus fréquents (global)")
        plt.tight_layout()
        plt.savefig("figures/mots_frequents.png")
        plt.close()
        print("Graphique des mots fréquents sauvegardé dans 'figures/mots_frequents.png'")

        # par classe
        if corpus_par_classe:
            mots_par_classe = {}
            all_top_words = set()
            for classe, textes_classe in corpus_par_classe.items():
                all_words_classe = " ".join(textes_classe).lower().split()
                word_freq_classe = Counter(all_words_classe)
                top30 = [w for w, _ in word_freq_classe.most_common(30)]
                mots_par_classe[classe] = word_freq_classe
                all_top_words.update(top30)

            all_top_words = sorted(all_top_words)
            data = np.array([[mots_par_classe[classe][mot] for classe in mots_par_classe] for mot in all_top_words])
            x = np.arange(len(all_top_words))
            width = 0.8 / len(mots_par_classe)
            colors = matplotlib.colormaps['tab10']

            plt.figure(figsize=(max(12, len(all_top_words) * 0.3), 7))
            for i, classe in enumerate(mots_par_classe):
                plt.bar(x + i * width, data[:, i], width=width, label=str(classe), color=colors(i))

            plt.xticks(x + width * (len(mots_par_classe) - 1) / 2, all_top_words, rotation=90)
            plt.ylabel("Fréquence")
            plt.title("30 mots les plus fréquents par classe")
            plt.legend()
            plt.tight_layout()
            plt.savefig("figures/mots_frequents_par_classe.png")
            plt.close()
            print("Graphique des mots fréquents par classe sauvegardé dans 'figures/mots_frequents_par_classe.png'")

    # Loi de Zipf
    if "zipf" in visualisations:
        all_words = " ".join(corpus).lower().split()
        word_freq = Counter(all_words)
        frequencies = np.array(sorted(word_freq.values(), reverse=True))
        ranks = np.arange(1, len(frequencies) + 1)

        if len(frequencies) > 0:
            plt.figure()
            plt.plot(ranks, frequencies)
            plt.xscale("log")
            plt.yscale("log")
            plt.title("Loi de Zipf")
            plt.xlabel("Rang")
            plt.ylabel("Fréquence")
            plt.tight_layout()
            plt.savefig("figures/zipf.png")
            plt.close()
            print("Graphique loi de Zipf sauvegardé dans 'figures/zipf.png')")
        else:
            print("Pas de données pour le graphique loi de Zipf.")

    # Nuage de mots
    if "wordcloud" in visualisations:
        all_words = " ".join(corpus).lower().split()
        if len(all_words) > 0:
            wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(" ".join(all_words))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("figures/wordcloud.png")
            plt.close()
            print("Word cloud sauvegardé dans 'figures/wordcloud.png')")
        else:
            print("Pas de données pour générer le word cloud.")

    # Taux de hapax sur le terminal
    if "mots_frequents" in visualisations or "zipf" in visualisations:
        hapax_count = len([w for w, c in Counter(" ".join(corpus).lower().split()).items() if c == 1])
        hapax_rate = hapax_count / len(set(" ".join(corpus).lower().split())) if len(corpus) > 0 else 0
        print(f"Taux de hapax : {hapax_rate:.2%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyse de corpus CSV avec visualisations.")
    parser.add_argument("chemin", help="Chemin vers le dossier contenant les fichiers CSV")
    parser.add_argument("--visualisations", "-v", help="Types de visualisations séparées par des virgules (longueur,mots_frequents,zipf,wordcloud)", default=None)
    args = parser.parse_args()

    visus = args.visualisations.split(",") if args.visualisations else None
    analyser_corpus(args.chemin, visus)
