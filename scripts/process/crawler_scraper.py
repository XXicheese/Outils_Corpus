import requests
from bs4 import BeautifulSoup
import sys
import time
import csv
import argparse

# Commande: python crawler3.0_csv.py https://www.allocine.fr/film/fichefilm-293908/critiques/spectateurs/ 2 200 mid mid_commit.csv
# 2 -> nb de page, 200 -> nb de commantaire minimale, min -> class de label


def fetch_reviews(movie_url, max_pages, min_length, label_type, total_reviews):
    # Utiliser un 'user-agent' pour éviter d’être bloqué
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }

    comments_data = []
    page = 1
    reviews_collected = 0
    # Utiliser une boucle while
    while True:
        url = f"{movie_url}?page={page}"
        print(f"Téléchargement de la page {page} : {url}")
        # Vérifier si la pagepeut être demandée correctement; sinon, sortir de la boucle
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Impossible d’accéder à la page (code {response.status_code})")
            break
        # Utiliser BeautifulSoup pour examiner le résultat de l’analyse
        soup = BeautifulSoup(response.text, 'html.parser')
        # Rechercher tous les blocs de critiques correspondants sur la page, c'est le class: 'hred review-card cf'
        review_cards = soup.find_all('div', class_='hred review-card cf')

        if not review_cards:
            print("Plus de critiques trouvées.")
            break

        for card in review_cards:
            # Lire la note de chaque critique
            note_tag = card.find('span', class_='stareval-note')
            # Si la critique n’a pas de note, l’ignorer
            if not note_tag:
                continue
            # Convertir en float et normaliser la ponctuation
            note = float(note_tag.get_text(strip=True).replace(',', '.'))
            # Extraire des critiques
            comment_tag = card.find('div', class_='content-txt review-card-content')
            if not comment_tag:
                continue
            comment = comment_tag.get_text(strip=True)
            # Filtrer les critiques selon le label
            if len(comment) >= min_length:
                if note >= 4.0:
                    label = 'pos'
                elif note >= 3.0:
                    label = 'mid'
                else:
                    label = 'neg'

                if label == label_type:
                    comments_data.append([note, comment, label])
                    reviews_collected += 1

                    if reviews_collected >= total_reviews:
                        print(f"{total_reviews} critiques récupérées, arrêt du programme.")
                        return comments_data

        if max_pages and page >= max_pages:
            break

        page += 1
        time.sleep(1)

    return comments_data

# Écrire dans le fichier csv
def save_to_csv(comments_data, file):
    with open(file, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(['Note', 'Comment', 'Class'])
        writer.writerows(comments_data)
    print(f"Les critiques ont été enregistrées dans le fichier {file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Allociné scraper : récupère les critiques par label")

    parser.add_argument("url", help="URL de la page des critiques")
    parser.add_argument("pages_max", type=int, help="Nombre maximal de pages à parcourir")
    parser.add_argument("min_length", type=int, help="Longueur minimale de chaque commentaire") 
    parser.add_argument("label_type", choices=["pos", "mid", "neg"], help="Filtrer des critiques par label")
    parser.add_argument("output_csv", help="Fichier CSV de sortie") 
    parser.add_argument("--total_reviews", type=int, default=30, help="Nombre total de critiques à récupérer (par défaut 30)")
    args = parser.parse_args()

    all_comments = fetch_reviews(
        movie_url=args.url.strip('/'),
        max_pages=args.pages_max,
        min_length=args.min_length,
        label_type=args.label_type,
        total_reviews=args.total_reviews
    )
    print(f"\nTotal de critiques récupérées : {len(all_comments)}")
    save_to_csv(all_comments, args.output_csv)
