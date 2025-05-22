# Projet d'Analyse de Sentiment

Dans ce projet, j’ai décidé de récupérer et poursuivre un projet en groupe, que j’avais commencé pendant le cours de Fouille de texte.

Le sujet choisi concerne l’analyse de sentiment des critiques spectateurs. Il s’agit de mesurer le degré de favorabilité des spectateurs. Pour cette analyse, le corpus provient du site **Allociné.fr**. Nous avons récupéré les critiques des spectateurs pour la série de films **Harry Potter**, composée de 8 films au total, avec leurs notations sous forme d’étoiles variant de 1 à 5.

Cependant, sur Allociné, il existe un système de demi-point (ou demi-étoile). Nous avons hésité à considérer 2,5 étoiles comme un avis neutre. Cependant, si les spectateurs estiment qu’un film suscite réellement un sentiment neutre, pourquoi prendraient-ils la peine d’attribuer une demi-étoile ? Il est probable que le film ne mérite pas ce statut neutre, et le sentiment général tendrait plutôt vers le négatif.

Finalement, nous avons décidé de classifier ces critiques en trois classes :

- **Négatif** : Notes inférieures ou égales à 3 (<= 3)  
- **Neutre** : Notes strictement supérieures à 3 mais inférieures à 4 (3 < Note < 4)  
- **Positif** : Notes supérieures ou égales à 4 (>= 4)

---

L’architecture et le modèle choisis sont **BERT** (modèle « bert-base-uncased »). Étant donné que BERT intègre une étape d’évaluation automatique, j’ai enregistré les résultats dans un fichier texte.