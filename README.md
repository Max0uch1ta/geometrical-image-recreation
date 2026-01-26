# Recréation Géométrique d'Images

Application Python pour recréer des images en utilisant des formes géométriques via des algorithmes d'optimisation.

**NOTE:** Le rapport est disponible dans le fichier `docs/rapport.md`.

## Installation

```bash
pip install -r requirements.txt
```

## Dépendances

| Package | Version | Description |
|---------|---------|-------------|
| numpy | ≥1.20.0 | Calculs numériques |
| Pillow | ≥9.0.0 | Traitement d'images |
| tqdm | ≥4.60.0 | Barres de progression |
| pandas | - | Tableaux de résumé (notebook) |
| matplotlib | - | Graphiques (notebook) |

**NOTE:** Je n'ai pas demander si vous aviez tqdm d'installer donc j'ai ajouté un bloc try/except pour éviter les erreurs si tqdm n'est pas installé. Vous n'aurez pas de barres de progression si tqdm n'est pas installé.

## Utilisation

### Ligne de commande

```bash
# Hill Climbing (recherche locale)
python main.py -i image.png -a hill_climbing -n 1000

# Recuit Simulé
python main.py -i image.png -a simulated_annealing -n 5000

# Algorithme Génétique (sélection par tournoi)
python main.py -i image.png -a ga_tournament -p 50 -n 500

# Algorithme Génétique (sélection gloutonne)
python main.py -i image.png -a ga_greedy -p 50 -n 500
```

### Options principales

| Option | Court | Description |
|--------|-------|-------------|
| `--image` | `-i` | Chemin vers l'image cible |
| `--algorithm` | `-a` | Algorithme : `hill_climbing`, `simulated_annealing`, `ga_greedy`, `ga_tournament` |
| `--max-iter` | `-n` | Nombre maximum d'itérations |
| `--max-time` | `-T` | Temps maximum en minutes |
| `--shape` | `-s` | Type de forme : `rect`, `circle`, `ellipse`, `triangle` |
| `--pop-size` | `-p` | Taille de la population (AG) |
| `--fitness` | `-f` | Métrique : `l1` (Manhattan) ou `l2` (SSD) |
| `--heuristic` | `-H` | Initialisation heuristique (AG) |
| `--use-opacity` | `-o` | Activer l'opacité (50%-100%) aléatoire sur les formes |
| `--track-by-time` | `-t` | Suivi par temps écoulé |
| `--min-size` | - | Taille minimale des formes (px) |
| `--show-loss` | - | Afficher le graphique de fitness (Matplotlib) |
| `--print-history` | - | Afficher l'historique tabulaire dans la console |

### Exemples avancés

```bash
# Exécution avec limite de temps (5 minutes)
python main.py -i image.png -a ga_tournament -T 5 -n 100000 -t

# Avec formes elliptiques et opacité
python main.py -i image.png -a hill_climbing -s ellipse --use-opacity

# AG avec initialisation heuristique et métrique L1
python main.py -i image.png -a ga_tournament --heuristic -f l1
```

## Résultats

Les images générées sont automatiquement sauvegardées dans le dossier `./output`.
Le nom du fichier inclut l'algorithme utilisé et un horodatage.

## Notebook de comparaison

Le fichier `comparison_notebook.ipynb` permet de comparer tous les algorithmes avec différentes configurations.

## Structure du projet

```
├── main.py          # Point d'entrée CLI
├── algorithms.py    # Implémentation des algorithmes
├── genotype.py      # Classes de formes (Rect, Circle, Ellipse)
├── mutation.py      # Opérations de mutation
├── utils.py         # Fonctions utilitaires
└── comparison_notebook.ipynb  # Notebook de comparaison
```

## Algorithmes

- **Hill Climbing** : Recherche locale acceptant uniquement les améliorations
- **Simulated Annealing** : Recherche locale avec acceptation probabiliste
- **AG Glouton** : Sélection par troncature des meilleurs individus
- **AG Tournoi** : Sélection par compétition entre individus aléatoires
