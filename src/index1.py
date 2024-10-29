from collections import defaultdict
import string
import os
import json
from tfidf1 import refined_compute_tfidf
from search1 import enhanced_compute_combined_score  
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Télécharger les stopwords et les ressources nécessaires avec NLTK
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('french'))
lemmatizer = WordNetLemmatizer()

class Indexer:
    def __init__(self, documents_path):
        self.documents_path = documents_path
        self.index = defaultdict(list)  # Dictionnaire de type {mot: [liste des fichiers]}
        self.documents = {}  # Dictionnaire contenant le contenu des documents

    def clean_text(self, text):
        """Nettoyer le texte en supprimant la ponctuation, les stop words, et en le mettant en minuscules avec lemmatisation"""
        translator = str.maketrans('', '', string.punctuation)
        words = text.translate(translator).lower().split()
        # Supprimer les stop words et appliquer la lemmatisation
        cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 1]
        return ' '.join(cleaned_words)

    def build_index(self):
        """Construire l'index inversé"""
        for filename in os.listdir(self.documents_path):
            if filename.endswith(".txt"):
                with open(os.path.join(self.documents_path, filename), 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Nettoyer le contenu et le stocker
                    cleaned_content = self.clean_text(content)
                    self.documents[filename] = cleaned_content
                    # Créer l'index inversé
                    words = set(cleaned_content.split())
                    for word in words:
                        self.index[word].append(filename)

    def display_sample_index(self, num_terms=5):
        """Afficher un échantillon de l'index pour vérification"""
        print(f"Nombre total de termes indexés : {len(self.index)}")
        for term, docs in list(self.index.items())[:num_terms]:
            print(f"Terme: '{term}' - Documents: {docs[:5]}")  # Affiche les 5 premiers documents pour chaque terme

# Charger l'index
indexer = Indexer("C:/Users/ahmad/Desktop/ET5/IE/Projet_IE/data/wiki_split_extract_2k")
indexer.build_index()

# Afficher un échantillon de l'index pour vérifier si tout fonctionne correctement
indexer.display_sample_index()

# Charger les requêtes et les réponses attendues depuis le fichier `requetes.jsonl`
query_file_path = "C:/Users/ahmad/Desktop/ET5/IE/Projet_IE/data/requetes.jsonl"
queries = []

# Lire et ignorer les lignes mal formatées
with open(query_file_path, 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file, 1):
        try:
            queries.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Erreur JSON à la ligne {line_number}: {e}")
            print(f"Ligne problématique: {line}")

# Fonction pour évaluer la précision du moteur de recherche
def evaluate_search_engine(indexer, queries):
    correct_matches = 0
    total_queries = len(queries)
    
    for query_data in queries:
        expected_file = query_data["Answer file"]
        query_list = query_data["Queries"]
        
        # Utiliser plusieurs phrases de la requête pour le test
        best_result = None
        highest_score = float('-inf')

        for query in query_list:
            # Calculer les scores TF-IDF pour la requête
            tfidf_scores = refined_compute_tfidf(query, indexer.documents, indexer.index, len(indexer.documents))
            
            # Combiner les scores avec la similarité cosinus
            results = enhanced_compute_combined_score(query, indexer.documents, tfidf_scores, weight_tfidf=0.5, weight_vector=0.5)
            
            # Prendre le meilleur score parmi les variantes de requêtes
            if results and results[0][1] > highest_score:
                highest_score = results[0][1]
                best_result = results[0][0]

        # Vérifier si le meilleur résultat correspond au fichier attendu
        if best_result == expected_file:
            correct_matches += 1
        
        # Afficher un résumé pour chaque requête
        print(f"Requête: '{query_list[0]}'")
        print(f"Fichier attendu: '{expected_file}'")
        print(f"Top fichier trouvé: '{best_result}'\n")

    # Calcul de la précision globale
    accuracy = correct_matches / total_queries * 100
    print(f"\nPrécision du moteur de recherche: {accuracy:.2f}%")

# Évaluer le moteur de recherche avec les requêtes chargées
evaluate_search_engine(indexer, queries)
