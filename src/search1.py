import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords

# Charger les stopwords
stop_words = set(stopwords.words('french'))

# Charger le modèle de vecteurs de mots FastText
word_vectors = KeyedVectors.load_word2vec_format("C:/Users/ahmad/Desktop/ET5/IE/Projet_IE/models/wiki-news-300d-1M-subword.vec", binary=False)

def enhanced_get_word_vector(word):
    """Obtenir un vecteur de mot depuis le modèle FastText pré-entraîné, retourne None si le mot n'existe pas"""
    try:
        return word_vectors[word]
    except KeyError:
        return None

def is_valid_vector(vector):
    """Vérifie si un vecteur est valide (pas vide et ne contient pas de NaN)"""
    return vector is not None and np.all(np.isfinite(vector)) and not np.isnan(np.sum(vector))

def compute_mean_vector(word_vectors_list):
    """Calcule la moyenne des vecteurs, avec une vérification pour éviter les listes vides"""
    valid_vectors = [vec for vec in word_vectors_list if vec is not None and is_valid_vector(vec)]
    if len(valid_vectors) == 0:
        return None  # Retourne None si aucune valeur valide n'est trouvée
    return np.mean(valid_vectors, axis=0)

def enhanced_compute_combined_score(query, documents, tfidf_scores, weight_tfidf=0.5, weight_vector=0.3, weight_bm25=0.2):
    """Combiner les scores TF-IDF, BM25 et Cosine Similarity avec des pondérations ajustées."""
    combined_scores = {}

    # Préparer le BM25
    tokenized_docs = [doc.lower().split() for doc in documents.values()]
    bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)

    # Traitement de la requête (supprimer les stop words)
    query_tokens = [word for word in query.lower().split() if word not in stop_words]

    for doc, tfidf_score in tfidf_scores.items():
        # Obtenir le score BM25
        doc_index = list(documents.keys()).index(doc)
        bm25_score = bm25.get_scores(query_tokens)[doc_index]

        # Calculer la similarité cosinus
        doc_tokens = [word for word in documents[doc].lower().split() if word not in stop_words]
        query_vector = compute_mean_vector([enhanced_get_word_vector(word) for word in query_tokens])
        doc_vector = compute_mean_vector([enhanced_get_word_vector(word) for word in doc_tokens])
        
        if query_vector is None or doc_vector is None:
            cosine_sim = 0  # Évite le calcul si les vecteurs sont invalides
        else:
            cosine_sim = cosine_similarity([query_vector], [doc_vector])[0][0]

        # Combiner les scores avec des pondérations ajustées
        combined_score = (weight_tfidf * tfidf_score) + (weight_vector * cosine_sim) + (weight_bm25 * bm25_score)
        combined_scores[doc] = combined_score

    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
