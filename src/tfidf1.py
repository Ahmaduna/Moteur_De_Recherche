import math

def compute_idf(word, index, total_docs):
    """Compute the Inverse Document Frequency (IDF) for a given word."""
    doc_count = len(index[word]) if word in index else 0
    return math.log((total_docs + 1) / (doc_count + 1)) + 1  # Ajustement pour éviter les divisions par zéro

def compute_tf(word, document):
    """Compute Term Frequency (TF) for a given word in a document."""
    words = document.lower().split()
    return words.count(word) / len(words)

def refined_compute_tfidf(query, documents, index, total_docs):
    """Compute more precise TF-IDF scores using improved calculations."""
    query_words = query.lower().split()
    tfidf_scores = {}
    
    for word in query_words:
        if word in index:
            idf = compute_idf(word, index, total_docs)
            for doc in index[word]:
                tf = compute_tf(word, documents[doc])
                tfidf_scores[doc] = tfidf_scores.get(doc, 0) + tf * idf
                
    return tfidf_scores
