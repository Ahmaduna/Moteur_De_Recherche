from sklearn.metrics import precision_score, recall_score, f1_score

class Evaluation:
    def __init__(self, relevant_docs, retrieved_docs):
        self.relevant_docs = set(relevant_docs)
        self.retrieved_docs = set(retrieved_docs)

    def precision_at_k(self, k):
        """Calcule la précision pour les k premiers résultats"""
        retrieved_at_k = list(self.retrieved_docs)[:k]
        relevant_at_k = self.relevant_docs.intersection(retrieved_at_k)
        precision = len(relevant_at_k) / len(retrieved_at_k) if retrieved_at_k else 0
        return precision

    def recall_at_k(self, k):
        """Calcule le rappel pour les k premiers résultats"""
        retrieved_at_k = list(self.retrieved_docs)[:k]
        relevant_at_k = self.relevant_docs.intersection(retrieved_at_k)
        recall = len(relevant_at_k) / len(self.relevant_docs) if self.relevant_docs else 0
        return recall

    def f1_at_k(self, k):
        """Calcule la F-mesure pour les k premiers résultats"""
        precision = self.precision_at_k(k)
        recall = self.recall_at_k(k)
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
