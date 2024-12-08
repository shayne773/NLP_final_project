import nltk
import string
from nltk.tag import pos_tag
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize SentenceTransformer
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens', device=device)

# Feature weights
weights = {
    "length": 0.1,
    "position": 0.1,
    "proper noun": 0.1,
    "numeric": 0.45
}

class Graph:
    def __init__(self, document) -> None:
        self.sentences = nltk.sent_tokenize(document)
        self.num_sentences = len(self.sentences)
        self.max_sentence_length = 0
        self.vertices = []

        # Initialize vertices
        for position, sentence in enumerate(self.sentences):
            vertex = Vertex(sentence=sentence, position=position)
            self.max_sentence_length = max(self.max_sentence_length, vertex.length)
            self.vertices.append(vertex)

        # Compute sentence embeddings and similarity matrix
        sentence_embeddings = model.encode(self.sentences, batch_size=32, show_progress_bar=False)
        similarity_matrix = cosine_similarity(sentence_embeddings)

        # Compute centralities
        centralities = [sum(row) for row in similarity_matrix]
        for i, vertex in enumerate(self.vertices):
            vertex.centrality = centralities[i]

    def summarize(self) -> list:
        # Compute scores for each sentence
        scores = [
            (vertex.original_sentence, vertex.score_vertex(self.max_sentence_length, self.num_sentences))
            for vertex in self.vertices
        ]
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return [sorted_scores[i] for i in range(min(3, len(sorted_scores)))]


class Vertex:
    def __init__(self, sentence: str, position: int) -> None:
        self.original_sentence = sentence
        self.position = position
        self.centrality = 0

        # Tokenize sentence and compute features
        self.tokens = nltk.word_tokenize(sentence)
        self.length = len([token for token in self.tokens if token not in string.punctuation])
        tagged_sent = pos_tag(self.tokens)
        propernouns = [word for word, pos in tagged_sent if pos == 'NNP']
        self.num_propernouns = len(propernouns)
        self.num_numerical_tokens = len([token for token in self.tokens if token.isdigit()])

    def score_vertex(self, max_sentence_length: int, num_sentences: int) -> float:
        self.length_score = self.length / max_sentence_length if max_sentence_length != 0 else 0
        self.position_score = 1 if self.position == 0 or self.position == num_sentences - 1 else (
            (num_sentences - self.position) / num_sentences if num_sentences != 0 else 0
        )
        self.propernoun_score = self.num_propernouns / self.length if self.length != 0 else 0
        self.numerical_token_score = self.num_numerical_tokens / self.length if self.length != 0 else 0

        # Compute final score based on weights
        self.score = (
            self.length_score * weights["length"] +
            self.position_score * weights["position"] +
            self.propernoun_score * weights["proper noun"] +
            self.numerical_token_score * weights["numeric"]
        ) * self.centrality

        return self.score


# Testing
if __name__ == "__main__":
    document = (
        "Technology is evolving at an unprecedented pace, reshaping industries and societies worldwide. "
        "Artificial intelligence, in particular, has transformed how we work, communicate, and solve problems. "
        "However, this rapid advancement also raises ethical questions about privacy, bias, and accountability. "
        "As we embrace innovation, it is crucial to address these challenges responsibly."
    )

    g = Graph(document=document)
    print(g.summarize())


print('gusum installed')