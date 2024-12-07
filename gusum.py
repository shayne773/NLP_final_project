import nltk
import string
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import time
import torch
import spacy
import pytextrank


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(f"Using device: {device}")
# Initialize SentenceTransformer with faster model and GPU
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens', device=device)  # Load faster model on GPU

# Load the pre-trained vectorizer

with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)


#feature weights
weights = {
  "length" : 0.2,
  "position": 0.35,
  "proper noun": 0.45,
  "keyword": 0.2,
  "numeric": 0,
  "tfidf": 0.3,
  "noun_phrase": 0
}


def extract_textrank_keywords(document, num_keywords) -> list:
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")
    doc = nlp(document)
    keywords = [phrase.text.lower() for phrase in doc._.phrases[:num_keywords]] 
    return keywords

class Graph:

    def __init__(self, document) -> None:
        
        self.max_sentence_length = 0
        self.sentences = nltk.sent_tokenize(document)
        self.num_sentences = len(self.sentences)
        self.vertices = []
        self.edges = []
        self.centralities = []
        self.tfidf_dict = {}
        self.sentence_tfidf_scores = []
        self.num_noun_phrases = 0
        self.keywords = extract_textrank_keywords(document=document, num_keywords=15)
        #self.keywords = []

        #initialize vertices
        for position, sentence in enumerate(self.sentences):
            vertex = Vertex(sentence=sentence, position=position)
            self.num_noun_phrases += len(vertex.noun_phrases)
            self.max_sentence_length = max(self.max_sentence_length, vertex.length)
            self.vertices.append(vertex)

        self.sentence_embeddings = model.encode(self.sentences, batch_size=32, show_progress_bar=False)  # Batch encoding
        
        # Compute similarity matrix using cosine similarity
        similarity_matrix = cosine_similarity(self.sentence_embeddings)
        self.edges = similarity_matrix.tolist()  # Convert similarity matrix into 2D array for edges

        # Compute sentence centralities
        self.centralities = [sum(row) for row in similarity_matrix]
        
        for i, vertex in enumerate(self.vertices):
            vertex.centrality = self.centralities[i]
        
        # Compute TF-IDF scores using the loaded vectorizer
        # Transform all sentences in the document at once
        tfidf_matrix = vectorizer.transform(self.sentences)

        # For each sentence (vertex), compute the sum of its TF-IDF terms and normalize by sentence length
        for i, vertex in enumerate(self.vertices):
            tfidf_values = tfidf_matrix[i].toarray()[0]  # TF-IDF vector for this sentence
            tfidf_sum = tfidf_values.sum()

            # Normalize by sentence length to avoid bias towards longer sentences
            normalized_tfidf = tfidf_sum / vertex.length if vertex.length != 0 else 0.0
            vertex.tfidf_score = normalized_tfidf
        
    
    #outputs the top three scored sentences in the document (will be changed later)
    def summarize(self) -> list:
        
        #each element is a tuple that contains the original sentence and its respective score

        #original features for scoring
        #scores = [(vertex.original_sentence, vertex.score_vertex(self.max_sentence_length, self.num_sentences)) for vertex in self.vertices]

        #------enable this for testing added tf-idf feature-------
        scores = [(vertex.original_sentence, vertex.score_vertex_2(self.max_sentence_length, self.num_sentences, self.num_noun_phrases, self.keywords)) for vertex in self.vertices]
        #---------------------------------------------------------
        sorted_scores = sorted(scores, key = lambda x : x[1], reverse = True)
        return [sorted_scores[i] for i in range(3)]
    

    def summarize_with_diversity(self, similarity_threshold=0.8, top_k=3) -> list:

        scores = [(i, vertex.original_sentence, vertex.score_vertex_2(self.max_sentence_length, self.num_sentences, self.num_noun_phrases, self.keywords)) 
                for i, vertex in enumerate(self.vertices)]
        sorted_scores = sorted(scores, key=lambda x: x[2], reverse=True)

        chosen_sentences = []
        chosen_indices = []

        for i, orig_sentence, score in sorted_scores:
            # If no sentence chosen yet, select the top one directly
            if not chosen_indices:
                chosen_sentences.append((orig_sentence, score))
                chosen_indices.append(i)
            else:
                # Check this sentence's similarity to already chosen sentences using precomputed embeddings
                current_embedding = self.sentence_embeddings[i]
                is_similar = False

                for chosen_idx in chosen_indices:
                    chosen_embedding = self.sentence_embeddings[chosen_idx]
                    sim = cosine_similarity([current_embedding], [chosen_embedding])[0][0]
                    if sim >= similarity_threshold:
                        is_similar = True
                        break

                # If not too similar to any chosen sentence, add it
                if not is_similar:
                    chosen_sentences.append((orig_sentence, score))
                    chosen_indices.append(i)

            if len(chosen_sentences) == top_k:
                break

        return chosen_sentences
        

#a Vertex class represents a sentence in the document
class Vertex:
    
    def __init__(self, sentence: str, position: int) -> None:
        
        self.centrality = 0
        self.original_sentence = sentence
        self.length_score = 0
        self.position_score = 0
        self.propernoun_score = 0
        self.score = 0
        self.tfidf_score = 0
        self.noun_phrase_score = 0
        self.noun_phrases = []
        self.keywords_score = 0
        self.tokens = nltk.word_tokenize(sentence)

        #sentence features
        self.length = len([token for token in self.tokens if token not in string.punctuation]) #get sentence length -> number of words in sentence
        tagged_sent = pos_tag(self.tokens)
        propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
        self.num_propernouns = len(propernouns) #get number of proper nouns in the sentence
        self.position = position #get sentence position in the document
        self.num_numerical_tokens = len([token for token in self.tokens if token.isdigit()]) #get the number of numerical tokens in sentence

    def calculate_textrank_score(self, keywords) -> int:
        return sum(1 for token in self.tokens if token.lower() in keywords)

    def calculate_tfidf_score(self, tfidf_dict) -> None:
        self.tfidf_score = sum(tfidf_dict.get(word, 0) for word in self.tokens if word.isalnum())
        

    def score_vertex(self, max_sentence_length: int, num_sentences: int) -> int:

        self.length_score = self.length/max_sentence_length if max_sentence_length!=0 else 0

        if (self.position==0 or self.position==num_sentences-1):
            self.position_score = 1
        else:    
            self.position_score = (num_sentences-self.position)/num_sentences if num_sentences!=0 else 0

        self.propernoun_score = self.num_propernouns/self.length if self.length!=0  else 0
        self.numerical_token_score = self.num_numerical_tokens/self.length if self.length!=0 else 0

        self.score = (self.length_score+self.position_score+self.propernoun_score+self.numerical_token_score)*self.centrality

        return self.score
    
    
    
    def score_vertex_2(self, max_sentence_length: int, num_sentences: int, num_noun_phrases: int, keywords: list) -> int:

        self.length_score = self.length/max_sentence_length if max_sentence_length!=0 else 0

        if (self.position==0 or self.position==num_sentences-1):
            self.position_score = 1
        else:    
            self.position_score = (num_sentences-self.position)/num_sentences if num_sentences!=0 else 0

        self.propernoun_score = self.num_propernouns/self.length if self.length!=0  else 0
        #self.numerical_token_score = self.num_numerical_tokens/self.length if self.length!=0 else 0
        max_keywords_score = len(keywords)
        self.keywords_score = self.calculate_textrank_score(keywords=keywords)/max_keywords_score if max_keywords_score !=0 else 0

        self.score = (self.length_score*weights["length"]
                      +self.position_score*weights["position"]
                      +self.propernoun_score*weights["proper noun"]
                      +self.keywords_score*weights["keyword"]
                      +self.tfidf_score*weights["tfidf"])*self.centrality

        return self.score
    


#testing
if __name__ == "__main__":

    start_time = time.time()
    for i in range(1):

        document = 'Technology is evolving at an unprecedented pace, reshaping industries and societies worldwide. Artificial intelligence, in particular, has transformed how we work, communicate, and solve problems. However, this rapid advancement also raises ethical questions about privacy, bias, and accountability. As we embrace innovation, it is crucial to address these challenges responsibly.'
        g = Graph(document=document)
        print(g.summarize_with_diversity())
    print(time.time()-start_time)
    
    