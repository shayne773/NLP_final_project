import nltk
import string
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import time
import torch


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
  "numeric": 0.1,
  "tfidf": 0.2,
  "noun_phrase": 0.3
}


def extract_noun_phrases(tagged_sent):
    grammar = "NP: {<DT>?<JJ>*<NN.*>+}"  # Define the grammar for noun phrases
    cp = nltk.RegexpParser(grammar=grammar)
    tree = cp.parse(tagged_sent)
    noun_phrases = []
    for subtree in tree.subtrees(filter=lambda x: x.label() == "NP"):
        np = " ".join(word for word, tag in subtree.leaves())  # Extract the words from the leaves
        noun_phrases.append(np)
    return noun_phrases

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

        #initialize vertices
        for position, sentence in enumerate(self.sentences):
            vertex = Vertex(sentence=sentence, position=position)
            self.num_noun_phrases += len(vertex.noun_phrases)
            self.max_sentence_length = max(self.max_sentence_length, vertex.length)
            self.vertices.append(vertex)

        sentence_embeddings = model.encode(self.sentences, batch_size=32, show_progress_bar=False)  # Batch encoding
        
        # Compute similarity matrix using cosine similarity
        similarity_matrix = cosine_similarity(sentence_embeddings)
        self.edges = similarity_matrix.tolist()  # Convert similarity matrix into 2D array for edges

        # Compute sentence centralities
        self.centralities = [sum(row) for row in similarity_matrix]
        
        '''
        # Initialize edges using Sentence BERT
        model = SentenceTransformer('bert-base-nli-mean-tokens')  # Load the pre-trained model
        sentence_embeddings = model.encode(self.sentences)  # Get sentence embeddings
        # Compute similarity matrix using cosine similarity
        similarity_matrix = cosine_similarity(sentence_embeddings)
        # Convert similarity matrix into 2D array for edges
        self.edges = similarity_matrix.tolist()
        #compute sentence centralities
        self.centralities = [sum(row) for row in similarity_matrix]
        
        '''
        for i, vertex in enumerate(self.vertices):
            vertex.centrality = self.centralities[i]

        

        '''
        # Transform sentences into TF-IDF representation
        tfidf_matrix = vectorizer.transform(self.sentences)
        self.sentence_tfidf_scores = tfidf_matrix.sum(axis=1).A1
        #get the max tfidf score for normalization
        max_tfidf_score = max(self.sentence_tfidf_scores) if self.sentence_tfidf_scores.size>0 else 1
        #calculate tf-idf scores for vertices
        for i, vertex in enumerate(self.vertices):
            vertex.tfidf_score = self.sentence_tfidf_scores[i]/max_tfidf_score  
        '''
        
        


    
    #outputs the top three scored sentences in the document (will be changed later)
    def summarize(self) -> list:
        
        #each element is a tuple that contains the original sentence and its respective score

        #original features for scoring
        #scores = [(vertex.original_sentence, vertex.score_vertex(self.max_sentence_length, self.num_sentences)) for vertex in self.vertices]

        #------enable this for testing added tf-idf feature-------
        scores = [(vertex.original_sentence, vertex.score_vertex_2(self.max_sentence_length, self.num_sentences, self.num_noun_phrases)) for vertex in self.vertices]
        #---------------------------------------------------------
        sorted_scores = sorted(scores, key = lambda x : x[1], reverse = True)
        return [sorted_scores[i] for i in range(3)]
        

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
        
        #sentence features
        self.tokens = nltk.word_tokenize(sentence)
        self.length = len([token for token in self.tokens if token not in string.punctuation]) #get sentence length -> number of words in sentence
        tagged_sent = pos_tag(self.tokens)
        propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
        self.num_propernouns = len(propernouns) #get number of proper nouns in the sentence
        self.position = position #get sentence position in the document
        self.num_numerical_tokens = len([token for token in self.tokens if token.isdigit()]) #get the number of numerical tokens in sentence
        self.noun_phrases = extract_noun_phrases(tagged_sent) #get the noun phfases in the sentence

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
    
    
    #same as score_vertex, but considers td-idf scores of sentneces. Also assigns 
    def score_vertex_2(self, max_sentence_length: int, num_sentences: int, num_noun_phrases: int) -> int:

        self.length_score = self.length/max_sentence_length if max_sentence_length!=0 else 0

        if (self.position==0 or self.position==num_sentences-1):
            self.position_score = 1
        else:    
            self.position_score = (num_sentences-self.position)/num_sentences if num_sentences!=0 else 0

        self.propernoun_score = self.num_propernouns/self.length if self.length!=0  else 0
        self.numerical_token_score = self.num_numerical_tokens/self.length if self.length!=0 else 0
        self.noun_phrase_score = len(self.noun_phrases)/num_noun_phrases

        self.score = (self.length_score*weights["length"]
                      +self.position_score*weights["position"]
                      +self.propernoun_score*weights["proper noun"]
                      +self.noun_phrase_score*weights["noun_phrase"])*self.centrality

        return self.score


#testing
if __name__ == "__main__":

    start_time = time.time()
    for i in range(1):

        document = 'Technology is evolving at an unprecedented pace, reshaping industries and societies worldwide. Artificial intelligence, in particular, has transformed how we work, communicate, and solve problems. However, this rapid advancement also raises ethical questions about privacy, bias, and accountability. As we embrace innovation, it is crucial to address these challenges responsibly.'
        g = Graph(document=document)
        print(g.summarize())
    print(time.time()-start_time)
    
    