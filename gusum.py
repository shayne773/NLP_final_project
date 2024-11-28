import nltk
import string
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

class Graph:

    def __init__(self, document) -> None:
        
        self.max_sentence_length = 0
        self.sentences = nltk.sent_tokenize(document)
        self.num_sentences = len(self.sentences)
        self.vertices = []
        self.edges = []
        self.centralities = []
        self.tfidf_dict = {}

        #initialize vertices
        for position, sentence in enumerate(self.sentences):
            vertex = Vertex(sentence=sentence, position=position)
            self.max_sentence_length = max(self.max_sentence_length, vertex.length)
            self.vertices.append(vertex)


        # Initialize edges using Sentence BERT
        model = SentenceTransformer('bert-base-nli-mean-tokens')  # Load the pre-trained model
        sentence_embeddings = model.encode(self.sentences)  # Get sentence embeddings
        # Compute similarity matrix using cosine similarity
        similarity_matrix = cosine_similarity(sentence_embeddings)
        # Convert similarity matrix into 2D array for edges
        self.edges = similarity_matrix.tolist()
        #compute sentence centralities
        self.centralities = [sum(row) for row in similarity_matrix]
        for i, vertex in enumerate(self.vertices):
            vertex.centrality = self.centralities[i]


        #initializie tf-idf dictionary
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.sentences)
        feature_names = vectorizer.get_feature_names_out()
        idf_scores = vectorizer.idf_
        self.tfidf_dict = dict(zip(feature_names, idf_scores))
        #calculate tf-idf scores for vertices
        for vertex in self.vertices:
            vertex.calculate_tfidf_score(self.tfidf_dict)

    
    #outputs the top three scored sentences in the document (will be changed later)
    def summarize(self) -> list:
        
        #each element is a tuple that contains the original sentence and its respective score

        #original features for scoring
        #scores = [(vertex.original_sentence, vertex.score_vertex(self.max_sentence_length, self.num_sentences)) for vertex in self.vertices]

        #------enable this for testing added tf-idf feature-------
        scores = [(vertex.original_sentence, vertex.score_vertex_2(self.max_sentence_length, self.num_sentences)) for vertex in self.vertices]
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
        
        #sentence features
        self.tokens = nltk.word_tokenize(sentence)
        self.length = len([token for token in self.tokens if token not in string.punctuation]) #get sentence length -> number of words in sentence
        tagged_sent = pos_tag(sentence.split())
        propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
        self.num_propernouns = len(propernouns) #get number of proper nouns in the sentence
        self.position = position #get sentence position in the document
        self.num_numerical_tokens = len([token for token in self.tokens if token.isdigit()]) #get the number of numerical tokens in sentence

    def calculate_tfidf_score(self, tfidf_dict) -> None:
        self.tfidf_score = sum(tfidf_dict.get(word, 0) for word in self.tokens)
        

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
    
    
    #same as score_vertex, but considers td-idf scores of sentneces
    def score_vertex_2(self, max_sentence_length: int, num_sentences: int) -> int:

        self.length_score = self.length/max_sentence_length if max_sentence_length!=0 else 0

        if (self.position==0 or self.position==num_sentences-1):
            self.position_score = 1
        else:    
            self.position_score = (num_sentences-self.position)/num_sentences if num_sentences!=0 else 0

        self.propernoun_score = self.num_propernouns/self.length if self.length!=0  else 0
        self.numerical_token_score = self.num_numerical_tokens/self.length if self.length!=0 else 0

        self.score = (self.length_score+self.position_score+self.propernoun_score+self.numerical_token_score)*self.centrality

        return self.score


#testing
if __name__ == "__main__":
    document = 'Technology is evolving at an unprecedented pace, reshaping industries and societies worldwide. Artificial intelligence, in particular, has transformed how we work, communicate, and solve problems. However, this rapid advancement also raises ethical questions about privacy, bias, and accountability. As we embrace innovation, it is crucial to address these challenges responsibly.'
    g = Graph(document=document)
    print(g.summarize())
    
    