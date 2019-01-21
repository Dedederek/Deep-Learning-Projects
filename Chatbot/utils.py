import nltk
import pickle
import re
import numpy as np
from gensim.models import KeyedVectors

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'starspace_embeddings.txt',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file,txt file that can be loaded by gensim

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=False)
    with open(embeddings_path,'r') as f:
        dimension=f.readline().strip().split()[1]
    return embeddings,dimension


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    

    """
        question: a string
        embeddings: gensim embeddings
        dim: size of the representation

        result: vector representation for the question
    """
    question_tokens=question.split()
    vec_repre=np.zeros(dim)
    count=0
    for token in question_tokens:
      try:
        vec_repre+=embeddings.get_vector(token)
        count+=1
      except KeyError:
        continue
    if count>0:
      vec_repre=vec_repre/count
    return vec_repre


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
