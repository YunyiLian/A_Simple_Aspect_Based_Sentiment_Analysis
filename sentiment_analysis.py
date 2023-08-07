import pandas as pd
import numpy as np
import spacy
from pyabsa.tasks.AspectPolarityClassification import SentimentClassifier
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings("ignore")

nlp = spacy.load('model-best')

clf = SentimentClassifier("multilingual")

def absa(reviews):
    """
    Takes a list of reviews as input and split into sentences;
    Tag the dish aspects with pretrained model;
    Generate sentiment and confidence with predefined model in pyabsa.

    Parameters:
        reviews : list
            A list containing reviews for restaurants;
            Each element of the list contains multiple review sentences.

    Returns:
        new_data : DataFrame
            A DataFrame containing review_id, dish, sentiment and confidence
            for each dish aspect.
    """

    review_id_list = []
    dish_list = []
    sentiment_list = []
    confidence_list = []

    for row_num, review in enumerate(reviews):
        sentences = sent_tokenize(review)

        for sentence in sentences:
            trained = nlp(sentence)
            tagged_review = sentence
            for ent in reversed(trained.ents):
                if ent.label_ == 'DISH':
                    tagged = f'[B-ASP]{ent.text}[E-ASP]'
                    start = ent.start_char
                    end = ent.end_char
                    tagged_review = tagged_review[:start] + tagged + tagged_review[end:]

                result = clf.predict(tagged_review,
                                     save_result=False,
                                     print_result=False,
                                     ignore_error=True,
                                     pred_sentiment=True)


                if result['aspect'] != ['Global Sentiment']:
                    review_id_list.extend([row_num for i in range(len(result['aspect']))])
                    dish_list.extend(result['aspect'])
                    sentiment_list.extend(result['sentiment'])
                    confidence_list.extend(result['confidence'])

    new_data = pd.DataFrame(list(zip(review_id_list, dish_list, sentiment_list, confidence_list)),
                            columns=['review_id', 'dish', 'sentiment', 'confidence'])

    return new_data
