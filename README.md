# A_Simple_Aspect_Based_Sentiment_Analysis
A school project of applying aspect-based sentiment analysis on restaurant reviews.

## Project Description
1. Trained a custom `Name Entity Recognition` (NER) model using `prodigy`.
2. Set up a virtual environment in the local machine and access the `prodigy` interface locally.
3. In the `Prodigy` interface, created a new name entity `DISH` and manually labelled it in restaurant reviews.
4. Exported model as `model-best`.
5. In a `Python` environment, load the saved model `model-best` using `spaCy`; import `SentimentClassifier` from `pyabsa` (Python Aspect-based Sentiment Analysis) library.
6. Built a function `absa` in which reviews will be passed and iterated for `sentence tokenization` using `nltk`; Pass each sentence to `model-bset` and verify if the sentence contains any name entities labelled in `DISH`; If it does, embed the corresponding text; Predict the sentiment of the embedded sentences using `SentimentClassifer` and store the `review_id`, `dish`, `sentiment` and `confidence` in 4 separated dictionaries; Create a `pandas` `DataFrame` containing the content in the 4 dictionaries and return the `DataFrame`.
