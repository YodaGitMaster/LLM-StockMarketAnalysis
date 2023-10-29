import sqlite3
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch  # Import the torch library
# Load the finbert tokenizer and model
import transformers
import threading
import concurrent.futures
from database import SentimentDatabase
import json
import time
from typing import Union
from transformers import logging

logging.set_verbosity_warning()


db = SentimentDatabase()


def timer(func):
    """
    A timer decorator to measure the execution time of a function and write it to a file.

    Args:
        func (function): The function to be wrapped.

    Returns:
        function: The wrapped function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken by {func.__name__}: {int(elapsed_time)} seconds")
        
        # Write elapsed time to a file
        with open("elapsed_time.txt", "a") as file:
            file.write(f"{func.__name__}:{elapsed_time:.4f}\n")
        
        return result
    return wrapper

def analyze_emotions(text: str) -> dict:
    """
    Analyze emotions in the given text using the SamLowe/roberta-base-go_emotions model.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary containing emotion labels and scores.
    """
    
    tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions", model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Run the model on the tokenized input
    outputs = model(**inputs)

    # Get the predicted emotion scores
    emotion_scores = outputs.logits.softmax(dim=1).tolist()[0]

    # Define the emotion labels (assuming this model uses the labels "admiration", "amusement", etc.)
    emotion_labels = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
                      "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
                      "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
                      "relief", "remorse", "sadness", "surprise", "neutral"]

    # Create a dictionary mapping emotion labels to scores
    predominant_emotion = emotion_labels[emotion_scores.index(max(emotion_scores))]

    return predominant_emotion
    
    
    # return generated_text


def analyze_title_finbert_tone(title: str) -> str:
    """
    Analyze sentiment of a title using the yiyanghkust/finbert-tone model.

    Args:
        title (str): The title to analyze.

    Returns:
        Union: A Union containing sentiment label and predicted class.
    """
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone", model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    labels = ["negative", "neutral", "positive"]
    sentiment = labels[predicted_class]
    return sentiment, predicted_class


# Function to analyze text using finbert model
def analyze_title_ProsusAI_finbert(title: str) -> Union[str, int]:
    """
    Analyze the sentiment of a title using the ProsusAI/finbert model.
    """
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    labels = ["negative", "neutral", "positive"]
    sentiment = labels[predicted_class]
    return sentiment, predicted_class

def analyze_title_finbert_tone(title: str) -> Union[str, int]:
    """
    Analyze the sentiment of a title using the yiyanghkust/finbert-tone model.
    """
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone", model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    labels = ["positive", "neutral", "negative"]
    sentiment = labels[predicted_class]
    return sentiment, predicted_class

def analyze_title_Farshid_allagree2(title: str) -> Union[str, int]:
    """
    Analyze the sentiment of a title using the Farshid/bert-large-uncased-financial-phrasebank-allagree2 model.
    """
    #https://huggingface.co/Farshid/bert-large-uncased-financial-phrasebank-allagree2?text=Arm+IPO+Could+Open+Door+For+More+Tech+Stock+Offerings
    tokenizer = AutoTokenizer.from_pretrained("Farshid/bert-large-uncased-financial-phrasebank-allagree2", model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained("Farshid/bert-large-uncased-financial-phrasebank-allagree2")
    inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    labels = ["negative", "neutral", "positive"]
    sentiment = labels[predicted_class]
    return sentiment, predicted_class

def analyze_title_distilled_roberta(title: str) -> Union[str, int]:
    """
    Analyze the sentiment of a title using the mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis model.
    """
    #https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    labels = ["negative", "neutral", "positive"]
    sentiment = labels[predicted_class]
    return sentiment, predicted_class

def analyze_title_sigma(title: str) -> Union[str, int]:
    """
    Analyze the sentiment of a title using the Sigma/financial-sentiment-analysis model.
    """
    tokenizer = AutoTokenizer.from_pretrained("Sigma/financial-sentiment-analysis", model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained("Sigma/financial-sentiment-analysis")
    inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    labels = ["negative", "neutral", "positive"]
    sentiment = labels[predicted_class]
    return sentiment, predicted_class

def analyze_title_twitter_roberta(title: str) -> Union[str, int]:
    """
    Analyze the sentiment of a title using the cardiffnlp/twitter-roberta-base-sentiment-latest model.
    """
    #https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest", model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    labels = ["negative", "neutral", "positive"]
    sentiment = labels[predicted_class]
    return sentiment, predicted_class

def analyze_title_deberta_v3(title: str) -> Union[str, int]:
    """
    Analyze the sentiment of a title using the nickmuchi/deberta-v3-base-finetuned-finance-text-classification model.
    """
    #https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    tokenizer = AutoTokenizer.from_pretrained("nickmuchi/deberta-v3-base-finetuned-finance-text-classification", model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained("nickmuchi/deberta-v3-base-finetuned-finance-text-classification")
    inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    labels = ["negative", "neutral", "positive"]
    sentiment = labels[predicted_class]
    return sentiment, predicted_class


def find_most_frequent_number(numbers):
    """
    Determine the most frequent sentiment number in a list.

    Args:
        numbers (list[int]): List of sentiment numbers.

    Returns:
        str: The most frequent sentiment.
    """
    count = [0, 0, 0]  # Initialize counters for 0 (neutral), 1 (positive), and 2 (negative)
    
    for num in numbers:
        count[num] += 1  # Increment the corresponding counter
    
    neg = count[0]
    neu = count[1] 
    pos = count[2]
    
    return 'positive' if pos >= neu > neg else 'negative'

   
def update_sentiment_for_title(title_id, title, semaphore):
    """
    Update sentiment for a given title.

    Args:
        title_id (int): The ID of the title.
        title (str): The actual title text.
        semaphore (threading.Semaphore): Semaphore to limit concurrent threads.
    """
    with semaphore:
        # Perform sentiment analysis and get sentiment score
        sentiment_scores = []
        for analyze_function in [analyze_title_ProsusAI_finbert, analyze_title_finbert_tone,
                                analyze_title_distilled_roberta, analyze_title_sigma,
                                analyze_title_Farshid_allagree2, analyze_title_twitter_roberta,
                                analyze_title_deberta_v3]:
            result = analyze_function(title)
            sentiment_scores.append(result[1])

        # Determine the sentiment score based on your logic (e.g., using find_most_frequent_number)
        score = find_most_frequent_number(sentiment_scores)
        labels = ["negative", "neutral", "positive"]
        sentiment = labels[score]

        if sentiment:
            # Update sentiment data in the database
            db.update_sentiment(title_id, title, {
                "sentiment": sentiment,
                "prosusai_finbert": labels[sentiment_scores[0]],
                "finbert_tone": labels[sentiment_scores[1]],
                "distilled_roberta": labels[sentiment_scores[2]],
                "sigma": labels[sentiment_scores[3]],
                "farshid_allagree2": labels[sentiment_scores[4]],
                "twitter_roberta": labels[sentiment_scores[5]],
                "deberta_v3": labels[sentiment_scores[6]]
            })


@timer
def update_database_with_sentiment(max_threads:int):
    """
    Update the database with sentiment analysis for all titles.

    Args:
        max_threads (int): Maximum number of threads to use.
    """
    # Fetch all rows from the 'stocks_sentiment' table 
    conn = sqlite3.connect('stock_sentiment.db')  # Updated database name
    c = conn.cursor()
    c.execute('SELECT title_id, title FROM sentiment_data')  # Updated table name
    rows = c.fetchall()
    # Create a semaphore to limit the number of concurrent threads
    max_threads = max_threads
    semaphore = threading.Semaphore(max_threads)

    # Create a list to hold thread objects
    threads = []

    for thread_num, row in enumerate(rows, start=1):
        title_id = row[0]
        title = row[1]

        # Create a thread for each title and start it
        thread = threading.Thread(target=update_sentiment_for_title, args=(title_id, title, semaphore))
        threads.append(thread)
        thread.start()

        print(f"Thread {thread_num} started for title: {title}")
        
    
    # Wait for all threads to finish
    for thread_num, thread in enumerate(threads, start=1):
        thread.join()
        print(f"Thread {thread_num} completed")

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
    

if __name__ == "__main__":
    update_database_with_sentiment(max_threads=4)
    db.dump_to_jsonl('dump.jsonl')
