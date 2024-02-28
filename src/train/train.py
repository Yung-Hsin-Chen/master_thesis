import random
import re
from langdetect import detect, LangDetectException
import wikipedia
from tqdm import tqdm
import os

def is_english(text):
    try:
        # Use langdetect to detect the language
        language = detect(text)
        # Check if the detected language is English
        return language == 'en'
    except LangDetectException:
        # In case the language detection fails or the text is too short
        return False

def wiki_dataset(path, sample_size=100000, sample_articles=10, sample_sentences=10):
    file_path = "./data/wiki/enwiki_title.txt"

    with open(file_path, 'r') as file:
        lines = list(file)
    
    # If the file has fewer lines than the sample size, return all lines
    if len(lines) < sample_size:
        return lines
    
    # Otherwise, randomly sample 'sample_size' lines from the list
    articles = random.sample(lines, sample_size)
    
    # Optionally, strip newline characters from sampled lines
    articles = [line.strip() for line in articles]
    for article in tqdm(articles):
        page_ids = wikipedia.search(article)
        page_ids = random.sample(page_ids, sample_articles) if len(page_ids) >= sample_articles else page_ids
        final_sentences = []
        for id in page_ids:
            try:
                page = wikipedia.page(id)
                content = page.content
                content = content.replace("\n", "")
                content = re.sub(r'=+.*?=+', ' ', content)
                # sentences = re.split(r'\. |\.\n', content)
                sentences = content.split(". ")
                # Sample 2 sentences randomly from the list of sentences
                sampled_sentences = random.sample(sentences, sample_sentences) if len(sentences) >= sample_sentences else sentences
                final_sentences += [line+". ".strip() for line in sampled_sentences if is_english(line)]
            except:
                pass
            with open(path, "a") as file:
                # Iterate over each string in the list
                for string in final_sentences:
                    # Write the string followed by a newline character to the file
                    file.write(f"{string}\n")
    return

if __name__=="__main__":
    # if not os.path.exists("./data/wiki/enwiki_train.txt"):
        # print("ok")
    wiki_dataset("./data/wiki/enwiki_train.txt", sample_size=1000)
    # if not os.path.exists("./data/wiki/enwiki_val.txt"):
    wiki_dataset("./data/wiki/enwiki_val.txt", sample_size=1000)