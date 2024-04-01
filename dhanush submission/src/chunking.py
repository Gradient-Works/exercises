from bs4 import BeautifulSoup
import pandas as pd
import utils
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)


def extract_text(html_content):
    """
    Extracts text from HTML content.

    Args:
        html_content (str): The HTML content to extract text from.

    Returns:
        str: The extracted text.
    """
    try:
        soup = BeautifulSoup(html_content, "lxml")
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        logging.error(f"Failed to extract text due to: {e}")
        return ""


def split_text_recursively(text, max_length=1500, overlap_ratio=0.3):
    """
    Splits a given text into chunks recursively based on the maximum length and overlap ratio.

    Args:
        text (str): The input text to be split.
        max_length (int, optional): The maximum length of each chunk. Defaults to 1500 words.
        overlap_ratio (float, optional): The overlap ratio between consecutive chunks. Defaults to 0.3.

    Returns:
        list: A list of text chunks where the no of words in a sentence is <1500.

    """
    words = text.split()
    if len(words) <= max_length:
        return [' '.join(words)]
    else:
        overlap_length = int(max_length * overlap_ratio)
        split_position = max_length
        first_chunk = ' '.join(words[:split_position])
        remaining_words = words[split_position - overlap_length:]
        remaining_chunks = split_text_recursively(' '.join(remaining_words), max_length, overlap_ratio)
        return [first_chunk] + remaining_chunks

    
def split_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split the 'extracted_text' column into chunks and explode the resulting DataFrame.

    Args:
        df: The input DataFrame.

    Returns:
        The modified DataFrame with the 'split_texts' column added and exploded.
    """
    df['split_texts'] = df['extracted_text'].apply(lambda x: split_text_recursively(x, 1500, 0.3))
    df = df.explode('split_texts')
    df.reset_index(inplace = True, drop=True)
    return df


def process_data():
    """
    Process the data by reading a CSV file, extracting text, filtering, splitting, and returning the processed DataFrame.

    Returns:
        pandas.DataFrame: The processed DataFrame containing extracted and split text.
    """
    # file_obj = utils.load_data_from_s3(os.getenv("BUCKET_NAME"), 'content.csv')
    # df = pd.read_csv(file_obj)

    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    df = pd.read_csv('data/content.csv')
    extracted_text = [extract_text(x) for x in df['chunk']]
    df['extracted_text'] = extracted_text
    df['length'] = [len(x) for x in df['extracted_text']]
    df = df[df['length'] > 3]
    df.reset_index(inplace=True, drop=True)
    df.loc[df['chunk_type'] == 'main', 'extracted_text'] = "Company name: " + \
        df['company_name'] + " " + df['extracted_text']        
    df = split_text(df)
    df.to_csv('data/processed_data.csv', index=False)



if __name__ == "__main__":
    process_data()
