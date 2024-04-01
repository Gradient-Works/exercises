import utils
import numpy as np
from typing import List
import pandas as pd
import os

def create_embeddings(text: str) -> List[float]:
    """
    Create embeddings for the given text using a text-embedding-3-small model.

    Args:
        text: The input text for which embeddings need to be created.

    Returns:
        The embedding vector for the input text.
    """
    client = utils.create_client()
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def main():
    """
    Main function to load data, create embeddings, and save them to a file.
    """
    # file_obj = utils.load_data_from_s3(os.getenv("BUCKET_NAME"), 'processed_data.csv')
    # df = pd.read_csv(file_obj)
    df = pd.read_csv('data/processed_data.csv')
    embeddings = [create_embeddings(x) for x in df['split_texts']]
    split_embeddings = np.array(embeddings)
    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')
    np.save('embeddings/split_embeddings.npy', split_embeddings)

if __name__ == "__main__":
    main()