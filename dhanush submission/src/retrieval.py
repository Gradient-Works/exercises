from sklearn.metrics.pairwise import cosine_similarity
import cohere
from embedding import create_embeddings
import numpy as np
import pandas as pd
import os
from typing import List, Tuple
import logging
from s3fs.core import S3FileSystem

logging.basicConfig(level=logging.INFO)


def find_relevant_vectors_cosine(query: str, document_embeddings: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the top k relevant vectors based on cosine similarity.

    Args:
        query (str): The query string.
        document_embeddings (np.ndarray): Array of document embeddings.
        top_k (int): Number of top relevant vectors to return. Default is 5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Indices of top k relevant vectors
        and their corresponding cosine similarity scores.
    """
    try:
        query_embedding = np.array(create_embeddings(query)).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, document_embeddings)
        sorted_indices = np.argsort(similarities[0])[::-1]
        top_scores = np.sort(similarities.flatten())[::-1][:top_k]
        return sorted_indices[:top_k], top_scores
    except Exception as e:
        logging.error(f"Failed to find relevant vectors due to: {e}")

def rerank_documents(query: str, extracted_docs: List[str]) -> List[int]:
    """
    Rerank extracted documents based on a given query using Cohere API.

    Args:
        query (str): The query for reranking the documents.
        extracted_docs (List[str]): List of extracted documents to rerank.

    Returns:
        List[int]: List of reranked document indices based on the query.
    """
    try:
        co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        results = co.rerank(query=query, documents=extracted_docs, top_n=3)
        reranked_indices = [result.index for result in results.results]
        return reranked_indices
    except Exception as e:
        logging.error(f"Failed to rerank documents due to: {e}")

def extract_documents(query: str, relevant_indices: np.ndarray, df: pd.DataFrame, similarity_scores: np.ndarray) -> Tuple[str, List[str], List[float]]:
    """
    Extract documents from a DataFrame and rerank them using a query.

    Args:
        query (str): The query used for reranking the documents.
        relevant_indices (np.ndarray): Indices of relevant documents.
        df (pd.DataFrame): The DataFrame containing the documents.
        similarity_scores (np.ndarray): Cosine similarity scores.

    Returns:
        Tuple[str, List[str], List[float]]: Extracted documents,
        URLs of extracted documents, and cosine similarity scores.
    """
    try:
        extracted_docs = [df.iloc[i]["split_texts"] for i in relevant_indices]
        extracted_urls = np.array(
            [[df.iloc[i]["url"] for i in relevant_indices]]).flatten()
        print(extracted_docs, extracted_urls)
        reranked_indices = rerank_documents(query, extracted_docs)
        reranked_scores = [similarity_scores[x] for x in reranked_indices]
        reranked_docs = [extracted_docs[x] for x in reranked_indices]
        reranked_urls = [extracted_urls[x] for x in reranked_indices]
        extracted_text = " ".join(reranked_docs)
        return extracted_text, reranked_urls, reranked_scores
    except Exception as e:
        logging.error(f"Failed to extract documents due to: {e}")

def find_relevant_docs(query: str, df: pd.DataFrame) -> Tuple[str, List[str], List[float]]:
    """
    Find relevant documents based on a given query and a DataFrame.

    Args:
        query (str): The query string.
        df (pd.DataFrame): The DataFrame containing the documents.

    Returns:
        Tuple[str, List[str], List[float]]: Extracted documents,
        URLs of extracted documents, and cosine similarity scores.
    """
    try:
        # s3 = S3FileSystem(key=os.getenv("AWS_ACCESS_KEY_ID"), secret=os.getenv("AWS_SECRET"))
        # split_embeddings = np.load(s3.open('{}/{}'.format(os.getenv("BUCKET_NAME"), 'split_embeddings.npy'), allow_pickle=True))

        split_embeddings = np.load('../embeddings/split_embeddings.npy')
        embeddings_reshaped = np.vstack(split_embeddings)
        relevant_indices, similarity_scores = find_relevant_vectors_cosine(
            query, embeddings_reshaped
        )
        extracted_text, extracted_urls, cosine_scores = extract_documents(
            query, relevant_indices, df, similarity_scores
        )
        return extracted_text, extracted_urls, cosine_scores
    except Exception as e:
        logging.error(f"Failed to find relevant documents due to: {e}")

