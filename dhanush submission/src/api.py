from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
import os

from retrieval import find_relevant_docs
from generate import generate_response
import utils
from typing import List

app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/process_query/")
async def process_query(query: Query):
    try:
        file_obj = utils.load_data_from_s3(os.getenv("BUCKET_NAME"), 'processed_data.csv')
        df = pd.read_csv(file_obj)
        extracted_text, extracted_urls, cosine_scores = find_relevant_docs(query.query, df)
        response = generate_response(query.query, extracted_text)
        return {"response": response, "extracted_documents": extracted_text, "urls": extracted_urls, "cosine_scores": cosine_scores}
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
