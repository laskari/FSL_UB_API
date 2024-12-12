import uvicorn
from fastapi import FastAPI, UploadFile, FastAPI, File, UploadFile, Form, HTTPException

from fastapi.responses import JSONResponse, FileResponse
from typing import Annotated, Optional, List

from fastapi import FastAPI, File, UploadFile, Form
import torch
import json, os
import sys
from extraction_util import run_ub_pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

@app.get("/")
async def root_route():
    return "Application working"

@app.post("/ub_extraction")
async def ml_extraction(data: dict):
    try:

        # Get the image path from the payload
        # image_file_path = data["ClipData"][0]['FilePath']

        # image_file_path = data.get('FilePath')

       # ada_logger.info(f"Got request to API {data}")

        image_file_path = data.get('FilePath')

        #ada_logger.info(f"Image file path in the server {image_file_path}")

        if not image_file_path:
            raise HTTPException(status_code=400, detail="FilePath field is required")

        if not os.path.exists(image_file_path):
            raise HTTPException(status_code=400, detail=f"File not found: {image_file_path}")

        result, error = run_ub_pipeline(image_file_path)
        print(error)

        if error:
            # If there's an error, raise HTTPException with status code 500 (Internal Server Error)
            raise HTTPException(status_code=500, detail=error)
        

        # If there's no error, return the result with file path
        response_data = {"file_path": data.get('FilePath'), "result": result['result']}
        return JSONResponse(content=response_data)

    except Exception as e:
        print(e)

        print(f"Error occured while processing {e} >>> ")
        return JSONResponse(
            status_code=500,
            content=f"Error while processing Extraction {e}"
        )

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run("app:app", host="0.0.0.0", port=port)

