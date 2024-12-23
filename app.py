import uvicorn
from fastapi import FastAPI, UploadFile, FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import Annotated, Optional, List
from fastapi import FastAPI, File, UploadFile, Form
import torch
import json, os
import sys
from src.extraction_util import run_ub_pipeline
from config import *
from src.logger import log_message, setup_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger, formatter = setup_logger(LOGFILE_DIR)
app = FastAPI()

@app.get("/")
async def root_route():
    return "Application working"

@app.post("/ub_extraction")
async def ml_extraction(data: dict):
    try:
        XELP_process_request = 'XELP_process_request'
        formatter.start_timing(XELP_process_request)
        log_message(logger, "Started UB ml_extraction", level="INFO")
        
        image_file_path = data.get('FilePath')

        if not image_file_path:
            log_message(logger, "FilePath field is required", level="ERROR")
            raise HTTPException(status_code=400, detail="FilePath field is required")

        if not os.path.exists(image_file_path):
            log_message(logger, f"File not found: {image_file_path}", level="ERROR")
            raise HTTPException(status_code=400, detail=f"File not found: {image_file_path}")

        log_message(logger, f"File found: {image_file_path}. Running pipeline...", level="INFO")
        result, error = run_ub_pipeline(image_file_path, logger, formatter)
        
        if error:
            # If there's an error, raise HTTPException with status code 500 (Internal Server Error)
            log_message(logger, f"Error in pipeline: {error}", level="ERROR")
            raise HTTPException(status_code=500, detail=error)

        # If there's no error, return the result with file path
        response_data = {"version": VERSION, "file_path": data.get('FilePath'), "result": result['result']}

        overall_elapsed_time = formatter.stop_timing(XELP_process_request)
        log_message(logger, f"The pipeline process completed with Data extraction and ROI prediction", level="DEBUG", elapsed_time=overall_elapsed_time)
        return JSONResponse(content=response_data)

    except Exception as e:
        log_message(logger, f"Error occurred: {e}", level="ERROR")
        return JSONResponse(
            status_code=500,
            content=f"Error while processing Extraction {e}"
        )

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run("app:app", host="0.0.0.0", port=port)

