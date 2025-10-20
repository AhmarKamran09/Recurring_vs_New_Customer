from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from models import (
    RecognizeBatchResponse,
    RecognizeItem,
    RecognizePerImage,
    RecognizeResponse,
)
from services import RecognitionService


router = APIRouter()


# @router.post("/recognize", response_model=RecognizeResponse)
# async def recognize(file: UploadFile = File(...)) -> RecognizeResponse:
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="File is required")

#     # Save the uploaded image to a temporary file that DeepFace can read
#     try:
#         suffix = os.path.splitext(file.filename)[1] or ".jpg"
#         fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="upload_")
#         os.close(fd)
#         data = await file.read()
#         np_arr = np.frombuffer(data, np.uint8)
#         img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#         if img is None:
#             raise ValueError("Invalid image file")
#         cv2.imwrite(tmp_path, img)
#     except Exception as e:  # noqa: BLE001
#         raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

#     service = RecognitionService.instance()
#     try:
#         results_raw = service.recognize_image_path(tmp_path)
#     finally:
#         try:
#             os.remove(tmp_path)
#         except OSError:
#             pass

#     items = [RecognizeItem(**r) for r in results_raw]
#     return RecognizeResponse(num_faces=len(items), results=items)


@router.post("/recognize-batch", response_model=RecognizeBatchResponse)
async def recognize_batch(files: list[UploadFile] = File(...)) -> RecognizeBatchResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    service = RecognitionService.instance()
    outputs: list[RecognizePerImage] = []

    for f in files:
        if not f.filename:
            outputs.append(RecognizePerImage(filename="", num_faces=0, results=[]))
            continue
        
        try:
            # Read file data
            data = await f.read()
            if not data or len(data) == 0:
                outputs.append(RecognizePerImage(filename=f.filename, num_faces=0, results=[]))
                continue
            
            # Convert to numpy array and decode
            np_arr = np.frombuffer(data, np.uint8)
            if len(np_arr) == 0:
                outputs.append(RecognizePerImage(filename=f.filename, num_faces=0, results=[]))
                continue
                
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                outputs.append(RecognizePerImage(filename=f.filename, num_faces=0, results=[]))
                continue
            
            # Save to temporary file
            suffix = os.path.splitext(f.filename)[1] or ".jpg"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="upload_")
            os.close(fd)
            cv2.imwrite(tmp_path, img)
            
            # Process with recognition service
            results_raw = service.recognize_image_path(tmp_path)
            items = [RecognizeItem(**r) for r in results_raw]
            outputs.append(RecognizePerImage(filename=f.filename, num_faces=len(items), results=items))
            
        except Exception as e:
            print(f"Error processing {f.filename}: {e}")
            outputs.append(RecognizePerImage(filename=f.filename, num_faces=0, results=[]))
        finally:
            # Clean up temporary file
            try:
                if 'tmp_path' in locals():
                    os.remove(tmp_path)
            except OSError:
                pass
    print(f"Outputs: {outputs}")
    return RecognizeBatchResponse(items=outputs)



