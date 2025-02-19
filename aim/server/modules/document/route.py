# aim/server/modules/document/route.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import os
import logging
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, Response

from ....config import ChatConfig
from ....io.documents import Library
from .dto import DocumentInfo, DocumentListResponse

logger = logging.getLogger(__name__)

class DocumentModule:
    def __init__(self, config: ChatConfig, security: HTTPBearer):
        self.router = APIRouter(prefix="/api/document", tags=["document"])
        self.security = security
        self.config = config
        self.library = Library(documents_dir=config.documents_dir)
        
        self.setup_routes()

    def setup_routes(self):
        @self.router.get("/list")
        async def list_documents(
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """List all available documents"""
            try:
                documents = []
                for name, modified_time, size in self.library.list_documents:
                    documents.append(DocumentInfo(
                        name=name,
                        modified_time=modified_time,
                        size=size
                    ))
                
                return DocumentListResponse(
                    status="success",
                    message=f"Found {len(documents)} documents",
                    documents=documents
                )
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/{document_name}")
        async def download_document(
            document_name: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Download a specific document"""
            try:
                file_path = os.path.join(self.config.documents_dir, document_name)
                if not os.path.exists(file_path):
                    raise HTTPException(status_code=404, detail=f"Document {document_name} not found")

                return FileResponse(
                    path=file_path,
                    filename=document_name,
                    media_type="application/octet-stream"
                )
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/upload")
        async def upload_document(
            file: UploadFile = File(...),
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Upload a new document"""
            try:
                # Validate file extension
                if not any(file.filename.endswith(ext) for ext in self.library.extensions):
                    valid_extensions = ", ".join(self.library.extensions)
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid file extension. Supported extensions: {valid_extensions}"
                    )

                file_path = os.path.join(self.config.documents_dir, file.filename)
                
                # Read file content and write to documents directory
                content = await file.read()
                with open(file_path, "wb") as f:
                    f.write(content)

                return {
                    "status": "success",
                    "message": f"Document {file.filename} uploaded successfully",
                    "filename": file.filename
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/{document_name}/remove")
        async def delete_document(
            document_name: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Delete a document"""
            try:
                file_path = os.path.join(self.config.documents_dir, document_name)
                if not os.path.exists(file_path):
                    raise HTTPException(status_code=404, detail=f"Document {document_name} not found")

                os.remove(file_path)
                return {
                    "status": "success", 
                    "message": f"Document {document_name} deleted"
                }
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/{document_name}/contents")
        async def get_document_contents(
            document_name: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get the contents of a specific document for browser display"""
            try:
                file_path = os.path.join(self.config.documents_dir, document_name)
                if not os.path.exists(file_path):
                    raise HTTPException(status_code=404, detail=f"Document {document_name} not found")
                
                # Check if file is text-based (you may want to expand this list)
                if not any(document_name.lower().endswith(ext) for ext in self.library.extensions):
                    raise HTTPException(
                        status_code=400,
                        detail="File type not supported for text viewing"
                    )

                try:
                    # Try to read file as UTF-8
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # If UTF-8 fails, this might not be a text file
                    raise HTTPException(
                        status_code=400,
                        detail="File appears to be binary or uses an unsupported encoding"
                    )

                return Response(
                    content=content,
                    media_type="text/plain",
                    headers={
                        "Content-Disposition": f"inline; filename={document_name}"
                    }
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))