# aim/server/__main__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import logging
from fastapi.middleware.cors import CORSMiddleware
from .serverapi import ServerApi

def create_app():
    server_api = ServerApi()
    app = server_api.app
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )
    
    return app

if __name__ == "__main__":
    import uvicorn
    
    # Configure basic logging to show errors
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Set AI-Mind logger to INFO level
    logging.getLogger("aim").setLevel(logging.INFO)
    logging.getLogger("aim").addHandler(logging.StreamHandler())
    
    # Set the format for the AI-Mind logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s")
    for handler in logging.getLogger("aim").handlers:
        handler.setFormatter(formatter)
    
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
