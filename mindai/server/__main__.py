# mindai/server/__main__.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

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
    logging.getLogger("mindai").setLevel(logging.DEBUG)
    logging.getLogger("mindai").addHandler(logging.StreamHandler())
    
    # Set the format for the mindai logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s")
    for handler in logging.getLogger("mindai").handlers:
        handler.setFormatter(formatter)
    
    # Set other loggers to a higher level (e.g., WARNING)
    logging.getLogger().setLevel(logging.WARNING)
    
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
