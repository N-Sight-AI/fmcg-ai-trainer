#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the FMCG AI Trainer API.

This script starts the FastAPI application server.
"""

import os
import uvicorn

if __name__ == "__main__":
    # Disable reload in Docker/Production
    reload = os.getenv("ENV", "production") == "development"
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8003")),
        reload=reload,
        log_level="info",
        access_log=True
    )
