#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the FMCG AI Trainer API.

This script starts the FastAPI application server.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
