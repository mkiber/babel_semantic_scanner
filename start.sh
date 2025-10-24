#!/bin/bash
source venv/bin/activate
uvicorn app:app --reload
chmod +x start.sh
