#!/bin/bash
# Production server startup script

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run Gunicorn with the configuration file
gunicorn --config gunicorn_config.py wsgi:app