#!/bin/bash
# Auto-setup and run script

cd "$(dirname "$0")"

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "🔧 First run: Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    echo "✅ Setup complete!"
else
    source venv/bin/activate
fi

# Run the main script with any passed arguments
python3 ts2md.py "$@"
