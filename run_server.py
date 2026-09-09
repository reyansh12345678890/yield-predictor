import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)