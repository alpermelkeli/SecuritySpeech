import os

# Use environment variable if set, otherwise default to relative 'data' directory
DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", os.path.join(os.path.dirname(__file__), "data"))

if os.path.exists(DATA_FOLDER_PATH):
    folders = [f for f in os.listdir(DATA_FOLDER_PATH)]
    print(f"Found data folders: {folders}")
else:
    print(f"Data folder not found at: {DATA_FOLDER_PATH}")