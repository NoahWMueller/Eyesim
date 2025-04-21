import os
import shutil
from datetime import datetime

# Define the source file and backup folder
source_file = "Gymnasium_PPO.py"
backup_folder = "backups"

# Create the backups folder if it doesn't exist
os.makedirs(backup_folder, exist_ok=True)

# Get the current date in YYYY-MM-DD format
current_date = datetime.now().strftime("%Y-%m-%d")

# Construct the backup file name
backup_file_name = f"Gymnasium_PPO_backup_{current_date}.py"
backup_file_path = os.path.join(backup_folder, backup_file_name)

# Copy the source file to the backup folder with the new name
try:
    shutil.copy(source_file, backup_file_path)
    print(f"Backup created: {backup_file_path}")
except FileNotFoundError:
    print(f"Source file '{source_file}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")