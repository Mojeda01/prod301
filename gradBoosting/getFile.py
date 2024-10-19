import os

def get_latest_creation_date_file(directory):
    entries = os.listdir(directory) # Get list of all times in the directory
    # Create full paths and filter out directories
    full_paths = [os.path.join(directory, entry) for entry in entries if os.path.isfile(os.path.join(directory, entry))]
    if not full_paths:
        return None # Return None if the directory is empty or contains no files
    latest_file = max(full_paths, key=os.path.getctime) # Find the file with the latest creation time
    return os.path.basename(latest_file) # Return just the filename
