import os
from datetime import datetime

# Finding the filename for the latest file in the ../odds_data directory
def get_latest_odds_file(directory):
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Filter only JSON files
    json_files = [f for f in files if f.endswith('.json')]

    # If no JSON files found, return None
    if not json_files:
        return None

    # Find the latest file based on the modification time
    latest_file = max(json_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))

    return latest_file

# Find file by name
def find_file_by_name(directory, name_part):
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    # Filer files that contain the name_part in the filename
    matching_files = [f for f in files if name_part in f]
    # Return the matching file or None if no match found 
    if matching_files:
        return matching_files
    else:
        return None 

# Ultimate file identifier: fileFinderModel()
def fileFinderModel(directory, name_part):
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    # Filter files that contain the name_part in the filename
    matching_files = [f for f in files if name_part in f]
    if not matching_files:
        return None # Return None if no matching file is found

    # Extract data from filenames and store as tuple (filename, date)
    files_with_dates = []
    for file in matching_files:
        try:
            # Adjusted split to handle your specific filename format
            # Assume the date is at the beginning of the filename in the format 'YYYY-MM-DD-HH_MM'
            date_str = "_".join(file.split('_')[:2]) # Joining the first two parts to get 'YYYY-MM-DD-HH_MM'
            file_date = datetime.strptime(date_str, '%Y-%m-%d-%H_%M')
        except ValueError:
            # Skip files that don't have a valid date in the name
            continue

    if not files_with_dates:
        return None # No files with valid dates

    # Sort files by date and get the latest one
    latest_file = max(files_with_dates, key=lambda x: x[1])[0]

    # Return the latest file that matches both the name and date
    return latest_file

# Example usage:
directory_path = 'gradData'
name_part = 'rank_predictions'
result = fileFinderModel(directory_path, name_part)
print(f'Latest matching file: {result}')
