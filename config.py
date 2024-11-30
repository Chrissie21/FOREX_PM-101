import os
import logging
import pandas as pd

# Configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Define the file name for USDJPY CSV
csv_file_name = "USDJPY.csv"  # Modify this to the correct file name if needed

# Get the current working directory (where the script is located)
current_dir = os.path.dirname(os.path.realpath(__file__))

# Define the local file path where the CSV file should be
csv_file_path = os.path.join(current_dir, csv_file_name)

# Check if the file exists
if os.path.exists(csv_file_path):
    logging.info(f"File found at {csv_file_path}. Loading the CSV file...")
    
    # Load the CSV file using pandas
    try:
        df = pd.read_csv(csv_file_path)
        logging.info("CSV file loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading CSV file: {e}")
else:
    logging.error(f"File not found at {csv_file_path}. Please check the file path.")
    
# You can now proceed with further operations on the 'df' DataFrame (the loaded CSV)
