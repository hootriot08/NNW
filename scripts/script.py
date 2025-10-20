import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import sys
import time

# --- Configuration ---

# CRITICAL: Set to None to run on all files AFTER successful testing
TEST_LIMIT = 100 

# --- File Paths ---
BASE_PATH = "/Users/vazea/Desktop/NNW"
TRANSCRIPT_DIR = os.path.join(BASE_PATH, "extracted_transcripts_1")
DATA_DIR = os.path.join(BASE_PATH, "data")
CONDENSED_DATA_DIR = os.path.join(DATA_DIR, "condensed data")

EPS_ACTUAL_FILE = os.path.join(CONDENSED_DATA_DIR, "EPS_unadjusted_actual_condensed.txt")
EPS_FORECAST_FILE = os.path.join(CONDENSED_DATA_DIR, "EPS_summary_with_implicit_q4.txt")
EXISTING_MATCHES_FILE = os.path.join(DATA_DIR, "matched_transcript_eps_comprehensive_1.csv")
OUTPUT_CSV_FILE = os.path.join(DATA_DIR, "new_transcript_matches.csv")
LOG_FILE = os.path.join(DATA_DIR, "reclassification.log")

# --- Setup Logging ---
# Configures logging to both a file and the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Helper Functions ---

def clean_ticker(ticker: str) -> str:
    """
    Cleans ticker symbols by removing suffixes.
    Examples: 'AAPL.' -> 'AAPL', 'MSFT^' -> 'MSFT'
    """
    if not isinstance(ticker, str):
        return ""
    # Split by the first occurrence of '.', '^', or '/' and take the first part
    return re.split(r'[\.\^/]', ticker, 1)[0]

def parse_transcript_metadata(filepath: str) -> dict | None:
    """
    Efficiently parses Ticker and Start Date from transcript metadata.
    Stops reading the file as soon as metadata is found or content starts.
    """
    ticker = None
    start_date_str = None
    ticker_re = re.compile(r'^Ticker:\s*(.+)$', re.IGNORECASE)
    date_re = re.compile(r'^Start Date:\s*(.+)$', re.IGNORECASE)

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Stop parsing if we hit the content section
                if line.strip() == "TRANSCRIPT CONTENT:":
                    break
                
                # Find Ticker
                if not ticker:
                    ticker_match = ticker_re.search(line)
                    if ticker_match:
                        ticker = ticker_match.group(1).strip()
                
                # Find Start Date
                if not start_date_str:
                    date_match = date_re.search(line)
                    if date_match:
                        start_date_str = date_match.group(1).strip()
                
                # Exit early if we have both
                if ticker and start_date_str:
                    break
        
        if not ticker or not start_date_str:
            logging.warning(f"Missing Ticker or Start Date in {filepath}")
            return None
        
        # Parse the date. Expected format: 7-28-2005
        parsed_date = datetime.strptime(start_date_str, '%m-%d-%Y')
        return {'ticker': ticker, 'start_date': parsed_date}
    
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return None
    except ValueError:
        logging.error(f"Invalid date format '{start_date_str}' in {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error parsing metadata for {filepath}: {e}")
        return None

def analyze_content_for_period(filepath: str) -> str:
    """
    Counts 'annual' vs 'quarter/quarterly' in the transcript content
    to resolve matching conflicts. Defaults to 'QTR'.
    """
    quarter_count = 0
    annual_count = 0
    # Pre-compile regex for speed
    quarter_re = re.compile(r'\b(quarter|quarterly)\b', re.IGNORECASE)
    annual_re = re.compile(r'\b(annual)\b', re.IGNORECASE)
    in_content = False

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not in_content:
                    if line.strip() == "TRANSCRIPT CONTENT:":
                        in_content = True
                    continue
                
                # Find all occurrences in the line
                quarter_count += len(quarter_re.findall(line))
                annual_count += len(annual_re.findall(line))
        
        # Prefer ANNUAL only if it strictly appears more
        return 'ANNUAL' if annual_count > quarter_count else 'QTR'
    
    except Exception as e:
        logging.error(f"Error analyzing content for {filepath}: {e}")
        return 'QTR' # Default to QTR on any analysis error

# --- Main Processing Function ---

def main():
    """Main script logic to load, process, and save matches."""
    start_time = time.time()
    logging.info("--- Starting Transcript Reclassification Script ---")
    
    if TEST_LIMIT:
        logging.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.warning(f"!!! SCRIPT IS IN TEST MODE. PROCESSING ONLY {TEST_LIMIT} FILES. !!!")
        logging.warning("!!! DO NOT RUN ON FULL DATASET WITHOUT APPROVAL. !!!")
        logging.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    # === Phase 1: Load & Pre-process Data (Optimize for Speed) ===
    logging.info("Loading and pre-processing data...")
    try:
        # 1.1: Load Existing Matches (for skipping)
        if os.path.exists(EXISTING_MATCHES_FILE):
            # Only read the 'filepath' column for minimum memory usage
            existing_matches_df = pd.read_csv(EXISTING_MATCHES_FILE, usecols=['filepath'])
            # Convert to a set for O(1) average-case lookup time
            existing_matches_set = set(existing_matches_df['filepath'])
            logging.info(f"Loaded {len(existing_matches_set)} existing matches to skip.")
        else:
            logging.warning(f"Existing matches file not found: {EXISTING_MATCHES_FILE}. Will process all files.")
            existing_matches_set = set()

        # 1.2: Load EPS Actual Data
        actual_eps_df = pd.read_csv(
            EPS_ACTUAL_FILE, 
            sep='\t', 
            usecols=['TICKER', 'CUSIP', 'OFTIC', 'PDICITY', 'ANNDATS', 'VALUE', 'FIN_PERIOD'],
            low_memory=False
        )
        # Convert announcement dates to datetime objects *once*
        actual_eps_df['ANNDATS'] = pd.to_datetime(actual_eps_df['ANNDATS'], errors='coerce')
        actual_eps_df.dropna(subset=['ANNDATS'], inplace=True) # Drop rows where date parsing failed
        
        # Create pre-grouped pandas objects for fast ticker lookups
        actual_eps_grouped_ticker = actual_eps_df.groupby('TICKER')
        actual_eps_grouped_oftic = actual_eps_df.groupby('OFTIC')
        logging.info(f"Loaded {len(actual_eps_df)} actual EPS records and created ticker groups.")

        # 1.3: Load EPS Forecast Data
        forecast_eps_df = pd.read_csv(
            EPS_FORECAST_FILE, 
            sep='\t',
            usecols=['TICKER', 'CUSIP', 'CNAME', 'FIN_PERIOD', 'MEDEST'],
            low_memory=False
        )
        # Create a fast multi-index lookup: (TICKER, FIN_PERIOD)
        forecast_eps_df.set_index(['TICKER', 'FIN_PERIOD'], inplace=True)
        forecast_eps_df.sort_index(inplace=True) # Sorting index speeds up lookups
        
        # Create a secondary lookup for CNAME by TICKER (fallback)
        cname_lookup_df = forecast_eps_df.reset_index().drop_duplicates(subset=['TICKER'])
        cname_lookup = dict(zip(cname_lookup_df['TICKER'], cname_lookup_df['CNAME']))
        logging.info(f"Loaded {len(forecast_eps_df)} forecast EPS records and built index/lookups.")

    except Exception as e:
        logging.critical(f"Failed to load initial data. Aborting. Error: {e}", exc_info=True)
        return

    # === Phase 2: Identify Transcripts to Process ===
    logging.info("Identifying transcripts to process...")
    try:
        all_files = [
            os.path.join(TRANSCRIPT_DIR, f) 
            for f in os.listdir(TRANSCRIPT_DIR) 
            if f.endswith('_T.txt')
        ]
        # Filter out files that are already in the existing matches CSV
        files_to_process = [
            f for f in all_files 
            if f not in existing_matches_set
        ]
        logging.info(f"Found {len(all_files)} total transcripts. {len(files_to_process)} are new.")
    except Exception as e:
        logging.critical(f"Failed to read transcript directory: {TRANSCRIPT_DIR}. Error: {e}", exc_info=True)
        return

    if not files_to_process:
        logging.info("No new transcripts to process. Exiting.")
        return

    # 1.4: Apply Test Limit if enabled
    if TEST_LIMIT:
        files_to_process = files_to_process[:TEST_LIMIT]
        logging.info(f"Applying TEST_LIMIT. Processing {len(files_to_process)} files.")

    # === Phase 3: Process Transcripts ===
    logging.info("Starting transcript processing...")
    new_matches = []
    error_count = 0
    skipped_no_match = 0
    
    # Wrap file list in tqdm for a progress bar
    for filepath in tqdm(files_to_process, desc="Processing transcripts", unit="file", ncols=100):
        try:
            # 3.1: Parse Transcript Metadata
            metadata = parse_transcript_metadata(filepath)
            if not metadata:
                logging.warning(f"Could not parse metadata for {filepath}. Skipping.")
                error_count += 1
                continue
            
            raw_ticker = metadata['ticker']
            transcript_date = metadata['start_date']
            clean_ticker_val = clean_ticker(raw_ticker)
            
            if not clean_ticker_val:
                logging.warning(f"Empty cleaned ticker for {filepath} (Raw: {raw_ticker}). Skipping.")
                error_count += 1
                continue

            # 3.2: Find Actual EPS Match
            date_limit = transcript_date + timedelta(days=5)
            candidate_rows = pd.DataFrame() # Start with an empty DataFrame

            # Try TICKER first (fast lookup)
            if clean_ticker_val in actual_eps_grouped_ticker.groups:
                ticker_group = actual_eps_grouped_ticker.get_group(clean_ticker_val)
                # Filter group by date range (vectorized)
                candidate_rows = ticker_group[
                    (ticker_group['ANNDATS'] >= transcript_date) & 
                    (ticker_group['ANNDATS'] <= date_limit)
                ]
            
            # Try OFTIC as fallback if no match
            if candidate_rows.empty and clean_ticker_val in actual_eps_grouped_oftic.groups:
                oftic_group = actual_eps_grouped_oftic.get_group(clean_ticker_val)
                candidate_rows = oftic_group[
                    (oftic_group['ANNDATS'] >= transcript_date) & 
                    (oftic_group['ANNDATS'] <= date_limit)
                ]

            if candidate_rows.empty:
                # Log this only if verbose logging is needed, otherwise it's too noisy
                # logging.info(f"No actual EPS match found for {filepath} (Ticker: {clean_ticker_val}, Date: {transcript_date})")
                skipped_no_match += 1
                continue
            
            # 3.3: Select Best Match & Resolve Conflicts
            best_match_row = None
            if len(candidate_rows) == 1:
                best_match_row = candidate_rows.iloc[0]
            else:
                # More than one match. Check for QTR/ANNUAL conflict.
                periods = set(candidate_rows['PDICITY'])
                if 'QTR' in periods and 'ANNUAL' in periods:
                    # Run content analysis to pick preferred period
                    period_preference = analyze_content_for_period(filepath)
                    preferred_rows = candidate_rows[candidate_rows['PDICITY'] == period_preference]
                    if not preferred_rows.empty:
                        candidate_rows = preferred_rows # Use preferred rows for next step
                
                # Sort by announcement date (closest to transcript date) and pick first
                best_match_row = candidate_rows.sort_values(by='ANNDATS').iloc[0]

            actual_eps_val = best_match_row['VALUE']
            fin_period = best_match_row['FIN_PERIOD']
            # Get CUSIP from actuals, but CNAME will come from forecast
            cusip = best_match_row['CUSIP'] 

            # 3.4: Find Forecast EPS Match
            forecast_eps_val = np.nan # Use numpy's NaN for missing float
            company_name = "Unknown"
            
            try:
                # Use fast multi-index lookup
                forecast_match = forecast_eps_df.loc[(clean_ticker_val, fin_period)]
                
                # Handle rare case where index is not unique
                if isinstance(forecast_match, pd.DataFrame):
                    forecast_match = forecast_match.iloc[0]
                    
                forecast_eps_val = forecast_match['MEDEST']
                company_name = forecast_match['CNAME']
            
            except KeyError:
                # No forecast match for this specific (ticker, period)
                # Try to get CNAME anyway using the secondary ticker-only lookup
                if company_name == "Unknown" and clean_ticker_val in cname_lookup:
                     company_name = cname_lookup[clean_ticker_val]
            
            # 3.5: Store Result
            new_matches.append({
                'filepath': filepath,
                'cusip': cusip,
                'ticker': clean_ticker_val,
                'company_name': company_name,
                'fin_period': fin_period,
                'eps_latest_forecast': forecast_eps_val,
                'actual_eps': actual_eps_val
            })

        except Exception as e:
            # Catch-all for any unexpected error during loop
            logging.error(f"Failed to process {filepath}: {e}", exc_info=True)
            error_count += 1
    
    # === Phase 4: Save Output ===
    logging.info("--- Processing Complete ---")
    total_time = time.time() - start_time
    
    if not new_matches:
        logging.info("No new matches found.")
    else:
        try:
            output_df = pd.DataFrame(new_matches)
            # Ensure columns are in the correct order
            output_columns = ['filepath', 'cusip', 'ticker', 'company_name', 'fin_period', 'eps_latest_forecast', 'actual_eps']
            output_df = output_df[output_columns]
            
            output_df.to_csv(OUTPUT_CSV_FILE, index=False)
            logging.info(f"Successfully saved {len(new_matches)} new matches to {OUTPUT_CSV_FILE}")
        except Exception as e:
            logging.error(f"Failed to save output CSV: {e}", exc_info=True)
    
    # === Final Summary ===
    logging.info("--- FINAL SUMMARY ---")
    logging.info(f"Total time taken: {total_time:.2f} seconds")
    logging.info(f"Total transcripts scanned: {len(files_to_process)}")
    logging.info(f"New matches found: {len(new_matches)}")
    logging.info(f"Skipped (no date match): {skipped_no_match}")
    logging.info(f"Errors (file/parse errors): {error_count}")
    
    if TEST_LIMIT:
        logging.warning("!!! SCRIPT WAS RUN IN TEST MODE. RESULTS ARE FROM A SAMPLE. !!!")
        if total_time > 0 and len(files_to_process) > 0:
            time_per_file = total_time / len(files_to_process)
            total_files = len(all_files) - len(existing_matches_set)
            estimated_total_time = (time_per_file * total_files) / 60
            logging.info(f"Est. time per transcript: {time_per_file*1000:.2f} ms")
            logging.info(f"ESTIMATED RUNTIME FOR FULL DATASET ({total_files} files): {estimated_total_time:.2f} minutes")

if __name__ == "__main__":
    main()