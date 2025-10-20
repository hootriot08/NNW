import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import sys
import time

# Configuration
TEST_LIMIT = None  # Set to None to process all files
BASE_PATH = "/Users/vazea/Desktop/NNW"
TRANSCRIPT_DIR = os.path.join(BASE_PATH, "extracted_transcripts_4")
DATA_DIR = os.path.join(BASE_PATH, "data")
CONDENSED_DATA_DIR = os.path.join(DATA_DIR, "condensed data")

EPS_ACTUAL_FILE = os.path.join(CONDENSED_DATA_DIR, "EPS_unadjusted_actual_condensed.txt")
EPS_FORECAST_FILE = os.path.join(CONDENSED_DATA_DIR, "EPS_summary_with_implicit_q4.txt")
EXISTING_MATCHES_FILE = os.path.join(DATA_DIR, "matched_transcript_eps_comprehensive_1.csv")
OUTPUT_CSV_FILE = os.path.join(DATA_DIR, "new_transcript_matches_4.csv")
LOG_FILE = os.path.join(DATA_DIR, "reclassification.log")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Helper functions

def clean_ticker(ticker: str) -> str:
    """Remove suffixes from ticker symbols like AAPL. -> AAPL"""
    if not isinstance(ticker, str):
        return ""
    # Remove common ticker suffixes
    return re.split(r'[\.\^/]', ticker, 1)[0]

def parse_transcript_metadata(filepath: str) -> dict | None:
    """Parse ticker and date from transcript metadata"""
    ticker = None
    start_date_str = None
    ticker_re = re.compile(r'^Ticker:\s*(.+)$', re.IGNORECASE)
    date_re = re.compile(r'^Start Date:\s*(.+)$', re.IGNORECASE)

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip() == "TRANSCRIPT CONTENT:":
                    break
                
                if not ticker:
                    ticker_match = ticker_re.search(line)
                    if ticker_match:
                        ticker = ticker_match.group(1).strip()
                
                if not start_date_str:
                    date_match = date_re.search(line)
                    if date_match:
                        start_date_str = date_match.group(1).strip()
                
                if ticker and start_date_str:
                    break
        
        if not ticker or not start_date_str:
            logging.warning(f"Missing Ticker or Start Date in {filepath}")
            return None
        
        # Parse date in format: 7-28-2005
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
    """Check if transcript is about annual or quarterly earnings"""
    quarter_count = 0
    annual_count = 0
    # Compile regex patterns
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
                
                # Count occurrences in the line
                quarter_count += len(quarter_re.findall(line))
                annual_count += len(annual_re.findall(line))
        
        # Return ANNUAL if it appears more often
        return 'ANNUAL' if annual_count > quarter_count else 'QTR'
    
    except Exception as e:
        logging.error(f"Error analyzing content for {filepath}: {e}")
        return 'QTR'  # Default to quarterly

def main():
    """Main processing function"""
    start_time = time.time()
    logging.info("Starting transcript processing")
    
    if TEST_LIMIT:
        logging.warning(f"Test mode: processing only {TEST_LIMIT} files")
    
    # Load data
    logging.info("Loading data...")
    try:
        # Load existing matches to skip
        if os.path.exists(EXISTING_MATCHES_FILE):
            existing_matches_df = pd.read_csv(EXISTING_MATCHES_FILE, usecols=['filepath'])
            existing_matches_set = set(existing_matches_df['filepath'])
            logging.info(f"Loaded {len(existing_matches_set)} existing matches to skip.")
        else:
            logging.warning(f"Existing matches file not found: {EXISTING_MATCHES_FILE}. Will process all files.")
            existing_matches_set = set()

        # Load actual EPS data
        actual_eps_df = pd.read_csv(
            EPS_ACTUAL_FILE, 
            sep='\t', 
            usecols=['TICKER', 'CUSIP', 'OFTIC', 'PDICITY', 'ANNDATS', 'VALUE', 'FIN_PERIOD'],
            low_memory=False
        )
        actual_eps_df['ANNDATS'] = pd.to_datetime(actual_eps_df['ANNDATS'], errors='coerce')
        actual_eps_df.dropna(subset=['ANNDATS'], inplace=True)
        
        # Group by ticker for faster lookups
        actual_eps_grouped_ticker = actual_eps_df.groupby('TICKER')
        actual_eps_grouped_oftic = actual_eps_df.groupby('OFTIC')
        logging.info(f"Loaded {len(actual_eps_df)} actual EPS records and created ticker groups.")

        # Load forecast EPS data
        forecast_eps_df = pd.read_csv(
            EPS_FORECAST_FILE, 
            sep='\t',
            usecols=['TICKER', 'CUSIP', 'CNAME', 'FIN_PERIOD', 'MEDEST'],
            low_memory=False
        )
        forecast_eps_df.set_index(['TICKER', 'FIN_PERIOD'], inplace=True)
        forecast_eps_df.sort_index(inplace=True)
        
        # Create fallback lookup for company names
        cname_lookup_df = forecast_eps_df.reset_index().drop_duplicates(subset=['TICKER'])
        cname_lookup = dict(zip(cname_lookup_df['TICKER'], cname_lookup_df['CNAME']))
        logging.info(f"Loaded {len(forecast_eps_df)} forecast EPS records.")

    except Exception as e:
        logging.critical(f"Failed to load data: {e}", exc_info=True)
        return

    # Find transcripts to process
    logging.info("Finding transcripts to process...")
    try:
        all_files = [
            os.path.join(TRANSCRIPT_DIR, f) 
            for f in os.listdir(TRANSCRIPT_DIR) 
        ]
        # Skip files already processed
        files_to_process = [
            f for f in all_files 
            if f not in existing_matches_set
        ]
        logging.info(f"Found {len(all_files)} total transcripts. {len(files_to_process)} are new.")
    except Exception as e:
        logging.critical(f"Failed to read transcript directory: {e}", exc_info=True)
        return

    if not files_to_process:
        logging.info("No new transcripts to process. Exiting.")
        return

    # Apply test limit if needed
    if TEST_LIMIT:
        files_to_process = files_to_process[:TEST_LIMIT]
        logging.info(f"Applying TEST_LIMIT. Processing {len(files_to_process)} files.")

    # Process transcripts
    logging.info("Processing transcripts...")
    new_matches = []
    error_count = 0
    skipped_no_match = 0
    
    # Progress bar
    for filepath in tqdm(files_to_process, desc="Processing transcripts", unit="file", ncols=100):
        try:
            # Parse metadata
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

            # Find actual EPS match
            date_limit = transcript_date + timedelta(days=5)
            candidate_rows = pd.DataFrame()

            # Try ticker lookup first
            if clean_ticker_val in actual_eps_grouped_ticker.groups:
                ticker_group = actual_eps_grouped_ticker.get_group(clean_ticker_val)
                # Filter by date range
                candidate_rows = ticker_group[
                    (ticker_group['ANNDATS'] >= transcript_date) & 
                    (ticker_group['ANNDATS'] <= date_limit)
                ]
            
            # Try OFTIC as fallback
            if candidate_rows.empty and clean_ticker_val in actual_eps_grouped_oftic.groups:
                oftic_group = actual_eps_grouped_oftic.get_group(clean_ticker_val)
                candidate_rows = oftic_group[
                    (oftic_group['ANNDATS'] >= transcript_date) & 
                    (oftic_group['ANNDATS'] <= date_limit)
                ]

            if candidate_rows.empty:
                skipped_no_match += 1
                continue
            
            # Select best match
            best_match_row = None
            if len(candidate_rows) == 1:
                best_match_row = candidate_rows.iloc[0]
            else:
                # Multiple matches - check for QTR/ANNUAL conflict
                periods = set(candidate_rows['PDICITY'])
                if 'QTR' in periods and 'ANNUAL' in periods:
                    # Analyze content to pick period
                    period_preference = analyze_content_for_period(filepath)
                    preferred_rows = candidate_rows[candidate_rows['PDICITY'] == period_preference]
                    if not preferred_rows.empty:
                        candidate_rows = preferred_rows
                
                # Sort by date and pick closest
                best_match_row = candidate_rows.sort_values(by='ANNDATS').iloc[0]

            actual_eps_val = best_match_row['VALUE']
            fin_period = best_match_row['FIN_PERIOD']
            # Get CUSIP from actuals
            cusip = best_match_row['CUSIP'] 

            # Find forecast EPS match
            forecast_eps_val = np.nan
            company_name = "Unknown"
            
            try:
                # Look up forecast data
                forecast_match = forecast_eps_df.loc[(clean_ticker_val, fin_period)]
                
                # Handle multiple matches
                if isinstance(forecast_match, pd.DataFrame):
                    forecast_match = forecast_match.iloc[0]
                    
                forecast_eps_val = forecast_match['MEDEST']
                company_name = forecast_match['CNAME']
            
            except KeyError:
                # No forecast match - try to get company name
                if company_name == "Unknown" and clean_ticker_val in cname_lookup:
                     company_name = cname_lookup[clean_ticker_val]
            
            # Store result
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
            logging.error(f"Failed to process {filepath}: {e}", exc_info=True)
            error_count += 1
    
    # Save results
    logging.info("Processing complete")
    total_time = time.time() - start_time
    
    if not new_matches:
        logging.info("No new matches found.")
    else:
        try:
            output_df = pd.DataFrame(new_matches)
            # Set column order
            output_columns = ['filepath', 'cusip', 'ticker', 'company_name', 'fin_period', 'eps_latest_forecast', 'actual_eps']
            output_df = output_df[output_columns]
            
            output_df.to_csv(OUTPUT_CSV_FILE, index=False)
            logging.info(f"Successfully saved {len(new_matches)} new matches to {OUTPUT_CSV_FILE}")
        except Exception as e:
            logging.error(f"Failed to save output CSV: {e}", exc_info=True)
    
    # Summary
    logging.info("Final summary")
    logging.info(f"Total time taken: {total_time:.2f} seconds")
    logging.info(f"Total transcripts scanned: {len(files_to_process)}")
    logging.info(f"New matches found: {len(new_matches)}")
    logging.info(f"Skipped (no date match): {skipped_no_match}")
    logging.info(f"Errors (file/parse errors): {error_count}")
    
    if TEST_LIMIT:
        logging.warning("Script was run in test mode - results are from a sample")
        if total_time > 0 and len(files_to_process) > 0:
            time_per_file = total_time / len(files_to_process)
            total_files = len(all_files) - len(existing_matches_set)
            estimated_total_time = (time_per_file * total_files) / 60
            logging.info(f"Time per transcript: {time_per_file*1000:.2f} ms")
            logging.info(f"Estimated runtime for full dataset ({total_files} files): {estimated_total_time:.2f} minutes")

if __name__ == "__main__":
    main()