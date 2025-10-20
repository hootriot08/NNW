#!/usr/bin/env python3
"""
Fast transcript extraction script
Extracts individual transcripts from metadata CSV files
"""

import pandas as pd
import os
import sys
from pathlib import Path
import time
from tqdm import tqdm

def extract_transcripts_from_csv(csv_file, output_dir, pbar):
    """Extract transcripts from a single CSV file"""
    # Read CSV in chunks for memory efficiency
    chunk_size = 1000
    processed_count = 0
    
    try:
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                # Skip if any required fields are missing
                if pd.isna(row['filename']) or pd.isna(row['text']):
                    continue
                
                # Create filename (remove .xml extension if present)
                filename = str(row['filename']).replace('.xml', '.txt')
                output_path = os.path.join(output_dir, filename)
                
                # Get the full text content - ensure we get the complete transcript
                full_text = str(row['text'])
                
                # Create transcript content with metadata header
                transcript_content = f"""TRANSCRIPT METADATA:
==================
Original Filename: {row['filename']}
Ticker: {row['ticker']}
Start Date: {row['startdate']}
Year: {row['year']}
Flag: {row['flag']}

TRANSCRIPT CONTENT:
==================
{full_text}"""
                
                # Write transcript file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(transcript_content)
                
                processed_count += 1
                pbar.update(1)
    
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return 0
    
    return processed_count

def main():
    # Set up paths
    metadata_dir = "/Users/vazea/Desktop/data project/transcript_metadata"
    output_dir = "/Users/vazea/Desktop/data project/extracted_transcripts"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(metadata_dir) if f.endswith('.csv')]
    csv_files.sort()
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Estimate total rows (faster than counting all)
    print("Estimating total transcripts...")
    # Sample a few files to estimate total
    sample_files = csv_files[:3] if len(csv_files) >= 3 else csv_files
    sample_rows = 0
    for csv_file in sample_files:
        csv_path = os.path.join(metadata_dir, csv_file)
        try:
            df = pd.read_csv(csv_file, nrows=1000)  # Just read first 1000 rows
            sample_rows += len(df)
        except Exception as e:
            print(f"Error sampling {csv_file}: {e}")
    
    # Estimate total based on sample
    estimated_total = (sample_rows / len(sample_files)) * len(csv_files)
    print(f"Estimated total transcripts: {int(estimated_total):,}")
    
    # Create progress bar with estimated total
    pbar = tqdm(total=int(estimated_total), desc="Extracting transcripts", unit="transcripts", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    total_processed = 0
    start_time = time.time()
    
    # Process each CSV file
    for i, csv_file in enumerate(csv_files):
        csv_path = os.path.join(metadata_dir, csv_file)
        print(f"Processing file {i+1}/{len(csv_files)}: {csv_file}")
        processed = extract_transcripts_from_csv(csv_path, output_dir, pbar)
        total_processed += processed
        print(f"  Extracted {processed} transcripts from {csv_file}")
    
    pbar.close()
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n=== EXTRACTION COMPLETE ===")
    print(f"Total transcripts extracted: {total_processed:,}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Average speed: {total_processed/elapsed:.0f} transcripts/second")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
