#!/usr/bin/env python3
"""
NNW Score Calculator - compares earnings transcripts with EPS data
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def clean_cols(df):
    # make column names consistent
    df.columns = df.columns.str.lower().str.strip()
    return df


def find_transcript(ticker, period, transcripts_dir):
    # look for transcript files that match ticker and period
    transcripts_path = Path(transcripts_dir)
    if not transcripts_path.exists():
        return None
    
    ticker_lower = ticker.lower()
    period_lower = period.lower()
    
    candidates = []
    for file_path in transcripts_path.glob("*.txt"):
        filename_lower = file_path.name.lower()
        if ticker_lower in filename_lower and period_lower in filename_lower:
            candidates.append((file_path, file_path.stat().st_size))
    
    if not candidates:
        return None
    
    # pick the biggest file if there are multiple matches
    return str(max(candidates, key=lambda x: x[1])[0])


def read_and_trim_transcript(path, hint, max_chars):
    # read transcript file and clean it up
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # cut off at Q&A section if it exists
        if hint in text:
            text = text.split(hint)[0]
        
        # limit length to avoid memory issues
        if len(text) > max_chars:
            text = text[:max_chars]
        
        return text.strip()
    except Exception as e:
        print(f"Warning: Could not read transcript {path}: {e}")
        return ""


def embed(texts, api_base, model, max_retries=3):
    # get embeddings from LM Studio, with some retry logic
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{api_base}/embeddings",
                headers={"Content-Type": "application/json"},
                json={"model": model, "input": texts},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()["data"]
            return [d["embedding"] for d in data]
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to get embeddings after {max_retries} attempts: {e}")
            print(f"Embedding attempt {attempt + 1} failed: {e}, retrying...")


def cosine(a, b):
    # calculate cosine similarity between two vectors
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def parse_float_safe(value):
    # try to parse a float, handle messy data
    if pd.isna(value):
        return None
    
    str_value = str(value).strip()
    if not str_value:
        return None
    
    # clean up commas and spaces
    str_value = re.sub(r'[,\s]+', '', str_value)
    
    try:
        return float(str_value)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="NNW score calculator")
    parser.add_argument("--eps_csv", default="data/eps_comparison.csv", 
                       help="EPS data file")
    parser.add_argument("--transcripts_dir", default="transcripts", 
                       help="Where the transcript files are")
    parser.add_argument("--out_csv", default="nnw_output.csv", 
                       help="Output file")
    parser.add_argument("--tickers", help="Filter by tickers (comma separated)")
    parser.add_argument("--periods", help="Filter by periods (comma separated)")
    parser.add_argument("--model", default="bge-small-en-v1.5", 
                       help="LM Studio model")
    parser.add_argument("--api_base", default="http://localhost:1234/v1", 
                       help="LM Studio API URL")
    parser.add_argument("--max_chars", type=int, default=8000, 
                       help="Max transcript length")
    parser.add_argument("--qa_split_hint", default="Q&A", 
                       help="Where to cut off transcript")
    
    args = parser.parse_args()
    
    # check if files exist
    if not os.path.exists(args.eps_csv):
        print(f"Error: EPS CSV file not found: {args.eps_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.transcripts_dir):
        print(f"Error: Transcripts directory not found: {args.transcripts_dir}")
        sys.exit(1)
    
    # load the data
    try:
        df = pd.read_csv(args.eps_csv)
        df = clean_cols(df)
        
        # make sure we have the columns we need
        required_cols = ['ticker', 'period', 'eps_actual', 'eps_forecast']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    # apply any filters
    if args.tickers:
        ticker_filter = [t.strip().upper() for t in args.tickers.split(',')]
        df = df[df['ticker'].str.upper().isin(ticker_filter)]
    
    if args.periods:
        period_filter = [p.strip() for p in args.periods.split(',')]
        df = df[df['period'].str.strip().isin(period_filter)]
    
    if df.empty:
        print("No data remaining after filtering")
        sys.exit(1)
    
    print(f"Processing {len(df)} rows...")
    
    # go through each row
    results = []
    skipped = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing NNW scores"):
        ticker = str(row['ticker']).strip()
        period = str(row['period']).strip()
        
        # get the EPS numbers
        eps_actual = parse_float_safe(row['eps_actual'])
        eps_forecast = parse_float_safe(row['eps_forecast'])
        
        if eps_actual is None or eps_forecast is None:
            print(f"Warning: Skipping {ticker} {period} - invalid EPS values")
            skipped += 1
            continue
        
        # find the transcript file
        transcript_path = None
        if 'transcript_path' in df.columns and pd.notna(row['transcript_path']):
            candidate_path = str(row['transcript_path']).strip()
            if os.path.exists(candidate_path):
                transcript_path = candidate_path
        
        if not transcript_path:
            transcript_path = find_transcript(ticker, period, args.transcripts_dir)
        
        if not transcript_path:
            print(f"Warning: No transcript found for {ticker} {period}")
            skipped += 1
            continue
        
        # read the transcript
        transcript_text = read_and_trim_transcript(
            transcript_path, args.qa_split_hint, args.max_chars
        )
        
        if not transcript_text:
            print(f"Warning: Empty transcript for {ticker} {period}")
            skipped += 1
            continue
        
        # create the numbers string for comparison
        numbers_text = f"EPS actual: {eps_actual}; EPS forecast: {eps_forecast}"
        
        try:
            # get embeddings for both texts
            embeddings = embed([transcript_text, numbers_text], args.api_base, args.model)
            transcript_embedding, numbers_embedding = embeddings
            
            # calculate similarity and NNW score
            cosine_sim = cosine(transcript_embedding, numbers_embedding)
            nnw = 1.0 - cosine_sim
            
            # save the result
            results.append({
                'ticker': ticker,
                'period': period,
                'eps_actual': eps_actual,
                'eps_forecast': eps_forecast,
                'transcript_path': transcript_path,
                'cosine': cosine_sim,
                'nnw': nnw
            })
            
        except Exception as e:
            print(f"Warning: Failed to process {ticker} {period}: {e}")
            skipped += 1
            continue
    
    # save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.out_csv, index=False)
        print(f"\nResults written to {args.out_csv}")
        
        # show summary
        processed = len(results)
        print(f"\nProcessed: {processed} | Skipped: {skipped}")
        
        if processed > 0:
            print("\nTop NNW scores:")
            top_results = results_df.nlargest(3, 'nnw')
            for i, (_, row) in enumerate(top_results.iterrows(), 1):
                print(f"{i}) {row['ticker']} {row['period']}  "
                      f"NNW={row['nnw']:.3f}  cosine={row['cosine']:.3f}")
    else:
        print("No results to write")


if __name__ == "__main__":
    main()
