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


def compute_balanced_nnw(transcript, eps_actual, eps_forecast, embed_func, api_base, model):
    """
    Enhanced NNW calculation with dynamic EPS enrichment based on surprise level.
    
    Args:
        transcript: Full transcript text
        eps_actual: Actual EPS value
        eps_forecast: Forecast EPS value
        embed_func: Function to get embeddings
        api_base: API base URL
        model: Model name
    
    Returns:
        tuple: (cosine_similarity, nnw_score)
    """
    # Step 1: Calculate surprise percentage
    surprise_pct = abs((eps_actual - eps_forecast) / eps_forecast) * 100
    beat_miss = eps_actual > eps_forecast
    
    # Step 2: Create dynamically enriched EPS context based on surprise level
    if beat_miss:
        if surprise_pct >= 100:  # Massive beat (100%+)
            numbers_text = f"The company delivered an extraordinary earnings beat with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a staggering {surprise_pct:.1f}% surprise. This exceptional performance demonstrates outstanding execution, remarkable operational excellence, and exceptional market outperformance that exceeded all expectations."
        elif surprise_pct >= 50:  # Large beat (50-99%)
            numbers_text = f"The company achieved a remarkable earnings beat with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing an impressive {surprise_pct:.1f}% surprise. This strong performance reflects excellent execution, robust operational results, and significant market outperformance."
        elif surprise_pct >= 20:  # Solid beat (20-49%)
            numbers_text = f"The company delivered a solid earnings beat with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a {surprise_pct:.1f}% surprise. This performance demonstrates good execution and positive operational results."
        else:  # Modest beat (<20%)
            numbers_text = f"The company slightly beat expectations with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a {surprise_pct:.1f}% surprise. This modest outperformance shows steady execution."
    else:
        if surprise_pct >= 100:  # Massive miss (100%+)
            numbers_text = f"The company experienced a devastating earnings miss with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a catastrophic {surprise_pct:.1f}% shortfall. This severe underperformance reflects significant operational challenges, major execution issues, and concerning market headwinds."
        elif surprise_pct >= 50:  # Large miss (50-99%)
            numbers_text = f"The company reported a significant earnings miss with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a substantial {surprise_pct:.1f}% shortfall. This disappointing performance reflects operational challenges and execution difficulties."
        elif surprise_pct >= 20:  # Solid miss (20-49%)
            numbers_text = f"The company missed expectations with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a {surprise_pct:.1f}% shortfall. This underperformance indicates some operational challenges."
        else:  # Modest miss (<20%)
            numbers_text = f"The company slightly missed expectations with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a {surprise_pct:.1f}% shortfall. This modest underperformance shows some execution challenges."
    
    # Step 3: Extract meaningful content sections (skip greetings and goodbyes)
    words = transcript.split()
    if len(words) > 200:
        # Skip initial greetings and find the start of actual content
        # Look for common greeting patterns and skip them
        start_idx = 0
        greeting_patterns = [
            'good morning', 'good afternoon', 'good evening', 'welcome', 
            'thank you', 'hello', 'hi', 'thank you for joining', 'operator',
            'conference call', 'earnings call', 'quarterly call'
        ]
        
        # Find the first meaningful content after greetings
        for i, word in enumerate(words[:50]):  # Check first 50 words for greetings
            if any(pattern in ' '.join(words[i:i+3]).lower() for pattern in greeting_patterns):
                start_idx = i + 3  # Skip the greeting phrase
                break
        
        # Skip closing pleasantries and find the end of actual content
        end_idx = len(words)
        closing_patterns = [
            'thank you', 'goodbye', 'good bye', 'have a good', 'that concludes',
            'end of call', 'call is concluded', 'operator', 'questions'
        ]
        
        # Find the last meaningful content before closing
        for i in range(len(words) - 50, len(words)):
            if any(pattern in ' '.join(words[i:i+3]).lower() for pattern in closing_patterns):
                end_idx = i
                break
        
        # Extract first 100 words after greetings and last 100 words before closing
        meaningful_words = words[start_idx:end_idx]
        if len(meaningful_words) > 200:
            transcript_extract = ' '.join(meaningful_words[:100] + meaningful_words[-100:])
        else:
            transcript_extract = ' '.join(meaningful_words)
        
        # Step 4: Extract additional financial sentences with key terminology
        financial_sentences = []
        sentences = transcript.split('.')
        
        # Financial terminology patterns
        financial_patterns = [
            # EPS and earnings terms
            'eps', 'earnings per share', 'earnings', 'profit', 'income', 'revenue',
            'beat', 'miss', 'exceed', 'outperform', 'underperform', 'surprise',
            'forecast', 'guidance', 'expectations', 'target', 'projection',
            
            # Performance indicators
            'growth', 'increase', 'decrease', 'rise', 'fall', 'up', 'down',
            'strong', 'weak', 'robust', 'solid', 'challenging', 'difficult',
            'outstanding', 'exceptional', 'disappointing', 'concerning',
            
            # Financial metrics
            'margin', 'profitability', 'operational', 'execution', 'performance',
            'results', 'quarterly', 'annual', 'year-over-year', 'sequential',
            
            # Numbers and percentages
            'percent', '%', 'basis points', 'million', 'billion', 'thousand',
            'dollar', '$', 'cents', 'share', 'shares'
        ]
        
        # Sentiment-based patterns for more targeted extraction
        positive_words = ['strong', 'exceed', 'outperform', 'growth', 'record', 'pleased', 
                         'excellent', 'robust', 'solid', 'positive', 'improve', 'increase', 
                         'rise', 'gain', 'success', 'momentum', 'expansion', 'outstanding', 
                         'exceptional', 'impressive', 'deliver', 'beat', 'surpass', 'outpace']
        
        negative_words = ['challenge', 'difficult', 'disappoint', 'pressure', 'headwind', 
                         'decline', 'weak', 'struggle', 'concern', 'risk', 'uncertain', 
                         'volatile', 'decrease', 'fall', 'drop', 'miss', 'underperform', 
                         'disappointing', 'concerning', 'struggle', 'headwinds', 'pressure']
        
        # Extract sentences containing financial terminology and sentiment
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if len(sentence_lower) > 20:  # Skip very short sentences
                # Check if sentence contains financial terminology OR sentiment words
                has_financial = any(pattern in sentence_lower for pattern in financial_patterns)
                has_positive = any(word in sentence_lower for word in positive_words)
                has_negative = any(word in sentence_lower for word in negative_words)
                
                if has_financial or has_positive or has_negative:
                    financial_sentences.append(sentence.strip())
        
        # Add top 5 most relevant financial sentences to the extract
        if financial_sentences:
            # Limit to top 5 sentences to avoid making extract too long
            top_financial_sentences = financial_sentences[:5]
            financial_text = ' '.join(top_financial_sentences)
            
            # Combine with the opening/closing extract
            if len(transcript_extract.split()) + len(financial_text.split()) < 300:
                transcript_extract = transcript_extract + ' ' + financial_text
            else:
                # If too long, prioritize financial sentences
                transcript_extract = financial_text
    else:
        # For short transcripts, use the full text
        transcript_extract = transcript
    
    # Step 5: Get embeddings and compute similarity
    try:
        embeddings = embed_func([transcript_extract, numbers_text], api_base, model)
        transcript_embedding, numbers_embedding = embeddings
        
        cosine_sim = cosine(transcript_embedding, numbers_embedding)
        
        # Step 6: Apply surprise-level weighting
        # Higher surprise = more weight to the comparison
        # This increases differentiation for larger surprises
        surprise_weight = min(2.0, max(0.5, surprise_pct / 50))  # Weight between 0.5x and 2.0x
        
        # Apply the weight to the similarity score
        weighted_similarity = cosine_sim * surprise_weight
        
        # Ensure the weighted similarity doesn't exceed 1.0
        weighted_similarity = min(1.0, weighted_similarity)
        
        nnw_score = 1.0 - weighted_similarity
        
        return cosine_sim, nnw_score
    except Exception as e:
        raise Exception(f"Failed to compute balanced NNW: {e}")


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
        
        # make sure we have the columns we need - handle different column name formats
        required_cols = ['eps_name', 'quarter', 'actual_eps', 'forecast_eps']
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
        df = df[df['eps_name'].str.upper().isin(ticker_filter)]
    
    if args.periods:
        period_filter = [p.strip() for p in args.periods.split(',')]
        df = df[df['quarter'].astype(str).str.strip().isin(period_filter)]
    
    if df.empty:
        print("No data remaining after filtering")
        sys.exit(1)
    
    print(f"Processing {len(df)} rows...")
    
    # go through each row
    results = []
    skipped = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing NNW scores"):
        ticker = str(row['eps_name']).strip()
        period = str(row['quarter']).strip()
        
        # get the EPS numbers
        eps_actual = parse_float_safe(row['actual_eps'])
        eps_forecast = parse_float_safe(row['forecast_eps'])
        
        if eps_actual is None or eps_forecast is None:
            print(f"Warning: Skipping {ticker} {period} - invalid EPS values")
            skipped += 1
            continue
        
        # find the transcript file
        transcript_path = None
        if 'filepath' in df.columns and pd.notna(row['filepath']):
            candidate_path = str(row['filepath']).strip()
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
        
        try:
            # use the new balanced NNW calculation with dynamic enrichment
            cosine_sim, nnw = compute_balanced_nnw(
                transcript_text, eps_actual, eps_forecast, 
                embed, args.api_base, args.model
            )
            
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
