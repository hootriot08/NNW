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


def parse_transcript_sections(path, qa_hint, max_chars):
    """
    Parse transcript into remarks and Q&A sections.
    
    Returns:
        tuple: (remarks_text, qa_text, has_qa_section)
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Check if Q&A section exists
        qa_section_markers = [
            "Questions and Answers",
            "Q&A",
            "Question and Answer",
            "Questions & Answers"
        ]
        
        has_qa_section = False
        qa_start_idx = -1
        
        for marker in qa_section_markers:
            if marker in text:
                qa_start_idx = text.find(marker)
                has_qa_section = True
                break
        
        if has_qa_section:
            # Split into remarks and Q&A sections
            remarks_text = text[:qa_start_idx].strip()
            qa_text = text[qa_start_idx:].strip()
            
            # Limit length to avoid memory issues
            if len(remarks_text) > max_chars:
                remarks_text = remarks_text[:max_chars]
            if len(qa_text) > max_chars:
                qa_text = qa_text[:max_chars]
        else:
            # No Q&A section found, use entire transcript as remarks
            remarks_text = text
            qa_text = ""
            
            # Limit length to avoid memory issues
            if len(remarks_text) > max_chars:
                remarks_text = remarks_text[:max_chars]
        
        return remarks_text.strip(), qa_text.strip(), has_qa_section
        
    except Exception as e:
        print(f"Warning: Could not read transcript {path}: {e}")
        return "", "", False


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


def compute_enhanced_nnw(remarks_text, qa_text, has_qa_section, eps_actual, eps_forecast, embed_func, api_base, model, prepared_weight=0.7, qa_weight=0.3):
    """
    Enhanced NNW calculation that considers both prepared remarks and Q&A sections.
    
    Args:
        remarks_text: Prepared remarks section text
        qa_text: Q&A section text (empty if no Q&A section)
        has_qa_section: Boolean indicating if Q&A section exists
        eps_actual: Actual EPS value
        eps_forecast: Forecast EPS value
        embed_func: Function to get embeddings
        api_base: API base URL
        model: Model name
        prepared_weight: Weight for prepared remarks (default 0.7)
        qa_weight: Weight for Q&A section (default 0.3)
    
    Returns:
        tuple: (cosine_similarity_prepared, cosine_similarity_qa, nnw_final, has_qa_section)
    """
    # calculate surprise - use dollar delta for small EPS values to avoid crazy percentages
    eps_threshold = 0.10  # Use dollar delta for EPS < $0.10
    
    if abs(eps_forecast) < eps_threshold:
        # Use dollar delta for small EPS values
        dollar_delta = abs(eps_actual - eps_forecast)
        surprise_pct = dollar_delta * 100  # Convert to cents for display
        use_dollar_delta = True
    else:
        # Use percentage for normal EPS values
        surprise_pct = abs((eps_actual - eps_forecast) / eps_forecast) * 100
        use_dollar_delta = False
    
    beat_miss = eps_actual > eps_forecast
    
    # create enriched EPS context with intensity-based language
    if beat_miss:
        if use_dollar_delta:
            # Use dollar delta thresholds for small EPS values
            if surprise_pct >= 5:  # 5+ cents difference
                numbers_text = f"The company delivered a standout earnings performance with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a remarkable {surprise_pct:.1f} cent surprise. This exceptional outperformance reflects strong execution, robust demand, and disciplined cost control."
            elif surprise_pct >= 2:  # 2-4 cents difference
                numbers_text = f"The company outperformed expectations with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a solid {surprise_pct:.1f} cent surprise. This strong result highlights meaningful operational momentum and better-than-anticipated fundamentals."
            elif surprise_pct >= 1:  # 1-2 cents difference
                numbers_text = f"The company beat expectations with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a {surprise_pct:.1f} cent surprise. This moderate upside indicates healthy underlying performance and stable execution."
            elif surprise_pct >= 0.5:  # 0.5-1 cent difference
                numbers_text = f"The company slightly beat expectations with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a modest {surprise_pct:.1f} cent surprise. This mild beat suggests consistent delivery in line with management guidance."
            else:  # < 0.5 cent difference
                numbers_text = f"The company reported EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, remaining essentially in line with expectations. Performance was steady and reflects predictable, disciplined operations."
        else:
            # Use percentage thresholds for normal EPS values
            if surprise_pct >= 50:  # Extraordinary beat (50%+)
                numbers_text = f"The company delivered a standout earnings performance with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a remarkable {surprise_pct:.1f}% surprise. This exceptional outperformance reflects strong execution, robust demand, and disciplined cost control."
            elif surprise_pct >= 20:  # Strong beat (20-49%)
                numbers_text = f"The company outperformed expectations with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a solid {surprise_pct:.1f}% surprise. This strong result highlights meaningful operational momentum and better-than-anticipated fundamentals."
            elif surprise_pct >= 5:  # Moderate beat (5-19%)
                numbers_text = f"The company beat expectations with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a {surprise_pct:.1f}% surprise. This moderate upside indicates healthy underlying performance and stable execution."
            elif surprise_pct >= 2:  # Narrow beat (2-4%)
                numbers_text = f"The company slightly beat expectations with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a modest {surprise_pct:.1f}% surprise. This mild beat suggests consistent delivery in line with management guidance."
            else:  # In line (<2%)
                numbers_text = f"The company reported EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, remaining essentially in line with expectations. Performance was steady and reflects predictable, disciplined operations."
    else:
        if use_dollar_delta:
            # Use dollar delta thresholds for small EPS values
            if surprise_pct >= 5:  # 5+ cents difference
                numbers_text = f"The company reported a substantial earnings shortfall with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a steep {surprise_pct:.1f} cent miss. This significant underperformance reflects material headwinds and operational challenges during the quarter."
            elif surprise_pct >= 2:  # 2-4 cents difference
                numbers_text = f"The company reported a notable earnings miss with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a {surprise_pct:.1f} cent shortfall. This downside indicates pressure on margins or revenue softness relative to expectations."
            elif surprise_pct >= 1:  # 1-2 cents difference
                numbers_text = f"The company missed expectations with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a {surprise_pct:.1f} cent shortfall. This moderate miss suggests weaker-than-expected performance in select areas but not a structural deterioration."
            elif surprise_pct >= 0.5:  # 0.5-1 cent difference
                numbers_text = f"The company slightly missed expectations with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a small {surprise_pct:.1f} cent shortfall. This marginal miss is minor and largely within the range of normal variability."
            else:  # < 0.5 cent difference
                numbers_text = f"The company reported EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, essentially in line with expectations. Performance was consistent and reflects stable operational execution."
        else:
            # Use percentage thresholds for normal EPS values
            if surprise_pct >= 50:  # Severe miss (50%+)
                numbers_text = f"The company reported a substantial earnings shortfall with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a steep {surprise_pct:.1f}% miss. This significant underperformance reflects material headwinds and operational challenges during the quarter."
            elif surprise_pct >= 20:  # Large miss (20-49%)
                numbers_text = f"The company reported a notable earnings miss with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a {surprise_pct:.1f}% shortfall. This downside indicates pressure on margins or revenue softness relative to expectations."
            elif surprise_pct >= 5:  # Moderate miss (5-19%)
                numbers_text = f"The company missed expectations with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a {surprise_pct:.1f}% shortfall. This moderate miss suggests weaker-than-expected performance in select areas but not a structural deterioration."
            elif surprise_pct >= 2:  # Narrow miss (2-4%)
                numbers_text = f"The company slightly missed expectations with EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, representing a small {surprise_pct:.1f}% shortfall. This marginal miss is minor and largely within the range of normal variability."
            else:  # In line (<2%)
                numbers_text = f"The company reported EPS of ${eps_actual:.2f} compared to forecast of ${eps_forecast:.2f}, essentially in line with expectations. Performance was consistent and reflects stable operational execution."

    def extract_meaningful_content(text):
        """Extract meaningful content from transcript text, skipping greetings and goodbyes."""
        words = text.split()
        if len(words) > 200:
            # Skip initial greetings and find the start of actual content
            start_idx = 0
            greeting_patterns = [
                'good morning', 'good afternoon', 'good evening', 'welcome', 
                'thank you', 'hello', 'hi', 'thank you for joining', 'operator',
                'conference call', 'earnings call', 'quarterly call'
            ]
            
            # find where actual content starts
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
            
            # find where content ends
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
            
            # extract financial sentences with key terms
            financial_sentences = []
            sentences = text.split('.')
            
            # financial terms to look for
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
            
            # sentiment words to look for
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
            transcript_extract = text
        
        return transcript_extract

    # Process prepared remarks section
    prepared_extract = extract_meaningful_content(remarks_text)
    
    # Process Q&A section if it exists
    qa_extract = ""
    if has_qa_section and qa_text:
        qa_extract = extract_meaningful_content(qa_text)
    
    # get embeddings and compute similarity
    try:
        # Prepare texts for embedding
        texts_to_embed = [prepared_extract, numbers_text]
        if has_qa_section and qa_extract:
            texts_to_embed.append(qa_extract)
        
        embeddings = embed_func(texts_to_embed, api_base, model)
        prepared_embedding = embeddings[0]
        numbers_embedding = embeddings[1]
        
        # Compute similarity for prepared remarks
        sim_prepared = cosine(prepared_embedding, numbers_embedding)
        
        # Compute similarity for Q&A section (if exists)
        sim_qa = 0.0
        if has_qa_section and qa_extract:
            qa_embedding = embeddings[2]
            sim_qa = cosine(qa_embedding, numbers_embedding)
        
        # Calculate weighted final NNW score
        if has_qa_section and qa_extract:
            nnw_final = 1.0 - (prepared_weight * sim_prepared + qa_weight * sim_qa)
        else:
            # If no Q&A section, use only prepared remarks
            nnw_final = 1.0 - sim_prepared
        
        return sim_prepared, sim_qa, nnw_final, has_qa_section
    except Exception as e:
        raise Exception(f"Failed to compute enhanced NNW: {e}")


def compute_simple_nnw(transcript, eps_actual, eps_forecast, embed_func, api_base, model):
    """
    Basic NNW calculation - just compares transcript to EPS context.
    This is kept for backward compatibility.
    """
    # Parse transcript into sections
    remarks_text, qa_text, has_qa_section = parse_transcript_sections(
        transcript, "Q&A", 8000
    )
    
    # Use enhanced NNW with default weights
    sim_prepared, sim_qa, nnw_final, _ = compute_enhanced_nnw(
        remarks_text, qa_text, has_qa_section, eps_actual, eps_forecast, 
        embed_func, api_base, model, 0.7, 0.3
    )
    
    # Return in the old format for compatibility
    return sim_prepared, nnw_final


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
    parser.add_argument("--model", default="text-embedding-finance-embeddings-investopedia", 
                       help="LM Studio model")
    parser.add_argument("--api_base", default="http://127.0.0.1:1234/v1", 
                       help="LM Studio API URL")
    parser.add_argument("--max_chars", type=int, default=8000, 
                       help="Max transcript length")
    parser.add_argument("--qa_split_hint", default="Q&A", 
                       help="Where to cut off transcript")
    parser.add_argument("--prepared_weight", type=float, default=0.7,
                       help="Weight for prepared remarks section (default 0.7)")
    parser.add_argument("--qa_weight", type=float, default=0.3,
                       help="Weight for Q&A section (default 0.3)")
    parser.add_argument("--use_enhanced", action="store_true",
                       help="Use enhanced NNW calculation with Q&A analysis")
    
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
        
        try:
            if args.use_enhanced:
                # Parse transcript into sections
                remarks_text, qa_text, has_qa_section = parse_transcript_sections(
                    transcript_path, args.qa_split_hint, args.max_chars
                )
                
                if not remarks_text:
                    print(f"Warning: Empty transcript for {ticker} {period}")
                    skipped += 1
                    continue
                
                # compute enhanced NNW score
                sim_prepared, sim_qa, nnw_final, has_qa = compute_enhanced_nnw(
                    remarks_text, qa_text, has_qa_section, eps_actual, eps_forecast, 
                    embed, args.api_base, args.model, args.prepared_weight, args.qa_weight
                )
                
                # save the result
                results.append({
                    'ticker': ticker,
                    'period': period,
                    'eps_actual': eps_actual,
                    'eps_forecast': eps_forecast,
                    'transcript_path': transcript_path,
                    'cosine_prepared': sim_prepared,
                    'cosine_qa': sim_qa,
                    'nnw_final': nnw_final,
                    'has_qa_section': has_qa
                })
            else:
                # Use original method for backward compatibility
                transcript_text = read_and_trim_transcript(
                    transcript_path, args.qa_split_hint, args.max_chars
                )
                
                if not transcript_text:
                    print(f"Warning: Empty transcript for {ticker} {period}")
                    skipped += 1
                    continue
                
                # compute simple NNW score
                cosine_sim, nnw = compute_simple_nnw(
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
            if args.use_enhanced:
                top_results = results_df.nlargest(3, 'nnw_final')
                for i, (_, row) in enumerate(top_results.iterrows(), 1):
                    qa_info = f" (Q&A: {row['cosine_qa']:.3f})" if row['has_qa_section'] else " (No Q&A)"
                    print(f"{i}) {row['ticker']} {row['period']}  "
                          f"NNW={row['nnw_final']:.3f}  prepared={row['cosine_prepared']:.3f}{qa_info}")
            else:
                top_results = results_df.nlargest(3, 'nnw')
                for i, (_, row) in enumerate(top_results.iterrows(), 1):
                    print(f"{i}) {row['ticker']} {row['period']}  "
                          f"NNW={row['nnw']:.3f}  cosine={row['cosine']:.3f}")
    else:
        print("No results to write")


if __name__ == "__main__":
    main()