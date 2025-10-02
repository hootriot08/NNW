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
    """Clean up column names - lowercase and strip whitespace"""
    df.columns = df.columns.str.lower().str.strip()
    return df


def find_transcript(ticker, period, transcripts_dir):
    """Find transcript file for ticker/period - picks the biggest one if multiple matches"""
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
    
    # Just grab the biggest file if we have multiple
    return str(max(candidates, key=lambda x: x[1])[0])


def read_and_trim_transcript(path, hint, max_chars):
    """Read transcript and clean it up"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Cut off Q&A if it exists
        if hint in text:
            text = text.split(hint)[0]
        
        # Trim if too long
        if len(text) > max_chars:
            text = text[:max_chars]
        
        return text.strip()
    except Exception as e:
        print(f"Warning: Could not read transcript {path}: {e}")
        return ""


def embed(texts, api_base, model, max_retries=3):
    """Get embeddings from LM Studio - with retries because it's flaky"""
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


def normalize_text(text):
    """Clean up text for embeddings"""
    if not text:
        return ""
    
    text = text.lower()
    
    # Fix number formatting
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    
    # Make financial terms consistent
    text = re.sub(r'\bq(\d)\b', r'q\1', text)
    text = re.sub(r'\beps\b', 'eps', text)
    
    return text.strip()


def create_rich_tokens(eps_actual, eps_forecast, sue=None):
    """Create a compact summary string with key metrics"""
    delta_eps = eps_actual - eps_forecast
    
    if eps_forecast != 0:
        pct_change = (delta_eps / abs(eps_forecast)) * 100.0
        pct_str = f"{pct_change:+.1f}"
    else:
        pct_str = "N/A"
    
    sue_str = f"{sue:+.1f}" if sue is not None else "N/A"
    delta_str = f"{delta_eps:+.2f}"
    
    return f"| SUE={sue_str} | ΔEPS={delta_str} | %{pct_str} |"


def cosine(a, b):
    """Calculate cosine similarity"""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def compute_sue(eps_actual, eps_forecast, eps_std):
    """Calculate SUE (Standardized Unexpected Earnings)"""
    if eps_std == 0 or eps_std is None:
        return None
    
    return (eps_actual - eps_forecast) / eps_std


def get_fallback_bucket_and_narrative(eps_actual: float, eps_forecast: float):
    """Generate narrative when we don't have SUE data"""
    TOL = 0.005  # Anything under 0.5¢ is basically zero
    D_CUTS = (0.05, 0.15, 0.30)   # Dollar cutoffs
    P_CUTS = (3.0, 10.0, 20.0)    # Percentage cutoffs

    def sign_with_tol(x):
        if x > TOL: return 1
        if x < -TOL: return -1
        return 0

    def bucket_from_dollar(d):
        """Bucket based on dollar difference"""
        if d <= -D_CUTS[2]: return "very_large_miss"
        if d <= -D_CUTS[1]: return "strong_miss"
        if d <= -D_CUTS[0]: return "modest_miss"
        if -D_CUTS[0] < d < D_CUTS[0]: return "slight_beat" if d >= 0 else "slight_miss"
        if d < D_CUTS[1]: return "modest_beat"
        if d < D_CUTS[2]: return "strong_beat"
        return "very_large_beat"

    def bucket_from_percent(p):
        """Bucket based on percentage change"""
        if p <= -P_CUTS[2]: return "very_large_miss"
        if p <= -P_CUTS[1]: return "strong_miss"
        if p <= -P_CUTS[0]: return "modest_miss"
        if -P_CUTS[0] < p < 0: return "slight_miss"
        if 0 <= p < P_CUTS[0]: return "slight_beat"
        if p < P_CUTS[1]: return "modest_beat"
        if p < P_CUTS[2]: return "strong_beat"
        return "very_large_beat"

    dollar_delta = eps_actual - eps_forecast
    s_act = sign_with_tol(eps_actual)
    s_fcst = sign_with_tol(eps_forecast)

    # Both basically zero
    if s_act == 0 and s_fcst == 0:
        return ("slight_beat",
                f"The company reported EPS of ${eps_actual:.2f} versus a forecast of ${eps_forecast:.2f}, "
                f"meeting expectations exactly with no earnings surprise.")

    # Profit to loss
    if s_fcst > 0 and s_act <= 0:
        b = bucket_from_dollar(dollar_delta)
        return (b,
                f"The company reported EPS of ${eps_actual:.2f} vs. ${eps_forecast:.2f} forecast, "
                f"an unexpected loss/zero outcome representing a {b.replace('_', ' ')} (${dollar_delta:.2f} swing).")

    # Loss to profit
    if s_fcst < 0 and s_act >= 0:
        b = bucket_from_dollar(dollar_delta)
        return (b,
                f"The company reported EPS of ${eps_actual:.2f} vs. ${eps_forecast:.2f} forecast, "
                f"returning to profitability/zero with a {b.replace('_', ' ')} (${dollar_delta:.2f} swing).")

    # Both negative (loss to loss)
    if s_fcst < 0 and s_act < 0:
        b = bucket_from_dollar(dollar_delta)
        narrower = abs(eps_actual) < abs(eps_forecast)
        direction = "narrower loss than expected" if narrower else "wider loss than expected"
        return (b,
                f"The company reported a loss of ${eps_actual:.2f} vs. an expected loss of ${eps_forecast:.2f}, "
                f"a {direction} and a {b.replace('_', ' ')} (${dollar_delta:.2f} change).")

    # Both positive (profit to profit)
    if abs(eps_forecast) <= 0.05:
        b = bucket_from_dollar(dollar_delta)
        verb = "beat" if dollar_delta >= 0 else "miss"
        return (b,
                f"The company reported EPS of ${eps_actual:.2f} vs. ${eps_forecast:.2f} forecast, "
                f"a {b.replace('_', ' ')} (${dollar_delta:.2f} {verb}).")

    if 0.05 < abs(eps_forecast) < 0.20:
        pct = (dollar_delta / abs(eps_forecast)) * 100.0
        b = bucket_from_dollar(dollar_delta)
        verb = "beat" if dollar_delta >= 0 else "miss"
        sign = "+" if pct >= 0 else ""
        return (b,
                f"The company reported EPS of ${eps_actual:.2f} vs. ${eps_forecast:.2f} forecast, "
                f"a {b.replace('_', ' ')} (${dollar_delta:.2f} {verb}, {sign}{pct:.1f}%).")

    # Use percentage for bigger numbers
    pct = (dollar_delta / abs(eps_forecast)) * 100.0
    b = bucket_from_percent(pct)
    verb = "beat" if pct >= 0 else "miss"
    sign = "+" if pct >= 0 else ""
    return (b,
            f"The company reported EPS of ${eps_actual:.2f} vs. ${eps_forecast:.2f} forecast, "
            f"a {b.replace('_', ' ')} (${dollar_delta:.2f} {verb}, {sign}{pct:.1f}%).")

def get_sue_bucket_and_narrative(sue: float, eps_actual: float, eps_forecast: float):
    """Generate narrative using SUE"""
    TOL = 0.005  # Anything under 0.5¢ is basically zero

    def bucket_from_sue(s):
        """Bucket based on SUE value"""
        if s <= -3:
            return "very_large_miss"
        if -3 < s <= -2:
            return "strong_miss"
        if -2 < s <= -1:
            return "modest_miss"
        if -1 < s < 0:
            return "slight_miss"
        if 0 <= s < 1:
            return "slight_beat"
        if 1 <= s < 2:
            return "modest_beat"
        if 2 <= s < 3:
            return "strong_beat"
        return "very_large_beat"

    def sign_with_tol(x):
        if x > TOL: return 1
        if x < -TOL: return -1
        return 0

    s_actual = sign_with_tol(eps_actual)
    s_forecast = sign_with_tol(eps_forecast)

    # Both basically zero
    if s_actual == 0 and s_forecast == 0:
        return (
            "inline",
            f"The company reported EPS of ${eps_actual:.2f} versus a forecast of ${eps_forecast:.2f}, "
            f"meeting expectations with no earnings surprise (SUE = {sue:.1f})."
        )

    # Profit to loss
    if s_forecast > 0 and s_actual <= 0:
        b = bucket_from_sue(sue)
        return (
            b,
            f"The company reported EPS of ${eps_actual:.2f} vs. ${eps_forecast:.2f} forecast, "
            f"an unexpected loss/zero outcome representing a {b.replace('_', ' ')} (SUE = {sue:.1f})."
        )

    # Loss to profit
    if s_forecast < 0 and s_actual >= 0:
        b = bucket_from_sue(sue)
        return (
            b,
            f"The company reported EPS of ${eps_actual:.2f} vs. ${eps_forecast:.2f} forecast, "
            f"returning to profitability/zero with a {b.replace('_', ' ')} (SUE = {sue:.1f})."
        )

    # Both negative (loss to loss)
    if s_forecast < 0 and s_actual < 0:
        b = bucket_from_sue(sue)
        narrower = abs(eps_actual) < abs(eps_forecast)
        if sue >= 0:
            return (
                b,
                f"The company reported a loss of ${eps_actual:.2f} vs. an expected loss of ${eps_forecast:.2f}, "
                f"a narrower loss than expected and a {b.replace('_', ' ')} (SUE = {sue:.1f})."
            )
        else:
            return (
                b,
                f"The company reported a loss of ${eps_actual:.2f} vs. an expected loss of ${eps_forecast:.2f}, "
                f"a wider loss than expected and a {b.replace('_', ' ')} (SUE = {sue:.1f})."
            )

    # Both positive (profit to profit)
    b = bucket_from_sue(sue)
    if sue >= 0:
        # beats
        if b == "slight_beat":
            text = (f"The company reported EPS of ${eps_actual:.2f} against a forecast of ${eps_forecast:.2f}, "
                    f"a slight positive surprise (SUE = {sue:.1f}). This narrow beat indicates performance "
                    f"was essentially in line with analyst projections, reflecting steady execution.")
        elif b == "modest_beat":
            text = (f"The company reported EPS of ${eps_actual:.2f} versus a forecast of ${eps_forecast:.2f}, "
                    f"a modest positive surprise (SUE = {sue:.1f}). This outcome signals slightly stronger "
                    f"operations and efficiency than analysts had anticipated.")
        elif b == "strong_beat":
            text = (f"The company posted EPS of ${eps_actual:.2f} versus expectations of ${eps_forecast:.2f}, "
                    f"a strong positive surprise (SUE = {sue:.1f}). This upside outcome demonstrates materially "
                    f"better-than-expected profitability and underscores robust demand.")
        else:  # very_large_beat
            text = (f"The company achieved EPS of ${eps_actual:.2f} compared to the ${eps_forecast:.2f} consensus, "
                    f"a very large positive surprise (SUE = {sue:.1f}). This extraordinary deviation from forecasts "
                    f"reflects exceptional performance and remarkable earnings momentum.")
        return b, text
    else:
        # misses
        if b == "slight_miss":
            text = (f"The company reported EPS of ${eps_actual:.2f} versus the ${eps_forecast:.2f} forecast, "
                    f"a slight negative surprise (SUE = {sue:.1f}). This narrow miss suggests results were "
                    f"effectively in line with consensus, with only minor execution gaps.")
        elif b == "modest_miss":
            text = (f"The company delivered EPS of ${eps_actual:.2f} versus the ${eps_forecast:.2f} estimate, "
                    f"a modest negative surprise (SUE = {sue:.1f}). This outcome indicates somewhat weaker results "
                    f"than analysts had projected, reflecting softer execution.")
        elif b == "strong_miss":
            text = (f"The company posted EPS of ${eps_actual:.2f} versus expectations of ${eps_forecast:.2f}, "
                    f"a strong negative surprise (SUE = {sue:.1f}). This degree of underperformance highlights "
                    f"materially weaker-than-expected results.")
        else:  # very_large_miss
            text = (f"The company reported EPS of ${eps_actual:.2f} compared to consensus of ${eps_forecast:.2f}, "
                    f"a very large negative surprise (SUE = {sue:.1f}). This severe shortfall signals extraordinary "
                    f"deviation from expectations, likely tied to significant operational challenges or shocks.")
        return b, text

def compute_simple_nnw(transcript, eps_actual, eps_forecast, embed_func, api_base, model, eps_std=None):
    """Compute NNW score"""
    sue = compute_sue(eps_actual, eps_forecast, eps_std)
    
    # Get the narrative
    if sue is not None:
        bucket, numbers_text = get_sue_bucket_and_narrative(sue, eps_actual, eps_forecast)
    else:
        bucket, numbers_text = get_fallback_bucket_and_narrative(eps_actual, eps_forecast)
    
    # Clean up the transcript
    words = transcript.split()
    if len(words) > 200:
        # Skip the usual greeting stuff
        start_idx = 0
        greeting_patterns = [
            'good morning', 'good afternoon', 'good evening', 'welcome', 
            'thank you', 'hello', 'hi', 'thank you for joining', 'operator',
            'conference call', 'earnings call', 'quarterly call'
        ]
        
        for i, word in enumerate(words[:50]):
            if any(pattern in ' '.join(words[i:i+3]).lower() for pattern in greeting_patterns):
                start_idx = i + 3
                break
        
        # Skip the ending stuff too
        end_idx = len(words)
        closing_patterns = [
            'thank you', 'goodbye', 'good bye', 'have a good', 'that concludes',
            'end of call', 'call is concluded', 'operator', 'questions'
        ]
        
        for i in range(len(words) - 50, len(words)):
            if any(pattern in ' '.join(words[i:i+3]).lower() for pattern in closing_patterns):
                end_idx = i
                break
        
        # Get the good stuff
        meaningful_words = words[start_idx:end_idx]
        if len(meaningful_words) > 200:
            transcript_extract = ' '.join(meaningful_words[:100] + meaningful_words[-100:])
        else:
            transcript_extract = ' '.join(meaningful_words)
        
        # Look for financial stuff
        financial_sentences = []
        sentences = transcript.split('.')
        
        financial_patterns = [
            'eps', 'earnings per share', 'earnings', 'profit', 'income', 'revenue',
            'beat', 'miss', 'exceed', 'outperform', 'underperform', 'surprise',
            'forecast', 'guidance', 'expectations', 'target', 'projection',
            'growth', 'increase', 'decrease', 'rise', 'fall', 'up', 'down',
            'strong', 'weak', 'robust', 'solid', 'challenging', 'difficult',
            'outstanding', 'exceptional', 'disappointing', 'concerning',
            'margin', 'profitability', 'operational', 'execution', 'performance',
            'results', 'quarterly', 'annual', 'year-over-year', 'sequential',
            'percent', '%', 'basis points', 'million', 'billion', 'thousand',
            'dollar', '$', 'cents', 'share', 'shares'
        ]
        
        positive_words = ['strong', 'exceed', 'outperform', 'growth', 'record', 'pleased', 
                         'excellent', 'robust', 'solid', 'positive', 'improve', 'increase', 
                         'rise', 'gain', 'success', 'momentum', 'expansion', 'outstanding', 
                         'exceptional', 'impressive', 'deliver', 'beat', 'surpass', 'outpace']
        
        negative_words = ['challenge', 'difficult', 'disappoint', 'pressure', 'headwind', 
                         'decline', 'weak', 'struggle', 'concern', 'risk', 'uncertain', 
                         'volatile', 'decrease', 'fall', 'drop', 'miss', 'underperform', 
                         'disappointing', 'concerning', 'struggle', 'headwinds', 'pressure']
        
        # Grab the relevant sentences
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if len(sentence_lower) > 20:
                has_financial = any(pattern in sentence_lower for pattern in financial_patterns)
                has_positive = any(word in sentence_lower for word in positive_words)
                has_negative = any(word in sentence_lower for word in negative_words)
                
                if has_financial or has_positive or has_negative:
                    financial_sentences.append(sentence.strip())
        
        # Mix it all together
        if financial_sentences:
            top_financial_sentences = financial_sentences[:5]
            financial_text = ' '.join(top_financial_sentences)
            
            if len(transcript_extract.split()) + len(financial_text.split()) < 300:
                transcript_extract = transcript_extract + ' ' + financial_text
            else:
                transcript_extract = financial_text
    else:
        transcript_extract = transcript
    
    # Do the actual computation
    try:
        rich_tokens = create_rich_tokens(eps_actual, eps_forecast, sue)
        
        transcript_normalized = normalize_text(transcript_extract)
        numbers_normalized = normalize_text(numbers_text)
        
        transcript_enhanced = f"{transcript_normalized} {rich_tokens}"
        numbers_enhanced = f"{numbers_normalized} {rich_tokens}"
        
        embeddings = embed_func([transcript_enhanced, numbers_enhanced], api_base, model)
        transcript_embedding, numbers_embedding = embeddings
        
        transcript_embedding = np.array(transcript_embedding, dtype=np.float32)
        numbers_embedding = np.array(numbers_embedding, dtype=np.float32)
        
        cosine_sim = cosine(transcript_embedding, numbers_embedding)
        nnw_score = 1.0 - cosine_sim
        
        return cosine_sim, nnw_score
    except Exception as e:
        raise Exception(f"Failed to compute NNW: {e}")


def parse_float_safe(value):
    """Parse float values - handles the messy data"""
    if pd.isna(value):
        return None
    
    str_value = str(value).strip()
    if not str_value:
        return None
    
    str_value = re.sub(r'[,\s]+', '', str_value)
    
    try:
        return float(str_value)
    except ValueError:
        return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="NNW score calculator")
    parser.add_argument("--eps_csv", default="data/eps_comparison.csv", 
                       help="EPS data file")
    parser.add_argument("--transcripts_dir", default="transcripts", 
                       help="Transcript files directory")
    parser.add_argument("--out_csv", default="nnw_output.csv", 
                       help="Output file")
    parser.add_argument("--tickers", help="Filter by tickers (comma separated)")
    parser.add_argument("--periods", help="Filter by periods (comma separated)")
    parser.add_argument("--model", default="text-embedding-bge-m3", 
                       help="LM Studio model")
    parser.add_argument("--api_base", default="http://localhost:1234/v1", 
                       help="LM Studio API URL")
    parser.add_argument("--max_chars", type=int, default=8000, 
                       help="Max transcript length")
    parser.add_argument("--qa_split_hint", default="Q&A", 
                       help="Where to cut off transcript")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.eps_csv):
        print(f"Error: EPS CSV file not found: {args.eps_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.transcripts_dir):
        print(f"Error: Transcripts directory not found: {args.transcripts_dir}")
        sys.exit(1)
    
    # Load the data
    try:
        df = pd.read_csv(args.eps_csv)
        df = clean_cols(df)
        
        required_cols = ['eps_name', 'quarter', 'actual_eps', 'forecast_eps']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        
        # Look for std dev column
        eps_std_col = None
        possible_std_cols = ['eps_std', 'std_eps', 'standard_deviation', 'stdev', 'eps_stdev']
        for col in possible_std_cols:
            if col in df.columns:
                eps_std_col = col
                break
        
        if eps_std_col:
            print(f"Found standard deviation column: {eps_std_col}")
        else:
            print("Warning: No standard deviation column found. SUE calculation will use fallback")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    # Apply any filters
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
    
    # Go through each row
    results = []
    skipped = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing NNW scores"):
        ticker = str(row['eps_name']).strip()
        period = str(row['quarter']).strip()
        
        # Get the EPS numbers
        eps_actual = parse_float_safe(row['actual_eps'])
        eps_forecast = parse_float_safe(row['forecast_eps'])
        eps_std = parse_float_safe(row[eps_std_col]) if eps_std_col else None
        
        if eps_actual is None or eps_forecast is None:
            print(f"Warning: Skipping {ticker} {period} - invalid EPS values")
            skipped += 1
            continue
        
        # Find the transcript
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
        
        # Read the transcript
        transcript_text = read_and_trim_transcript(
            transcript_path, args.qa_split_hint, args.max_chars
        )
        
        if not transcript_text:
            print(f"Warning: Empty transcript for {ticker} {period}")
            skipped += 1
            continue
        
        try:
            # Calculate the NNW score
            cosine_sim, nnw = compute_simple_nnw(
                transcript_text, eps_actual, eps_forecast, 
                embed, args.api_base, args.model, eps_std
            )
            
            sue = compute_sue(eps_actual, eps_forecast, eps_std)
            
            results.append({
                'ticker': ticker,
                'period': period,
                'eps_actual': eps_actual,
                'eps_forecast': eps_forecast,
                'eps_std': eps_std,
                'sue': sue,
                'transcript_path': transcript_path,
                'cosine': cosine_sim,
                'nnw': nnw
            })
            
        except Exception as e:
            print(f"Warning: Failed to process {ticker} {period}: {e}")
            skipped += 1
            continue
    
    # Save everything
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.out_csv, index=False)
        print(f"\nResults written to {args.out_csv}")
        
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
