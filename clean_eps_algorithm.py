#!/usr/bin/env python3
"""
Clean EPS Comparison Algorithm
Corrected version that uses transcript dates and FPEDATS to determine quarters
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_data():
    """Load all required data files"""
    print("Loading data files...")
    
    # Load mapping file
    mapping_df = pd.read_csv('data/transcript_eps_linked_results_backup.csv')
    print(f"Loaded mapping file: {len(mapping_df)} records")
    
    # Load EPS summary data
    eps_summary_df = pd.read_csv('data/EPS_summary.txt', sep='\t')
    print(f"Loaded EPS summary: {len(eps_summary_df)} records")
    
    # Load EPS actual data
    eps_actual_df = pd.read_csv('data/EPS_unadjusted_actual_full.txt', sep='\t')
    print(f"Loaded EPS actual: {len(eps_actual_df)} records")
    
    return mapping_df, eps_summary_df, eps_actual_df

def extract_transcript_info(mapping_df):
    """Extract transcript information from mapping file"""
    print("Extracting transcript information...")
    
    # Create transcript dataframe
    transcript_data = []
    
    for _, row in mapping_df.iterrows():
        transcript_data.append({
            'transcript_name': row['transcript_name'],
            'eps_name': row['eps_name'],
            'confidence': row['confidence'],
            'filepath': row['transcript_filepath'],
            'eps_value': row['eps_value']
        })
    
    transcript_df = pd.DataFrame(transcript_data)
    
    # Extract date from filepath (assuming format: Company_YYYYQN.txt)
    def extract_date_from_filepath(filepath):
        try:
            filename = os.path.basename(filepath)
            # Remove .txt extension
            name_part = filename.replace('.txt', '')
            # Split by underscore and get last part
            date_part = name_part.split('_')[-1]
            
            # Parse YYYYQN format
            if len(date_part) == 6 and date_part[4] == 'Q':
                year = int(date_part[:4])
                quarter = int(date_part[5])
                
                # Convert quarter to month
                quarter_to_month = {1: 3, 2: 6, 3: 9, 4: 12}
                month = quarter_to_month[quarter]
                
                # Create date (end of quarter)
                return pd.Timestamp(year, month, 1) + pd.offsets.QuarterEnd()
            else:
                return None
        except:
            return None
    
    transcript_df['transcript_date'] = transcript_df['filepath'].apply(extract_date_from_filepath)
    
    # Remove rows where we couldn't extract date
    transcript_df = transcript_df.dropna(subset=['transcript_date'])
    
    print(f"Extracted {len(transcript_df)} transcripts with valid dates")
    return transcript_df

def determine_quarter_from_fpedats(fpedats):
    """Determine quarter from FPEDATS"""
    try:
        date = pd.to_datetime(fpedats)
        month = date.month
        
        if month in [1, 2, 3]:
            return 1  # Q1
        elif month in [4, 5, 6]:
            return 2  # Q2
        elif month in [7, 8, 9]:
            return 3  # Q3
        elif month in [10, 11, 12]:
            return 4  # Q4
        else:
            return None
    except:
        return None

def match_eps_data(transcript_df, eps_summary_df, eps_actual_df):
    """Match transcript data with EPS forecasts and actuals"""
    print("Matching EPS data...")
    
    results = []
    
    for _, transcript in transcript_df.iterrows():
        print(f"Processing: {transcript['transcript_name']}")
        
        # Step 1: Find EPS actual records for this company
        company_actuals = eps_actual_df[
            (eps_actual_df['CUSIP'].notna()) & 
            (eps_actual_df['VALUE'].notna()) &
            (eps_actual_df['VALUE'] != 'NA')
        ].copy()
        
        if len(company_actuals) == 0:
            continue
            
        # Step 2: Find actuals where ANNDATS is close to transcript date
        company_actuals['ANNDATS'] = pd.to_datetime(company_actuals['ANNDATS'])
        company_actuals['date_diff'] = abs((company_actuals['ANNDATS'] - transcript['transcript_date']).dt.days)
        
        # Keep only records within 7 days
        close_actuals = company_actuals[company_actuals['date_diff'] <= 7]
        
        if len(close_actuals) == 0:
            continue
            
        # Get the closest match
        closest_actual = close_actuals.loc[close_actuals['date_diff'].idxmin()]
        
        # Step 3: Determine quarter from FPEDATS
        quarter = determine_quarter_from_fpedats(closest_actual['PENDS'])
        if quarter is None:
            continue
            
        # Step 4: Find forecasts for this company and quarter
        company_forecasts = eps_summary_df[
            (eps_summary_df['CNAME'].str.upper() == transcript['eps_name'].upper()) &
            (eps_summary_df['FISCALP'] == 'QTR')
        ].copy()
        
        if len(company_forecasts) == 0:
            continue
            
        # Convert FPEDATS to datetime for comparison
        company_forecasts['FPEDATS'] = pd.to_datetime(company_forecasts['FPEDATS'])
        
        # Convert STATPERS to datetime
        company_forecasts['STATPERS'] = pd.to_datetime(company_forecasts['STATPERS'])
        
        # Find forecasts for the same quarter
        quarter_forecasts = company_forecasts[
            company_forecasts['FPEDATS'].dt.month.isin([3, 6, 9, 12]) &
            (company_forecasts['FPEDATS'].dt.month == [3, 6, 9, 12][quarter-1])
        ]
        
        if len(quarter_forecasts) == 0:
            continue
            
        # Get the most recent forecast before the transcript date
        quarter_forecasts = quarter_forecasts[quarter_forecasts['STATPERS'] <= transcript['transcript_date']]
        
        if len(quarter_forecasts) == 0:
            continue
            
        latest_forecast = quarter_forecasts.loc[quarter_forecasts['STATPERS'].idxmax()]
        
        # Step 5: Create result record
        result = {
            'EPS_Name': transcript['eps_name'],
            'Transcript_Company_Name': transcript['transcript_name'],
            'EPS_Actual': float(closest_actual['VALUE']) if closest_actual['VALUE'] != 'NA' else None,
            'EPS_Forecast': float(latest_forecast['MEANEST']) if pd.notna(latest_forecast['MEANEST']) else None,
            'Transcript_Date': transcript['transcript_date'],
            'ANNDATS': closest_actual['ANNDATS'],
            'CUSIP': closest_actual['CUSIP'],
            'Financial_Period': closest_actual['PENDS'],
            'Confidence': transcript['confidence']
        }
        
        results.append(result)
        print(f"  ✅ Matched: Q{quarter}, Actual: {result['EPS_Actual']}, Forecast: {result['EPS_Forecast']}")
    
    return pd.DataFrame(results)

def validate_results(results_df):
    """Validate the results"""
    print("\n=== VALIDATION RESULTS ===")
    
    # Coverage check
    total_transcripts = len(results_df)
    print(f"✅ Total matched transcripts: {total_transcripts}")
    
    # Check for duplicates
    duplicates = results_df.duplicated(subset=['CUSIP', 'Financial_Period']).sum()
    print(f"✅ Duplicate records: {duplicates}")
    
    # Check temporal alignment
    results_df['date_diff'] = abs((results_df['Transcript_Date'] - results_df['ANNDATS']).dt.days)
    temporal_issues = (results_df['date_diff'] > 7).sum()
    print(f"✅ Temporal alignment issues: {temporal_issues}")
    
    # Check EPS variance
    results_df['eps_delta'] = results_df['EPS_Actual'] - results_df['EPS_Forecast']
    extreme_variance = (results_df['eps_delta'].abs() > 5).sum()
    print(f"✅ Extreme variance cases: {extreme_variance}")
    
    # Show sample results
    print(f"\n=== SAMPLE RESULTS ===")
    print(results_df[['EPS_Name', 'EPS_Actual', 'EPS_Forecast', 'Transcript_Date']].head(10))
    
    return results_df

def main():
    """Main execution function"""
    print("🚀 Starting Clean EPS Algorithm")
    print("=" * 50)
    
    try:
        # Load data
        mapping_df, eps_summary_df, eps_actual_df = load_data()
        
        # Extract transcript information
        transcript_df = extract_transcript_info(mapping_df)
        
        # Match EPS data
        results_df = match_eps_data(transcript_df, eps_summary_df, eps_actual_df)
        
        # Validate results
        results_df = validate_results(results_df)
        
        # Save results
        output_file = 'data/eps_comparison_clean.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
        print(f"✅ Total records: {len(results_df)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
