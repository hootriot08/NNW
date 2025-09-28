# NNW Score Calculator

A Python tool that compares earnings call transcripts with EPS numbers to calculate Narrative-Number Wedge (NNW) scores.

## What it does

The NNW score measures how much the earnings call narrative differs from the actual EPS numbers:

1. Takes the transcript text and EPS numbers
2. Gets embeddings for both using LM Studio
3. Calculates how similar they are (cosine similarity)
4. NNW = 1 - similarity (higher = more disconnect)

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start LM Studio

1. Download [LM Studio](https://lmstudio.ai/)
2. Load the `bge-small-en-v1.5` model
3. Start the local server (usually http://localhost:1234)

### Data structure

```
NNW/
├── data/
│   └── eps_comparison.csv      # Your EPS data
├── transcripts_by_company/     # Transcript files
├── nnw_minimal.py
└── requirements.txt
```

## CSV Format

Your EPS data needs these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `ticker` | Stock symbol | `AAPL` |
| `period` | Time period | `2024Q2` |
| `eps_actual` | Actual EPS | `1.25` |
| `eps_forecast` | Forecast EPS | `1.20` |
| `transcript_path` | (Optional) Path to transcript | `transcripts/AAPL_2024Q2.txt` |

## Usage

### Basic run

```bash
python3 nnw_minimal.py
```

### Filter data

```bash
python3 nnw_minimal.py --tickers AAPL,MSFT --periods 2024Q2
```

### Custom files

```bash
python3 nnw_minimal.py \
  --eps_csv data/my_data.csv \
  --transcripts_dir my_transcripts \
  --out_csv results.csv
```

## Options

| Option | Default | What it does |
|--------|---------|-------------|
| `--eps_csv` | `data/eps_comparison.csv` | EPS data file |
| `--transcripts_dir` | `transcripts` | Where transcripts are |
| `--out_csv` | `nnw_output.csv` | Output file |
| `--tickers` | None | Filter by tickers |
| `--periods` | None | Filter by periods |
| `--model` | `bge-small-en-v1.5` | LM Studio model |
| `--api_base` | `http://localhost:1234/v1` | API URL |
| `--max_chars` | `8000` | Max transcript length |
| `--qa_split_hint` | `Q&A` | Where to cut transcript |

## Output

The script creates a CSV with:

| Column | Description |
|--------|-------------|
| `ticker` | Stock symbol |
| `period` | Time period |
| `eps_actual` | Actual EPS |
| `eps_forecast` | Forecast EPS |
| `transcript_path` | Transcript file used |
| `cosine` | Similarity (0-1) |
| `nnw` | NNW score (0-1, higher = more disconnect) |

## How it finds transcripts

1. Uses `transcript_path` column if it exists
2. Otherwise searches for files with ticker and period in the name
3. Picks the biggest file if multiple matches
4. Skips with warning if nothing found

## Example output

```
Processing 2 rows...
Computing NNW scores: 100%|████████| 2/2 [00:05<00:00,  2.50s/it]

Results written to nnw_output.csv

Processed: 2 | Skipped: 0

Top NNW scores:
1) MSFT 2024Q2  NNW=0.372  cosine=0.628
2) AAPL 2024Q2  NNW=0.315  cosine=0.685
```

## Troubleshooting

### LM Studio not working
- Make sure it's running on http://localhost:1234
- Check the `bge-small-en-v1.5` model is loaded

### Can't find transcripts
- Check filenames have both ticker and period
- Use `--transcripts_dir` if files are elsewhere

### Memory issues
- Reduce `--max_chars` to limit transcript size
- Process fewer tickers at once