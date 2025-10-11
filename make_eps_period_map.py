# make_eps_period_map.py
import os
import duckdb

TSV_PATH = "/Users/vazea/Desktop/NNW/data/renaming_data/EPS_summary.txt"
PARQUET_FILE = "/Users/vazea/Desktop/NNW/data/renaming_data/eps_summary.parquet"
OUT_TXT = "/Users/vazea/Desktop/NNW/data/renaming_data/eps_period_map.txt"

def ensure_parquet(tsv_path: str, parquet_file: str):
    """Create a single Parquet file from the big TSV if it doesn't exist."""
    if os.path.exists(parquet_file):
        return

    con = duckdb.connect()
    con.execute(f"""
    PRAGMA threads=4;
    COPY (
      SELECT
        TICKER::VARCHAR                           AS TICKER,
        CUSIP::VARCHAR                            AS CUSIP,
        OFTIC::VARCHAR                            AS OFTIC,
        CNAME::VARCHAR                             AS CNAME,
        CAST(STATPERS AS DATE)                    AS STATPERS,
        MEASURE::VARCHAR                          AS MEASURE,
        FISCALP::VARCHAR                          AS FISCALP,
        CAST(FPI AS INT)                          AS FPI,
        ESTFLAG::VARCHAR                          AS ESTFLAG,
        CURCODE::VARCHAR                          AS CURCODE,
        CAST(NUMEST AS INT)                       AS NUMEST,
        CAST(NUMUP AS INT)                        AS NUMUP,
        CAST(NUMDOWN AS INT)                      AS NUMDOWN,
        CAST(MEDEST AS DOUBLE)                    AS MEDEST,
        CAST(MEANEST AS DOUBLE)                   AS MEANEST,
        CAST(STDEV AS DOUBLE)                     AS STDEV,
        CAST(HIGHEST AS DOUBLE)                   AS HIGHEST,
        CAST(LOWEST AS DOUBLE)                    AS LOWEST,
        CAST(USFIRM AS INT)                       AS USFIRM,
        CAST(FPEDATS AS DATE)                     AS FPEDATS
      FROM read_csv('{tsv_path}',
        delim='\t',
        header=true,
        auto_detect=true,
        dateformat='%Y-%m-%d',
        nullstr=['', 'NA', 'NaN', 'null', 'NULL']
      )
    )
    TO '{parquet_file}'
    (FORMAT PARQUET);
    """)
    con.close()

def build_period_map(parquet_file: str, out_txt: str):
    """Build (CUSIP, FPEDATS) → quarter + fiscal-year-end DATE, and export CSV."""
    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")
    sql = f"""
    WITH
    -- Unique (CUSIP, FPEDATS) we need to label: all QTR + Q4 ends (ANN,FPI=1)
    targets AS (
      SELECT DISTINCT CUSIP, FPEDATS
      FROM read_parquet('{parquet_file}')
      WHERE (FISCALP='QTR') OR (FISCALP='ANN' AND FPI=1)
    ),

    -- Fiscal year ends timeline (ANN & FPI=1)
    fye AS (
      SELECT CUSIP, FPEDATS AS FYE_DATE
      FROM read_parquet('{parquet_file}')
      WHERE FISCALP='ANN' AND FPI=1
    ),

    -- Representative TICKER/CNAME per (CUSIP, FPEDATS)
    meta_counts AS (
      SELECT
        CUSIP,
        FPEDATS,
        TICKER,
        CNAME,
        MAX(STATPERS) AS LATEST_STATPERS,
        COUNT(*)      AS CNT
      FROM read_parquet('{parquet_file}')
      WHERE (FISCALP='QTR') OR (FISCALP='ANN' AND FPI=1)
      GROUP BY ALL
    ),
    meta_pick AS (
      SELECT *
      FROM (
        SELECT
          CUSIP, FPEDATS, TICKER, CNAME,
          ROW_NUMBER() OVER (
            PARTITION BY CUSIP, FPEDATS
            ORDER BY CNT DESC, LATEST_STATPERS DESC, TICKER, CNAME
          ) AS rn
        FROM meta_counts
      )
      WHERE rn=1
    ),

    -- As-of previous FYE (latest FYE <= FPEDATS)
    asof_prev AS (
      SELECT
        t.CUSIP, t.FPEDATS, f.FYE_DATE AS PREV_FYE,
        ROW_NUMBER() OVER (
          PARTITION BY t.CUSIP, t.FPEDATS
          ORDER BY f.FYE_DATE DESC
        ) AS rn
      FROM targets t
      LEFT JOIN fye f
        ON t.CUSIP=f.CUSIP AND f.FYE_DATE<=t.FPEDATS
    ),
    picked_prev AS (SELECT * FROM asof_prev WHERE rn=1),

    -- For dates before first known FYE: use earliest next FYE and back off 12 months
    next_fye AS (
      SELECT
        t.CUSIP, t.FPEDATS, MIN(f.FYE_DATE) AS NEXT_FYE
      FROM targets t
      LEFT JOIN fye f
        ON t.CUSIP=f.CUSIP AND f.FYE_DATE>t.FPEDATS
      GROUP BY t.CUSIP, t.FPEDATS
    ),

    coalesced AS (
      SELECT
        p.CUSIP, p.FPEDATS,
        COALESCE(p.PREV_FYE, n.NEXT_FYE - INTERVAL 12 MONTH) AS EFFECTIVE_PREV_FYE
      FROM picked_prev p
      LEFT JOIN next_fye n
        ON p.CUSIP=n.CUSIP AND p.FPEDATS=n.FPEDATS
    ),

    -- Quarter assignment with TRUE month-end cutoffs (handles Feb/leap years).
    -- Also: if FPEDATS == EFFECTIVE_PREV_FYE, that's the FYE date itself => Q4.
    labeled AS (
      SELECT
        c.CUSIP,
        c.FPEDATS,
        c.EFFECTIVE_PREV_FYE,

        -- month_end(prev_fye + k months)
        (date_trunc('month', c.EFFECTIVE_PREV_FYE + INTERVAL 3 MONTH) + INTERVAL 1 MONTH - INTERVAL 1 DAY) AS Q1_END,
        (date_trunc('month', c.EFFECTIVE_PREV_FYE + INTERVAL 6 MONTH) + INTERVAL 1 MONTH - INTERVAL 1 DAY) AS Q2_END,
        (date_trunc('month', c.EFFECTIVE_PREV_FYE + INTERVAL 9 MONTH) + INTERVAL 1 MONTH - INTERVAL 1 DAY) AS Q3_END,
        (date_trunc('month', c.EFFECTIVE_PREV_FYE + INTERVAL 12 MONTH) + INTERVAL 1 MONTH - INTERVAL 1 DAY) AS Q4_END,

        CASE
          WHEN c.FPEDATS = c.EFFECTIVE_PREV_FYE THEN 'Q4'
          WHEN c.FPEDATS <= (date_trunc('month', c.EFFECTIVE_PREV_FYE + INTERVAL 3 MONTH) + INTERVAL 1 MONTH - INTERVAL 1 DAY) THEN 'Q1'
          WHEN c.FPEDATS <= (date_trunc('month', c.EFFECTIVE_PREV_FYE + INTERVAL 6 MONTH) + INTERVAL 1 MONTH - INTERVAL 1 DAY) THEN 'Q2'
          WHEN c.FPEDATS <= (date_trunc('month', c.EFFECTIVE_PREV_FYE + INTERVAL 9 MONTH) + INTERVAL 1 MONTH - INTERVAL 1 DAY) THEN 'Q3'
          ELSE 'Q4'
        END AS FISCAL_QUARTER
      FROM coalesced c
    ),

    -- Compute FYE DATE for each row (full YYYY-MM-DD), plus fiscal year string.
    fyed AS (
      SELECT
        l.CUSIP,
        l.FPEDATS,
        l.FISCAL_QUARTER,
        -- For Q4 at the FYE date, FYE is EFFECTIVE_PREV_FYE.
        -- For Q1–Q3, FYE is month_end(prev_fye + 12 months).
        CASE
          WHEN l.FISCAL_QUARTER = 'Q4'
            THEN l.EFFECTIVE_PREV_FYE
          ELSE (date_trunc('month', l.EFFECTIVE_PREV_FYE + INTERVAL 12 MONTH) + INTERVAL 1 MONTH - INTERVAL 1 DAY)
        END AS FYE_DATE
      FROM labeled l
    ),

    with_meta AS (
      SELECT
        f.CUSIP, m.TICKER, m.CNAME, f.FPEDATS, f.FISCAL_QUARTER, f.FYE_DATE
      FROM fyed f
      LEFT JOIN meta_pick m
        ON f.CUSIP=m.CUSIP AND f.FPEDATS=m.FPEDATS
    ),

    final_rows AS (
      SELECT
        CUSIP,
        COALESCE(TICKER, '') AS TICKER,
        COALESCE(CNAME,  '') AS CNAME,
        CAST(FPEDATS AS DATE) AS FPEDATS,
        CAST(FYE_DATE AS DATE) AS FYE,  -- full date (YYYY-MM-DD)
        CONCAT(
          CAST(EXTRACT(YEAR FROM FYE_DATE) AS VARCHAR),
          'Q',
          CASE FISCAL_QUARTER WHEN 'Q1' THEN '1' WHEN 'Q2' THEN '2'
                              WHEN 'Q3' THEN '3' ELSE '4' END
        ) AS FIN_PERIOD
      FROM with_meta
    )
    SELECT *
    FROM final_rows
    ORDER BY CUSIP,
             FYE,
             CASE SUBSTR(FIN_PERIOD, LENGTH(FIN_PERIOD), 1)
                  WHEN '1' THEN 1 WHEN '2' THEN 2 WHEN '3' THEN 3 ELSE 4 END
    """
    con.execute(f"COPY ({sql}) TO '{out_txt}' WITH (HEADER, DELIMITER ',');")
    con.close()

if __name__ == "__main__":
    ensure_parquet(TSV_PATH, PARQUET_FILE)
    build_period_map(PARQUET_FILE, OUT_TXT)
    print("Done →", OUT_TXT)