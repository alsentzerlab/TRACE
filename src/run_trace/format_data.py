import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import warnings
from collections import defaultdict

# pip
import pandas as pd
import tqdm
from google.cloud import bigquery

# Setup logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_client(project, location):
    return bigquery.Client(project=project, location=location)


def fetch_all_data_in_batches(client, sp_tbl, st_tbl, cp_tbl):
    """
    Fetch all smartphrase, smarttext, and copyforward data for given note_csn_ids
    in batches to avoid query size limits.
    
    Returns three dataframes: sp_df, st_df, cp_df
    """
    sp_df = client.query(f"SELECT * FROM {sp_tbl}").to_dataframe()
    sp_df['type'] = 'smartphrase'
    sp_df = sp_df.rename(columns={'smartphrases_id': 'template_id'})
    sp_df['index'] = sp_df.index

    st_df = client.query(f"SELECT * FROM {st_tbl}").to_dataframe()
    st_df['type'] = 'smarttext'
    st_df = st_df.rename(columns={'smarttexts_id': 'template_id'})
    st_df['index'] = st_df.index

    cp_df = client.query(f"""
        SELECT note_csn_id, src_note_csn, upd_aut_local_dttm, full_note_text
        FROM {cp_tbl}
    """).to_dataframe()

    return sp_df, st_df, cp_df


def format_rows(df):
    """Format template rows into string"""
    if df.empty:
        return ""
    
    formatted = ""
    for _, row in df.iterrows():
        header = (
            f"[type={row['type']}] "
            f"[index={row['index']}] "
            f"[template_id={row['template_id']}] "
        )
        content = row.get("text", "")
        formatted += header + content
    
    return formatted


def convert_to_json_serializable(obj):
    """Convert non-JSON-serializable objects to strings"""
    if pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
        return obj.strftime("%Y-%m-%dT%H:%M:%S")
    elif isinstance(obj, (pd.Timedelta)):
        return str(obj)
    elif hasattr(obj, 'isoformat'):  # datetime, date, time objects
        return obj.isoformat()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def build_lookup_dicts(sp_processed, st_processed, cp_df):
    """Pre-build dictionaries for O(1) lookup"""
    logger.info("Building lookup dictionaries...")
    
    sp_lookup = defaultdict(list)
    for _, row in sp_processed.iterrows():
        sp_lookup[row['note_csn_id']].append(row.to_dict())
    
    st_lookup = defaultdict(list)
    for _, row in st_processed.iterrows():
        st_lookup[row['note_csn_id']].append(row.to_dict())
    
    cp_lookup = defaultdict(list)
    for _, row in cp_df.iterrows():
        cp_data = {
            'src_note_csn': convert_to_json_serializable(row['src_note_csn']),
            'upd_aut_local_dttm': convert_to_json_serializable(row['upd_aut_local_dttm']),
            'text': row['full_note_text']
        }
        cp_lookup[row['note_csn_id']].append(cp_data)
    
    logger.info(f"Built lookups for {len(sp_lookup)} notes with smartphrases, "
                f"{len(st_lookup)} with smarttexts, {len(cp_lookup)} with copyforward")
    
    return sp_lookup, st_lookup, cp_lookup


def process_note_batch(note_batch, sp_lookup, st_lookup, cp_lookup):
    """Process a batch of notes. Returns list of JSON strings."""
    results = []
    
    for note_data in note_batch:
        try:
            note_csn_id = note_data['note_csn_id']
            
            sp_rows = sp_lookup.get(note_csn_id, [])
            st_rows = st_lookup.get(note_csn_id, [])
            
            template_list = sp_rows + st_rows
            if template_list:
                template_df = pd.DataFrame(template_list)
                template_string = format_rows(template_df)
            else:
                template_string = ""
            
            cp_list = cp_lookup.get(note_csn_id, [])
            
            output = {key: convert_to_json_serializable(value) for key, value in note_data.items()}
            output['template_string'] = template_string
            output['copyforward'] = cp_list
            
            results.append(json.dumps(output))
            
        except Exception as e:
            logger.error(f"Error processing note {note_data.get('note_csn_id', 'unknown')}: {str(e)}")
            continue
    
    return results


def main(args):
    client = get_client(args.project, args.location)

    logger.info(f"Fetching notes from {args.notes_table}...")
    notes_df = client.query(f"SELECT * FROM {args.notes_table}").to_dataframe()
    notes_df = notes_df.dropna()
    logger.info(f"Found {len(notes_df)} notes to process")

    logger.info("Fetching all smartphrase, smarttext, and copyforward data...")
    sp_df, st_df, cp_df = fetch_all_data_in_batches(
        client, args.smartphrase_table, args.smarttext_table, args.copyforward_table
    )
    logger.info(f"Fetched {len(sp_df)} smartphrase records")
    logger.info(f"Fetched {len(st_df)} smarttext records")
    logger.info(f"Fetched {len(cp_df)} copyforward records")

    sp_lookup, st_lookup, cp_lookup = build_lookup_dicts(sp_df, st_df, cp_df)

    logger.info("Processing notes in batches...")
    notes_list = notes_df.to_dict('records')
    batch_size = max(1000, len(notes_list) // (args.workers * 4))
    logger.info(f"Using batch size of {batch_size} notes per batch")

    batches = [notes_list[i:i + batch_size] for i in range(0, len(notes_list), batch_size)]
    logger.info(f"Split {len(notes_list)} notes into {len(batches)} batches")

    success_count = 0

    with open(args.output, 'w', buffering=8*1024*1024) as outfile:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_note_batch, batch, sp_lookup, st_lookup, cp_lookup): len(batch)
                for batch in batches
            }

            with tqdm.tqdm(total=len(notes_list), desc="Processing notes") as pbar:
                for future in as_completed(futures):
                    for json_line in future.result():
                        outfile.write(json_line + '\n')
                    success_count += len(future.result())
                    pbar.update(futures[future])

    logger.info(f"Processing complete! Successfully processed {success_count} notes")
    logger.info(f"Output written to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process clinical notes with templates and copyforward data")

    # BigQuery config
    parser.add_argument('--project',
                        type=str,
                        default='som-nero-phi-ema2016-ehr',
                        help="GCP project ID (default: som-nero-phi-ema2016-ehr)")
    parser.add_argument('--location',
                        type=str,
                        default='US',
                        help="BigQuery location (default: US)")

    # Table names
    parser.add_argument('--notes-table',
                        type=str,
                        required=True,
                        help="Fully qualified notes table (e.g. project.dataset.notes)")
    parser.add_argument('--smartphrase-table',
                        type=str,
                        required=True,
                        help="Fully qualified smartphrase table (e.g. project.dataset.smartphrase)")
    parser.add_argument('--smarttext-table',
                        type=str,
                        required=True,
                        help="Fully qualified smarttext table (e.g. project.dataset.smarttext)")
    parser.add_argument('--copyforward-table',
                        type=str,
                        required=True,
                        help="Fully qualified copyforward table (e.g. project.dataset.copyforward)")

    # Output
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help="Output JSONL file path (e.g. notes.jsonl)")
    parser.add_argument('--workers',
                        type=int,
                        default=10,
                        help="Number of parallel workers (default: 10)")

    args = parser.parse_args()
    main(args)