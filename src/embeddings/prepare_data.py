import json
import argparse
from pathlib import Path
import os
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm

def combine_spans(interval_list, dest_text, REMOVE_THRESHOLD, desc=None):
    """Combine intervals that overlap or are separated only by whitespace."""
    if not interval_list:
        return []
    
    intervals = sorted(interval_list, key=lambda x: x[0])
    combined = [intervals[0]]
    
    for current_start, current_end in intervals[1:]:
        last_start, last_end = combined[-1]
        
        if current_start <= last_end:
            # Overlapping or adjacent
            combined[-1] = (last_start, max(last_end, current_end))
        elif dest_text[last_end:current_start].strip() == "":
            # Separated only by whitespace
            combined[-1] = (last_start, current_end)
        else:
            # Separated by non-whitespace content
            combined.append((current_start, current_end))
    
    # Filter by threshold and add description
    if desc is None:
        return [(item[0], item[1]) for item in combined if item[1] - item[0] > REMOVE_THRESHOLD]
    else:
        return [(item[0], item[1], desc) for item in combined if item[1] - item[0] > REMOVE_THRESHOLD]
def remove_text(text, intervals, remove_labels = None):
    """Remove text within specified intervals."""
    if not intervals:
        return text
    
    # Sort and merge overlapping intervals
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for start, end, label in sorted_intervals[1:]:
        if label not in remove_labels:
            continue
        last_start, last_end, last_label = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end), last_label)
        else:
            merged.append((start, end, label))
    
    # Build result keeping text outside intervals
    result = []
    prev_end = 0
    
    for start, end, _ in merged:
        result.append(text[prev_end:start])
        prev_end = end
    
    result.append(text[prev_end:])
    return ''.join(result)


def process_patient_notes(patient_df, note_identifier):
    """Process all notes for a single patient efficiently."""

    # Get list of all copyforward notes
    patient_df['parsed'] =  patient_df['copyforward'].apply(parse_json_list)
    copyforward_df = patient_df.explode('parsed')
    mask = copyforward_df['parsed'].notna()
    copyforward_df = pd.json_normalize(copyforward_df[mask]['parsed']).rename(columns={'src_note_csn': 'note_csn_id', 'text': 'full_note_text'})
    
    if copyforward_df.shape[0] != 0: # process copyforward df if there are copyforward entries
        copyforward_df['processed_spans'] = [[] for _ in range(len(copyforward_df))]
        copyforward_df[note_identifier] = patient_df.iloc[0][note_identifier]
        copyforward_df["upd_aut_local_dttm"] = pd.to_datetime(copyforward_df["upd_aut_local_dttm"])

        # Concatenate to patient_df and drop duplicates
        patient_df = pd.concat([patient_df, copyforward_df]) 
    patient_df = patient_df.drop_duplicates(subset='note_csn_id', keep='first') # this will keep the trace version, since copyforward is appended at end
    patient_df['upd_aut_local_dttm']  = pd.to_datetime(patient_df["upd_aut_local_dttm"])

    # Sort by date (most to least recent)
    patient_df = patient_df.sort_values(by="upd_aut_local_dttm", ascending=False)

    # iterate through the copyforward_stage2_spans
    # if the original note is in the dataset --> do nothing
    # if the original note is not in the data --> find the earliest note with the copyforward span, and remove the removal tag
    
    
    # # Pre-allocate lists for concatenation
    # original_parts = []
    # tagged_parts = []
    # removed_parts = []
    # removedT_parts = []
    # removedC_parts = []
    original_parts = []
    removed_parts = []
    removed1_parts = []
    removed2_parts = []
    removedT_parts = []
    removedC_parts = []
    for _, row in patient_df.iterrows():
        text = row['full_note_text']
        spans = row['processed_spans']
        note_header = f'Note ID: {row["note_csn_id"]} \t Note Date: {row["upd_aut_local_dttm"]}\n'
        
        # Process all versions
        original_parts.append(note_header + text)
        removed_parts.append(note_header + remove_text(text, spans, remove_labels=['T1', 'T2', 'C1', 'C2']))
        removed1_parts.append(note_header + remove_text(text, spans, remove_labels=['T1', 'C1']))
        removed2_parts.append(note_header + remove_text(text, spans, remove_labels=['T2', 'C2']))
        removedT_parts.append(note_header + remove_text(text, spans, remove_labels=['T1', 'T1']))
        removedC_parts.append(note_header + remove_text(text, spans, remove_labels=['C2', 'C2']))

    
    # Join with newlines once per patient
    return {
        note_identifier: int(patient_df.iloc[0][note_identifier]),
        'original': '\n'.join(original_parts),
        'removed': '\n'.join(removed_parts),
        'removed_template':  '\n'.join(removedT_parts),
        'removed_copy':  '\n'.join(removedC_parts),
        'removed_stage1':  '\n'.join(removed1_parts),
        'removed_stage2':  '\n'.join(removed2_parts),
    }

def remove_invalid_copyforward_spans(
    df: pd.DataFrame,
    spans_col: str = "copyforward_spans_stage2",
    note_id_col: str = "note_id",
    time_col: str | None = None,  # optional timestamp column
) -> pd.DataFrame:
    """
    Removes copyforward span objects whose src note_id is missing
    from the earliest note containing that span.
    """

    df = df.copy()

    # Fast lookup
    note_id_set = set(df[note_id_col].astype(str))

    # Explode spans
    exploded = (
        df[[note_id_col, time_col, spans_col]]
        .explode(spans_col)
        .dropna(subset=[spans_col])
    )
    if len(exploded) == 0:
        return df

    # Normalize span dicts
    span_df = pd.concat(
        [
            exploded.drop(columns=[spans_col]),
            exploded[spans_col].apply(pd.Series),
        ],
        axis=1,
    )

    span_df["src"] = span_df["src"].astype(str)

    # Only spans whose src note is missing
    missing_src_spans = span_df[~span_df["src"].isin(note_id_set)]

    if missing_src_spans.empty:
        return df

    # Find earliest note per missing src (by timestamp)
    earliest = (
        missing_src_spans
        .sort_values(time_col)
        .groupby("src", as_index=False)
        .first()
    )

    # Build lookup:
    # note_id -> set of span signatures to remove
    to_remove = {}
    for _, row in earliest.iterrows():
        nid = row[note_id_col]
        span_sig = (row["src"], row["start"], row["end"], row["text"])
        to_remove.setdefault(nid, set()).add(span_sig)

    # Remove only from earliest note
    def filter_spans(note_id, spans):
        if note_id not in to_remove or not isinstance(spans, list):
            return spans

        remove_set = to_remove[note_id]
        return [
            s for s in spans
            if (str(s.get("src")), s.get("start"), s.get("end"), s.get("text"))
            not in remove_set
        ]

    df[spans_col] = [
        filter_spans(nid, spans)
        for nid, spans in zip(df[note_id_col], df[spans_col])
    ]
    return df

def process_spans(row,threshold):
    """Process template and copyforward spans."""
    template_spans = []
    cf_spans = []
    template_spans2 = []
    cf_spans2 = [] 
    if 'template_spans' in row.keys() and isinstance(row['template_spans'], list):
        intervals = [(x['start'], x['end']) for x in row['template_spans']]
        template_spans = combine_spans(intervals, row['full_note_text'], desc='T1', REMOVE_THRESHOLD=threshold)
    
    if 'copyforward_spans' in row.keys() and isinstance(row['copyforward_spans'], list):
        intervals = [(x['start'], x['end']) for x in row['copyforward_spans']]
        cf_spans = combine_spans(intervals, row['full_note_text'], desc='C1', REMOVE_THRESHOLD=threshold)
    
    if 'template_spans_stage2' in row.keys() and isinstance(row['template_spans_stage2'], list):
        intervals = [(x['start'], x['end']) for x in row['template_spans_stage2']]
        template_spans2 = combine_spans(intervals, row['full_note_text'], desc='T2', REMOVE_THRESHOLD=threshold)
    
    if 'copyforward_spans_stage2'  in row.keys() and isinstance(row['copyforward_spans_stage2'], list):
        intervals = [(x['start'], x['end']) for x in row['copyforward_spans_stage2']]
        cf_spans2 = combine_spans(intervals, row['full_note_text'], desc='C2', REMOVE_THRESHOLD=threshold)
    return template_spans + cf_spans + template_spans2 + cf_spans2
def parse_json_list(item):
    # Item is already a list, not a string
    if isinstance(item, list) and len(item) > 0:
        return item
    return None
def process_file(file_path, args, output_file, meta):
    processed_count = 0
    
    with open(file_path, 'r') as f:
        for l in f:
            item = json.loads(l)
        
            if 'notes' not in item:
                continue
            patient_df = pd.DataFrame(item['notes'])
            patient_df["upd_aut_local_dttm"] = pd.to_datetime(patient_df["upd_aut_local_dttm"])
            meta_subset = meta[meta[args.meta_identifier]==item[args.meta_identifier]]
            verbose=False # added for debugging
            # print(patient_df['upd_aut_local_dttm'])
            for idx, row in meta_subset.iterrows(): # iterating if multiple cut offs for a single patient
                patient_df_temp = patient_df.copy()
                if verbose:
                    print('Delivery: ', row[args.filter])
                if args.filter_offset_end or args.filter:
                    filter = pd.to_datetime(row[args.filter])
                    patient_df_temp["upd_aut_local_dttm"] = pd.to_datetime(patient_df_temp["upd_aut_local_dttm"])
                    if args.filter_offset_end:
                        filter += pd.offsets.DateOffset(hours=args.filter_offset_end)
                    patient_df_temp = patient_df_temp[
                        filter >= patient_df_temp["upd_aut_local_dttm"]
                    ]
                    # print(filter, patient_df_temp.shape[0])
                if verbose:
                    print(filter)
                if args.filter_offset_start:
                    # filter = pd.to_datetime(row[args.filter])
                    if args.filter_offset_start:
                        filter += pd.offsets.DateOffset(hours=args.filter_offset_start)
                    patient_df_temp["upd_aut_local_dttm"] = pd.to_datetime(patient_df_temp["upd_aut_local_dttm"])
                    patient_df_temp = patient_df_temp[
                        filter <= patient_df_temp["upd_aut_local_dttm"]
                    ]
                if args.filter_prior:
                    filter = pd.to_datetime(row[args.filter_prior])
                    patient_df_temp["upd_aut_local_dttm"] = pd.to_datetime(patient_df_temp["upd_aut_local_dttm"])
                    patient_df_temp = patient_df_temp[
                        filter <= patient_df_temp["upd_aut_local_dttm"]
                    ]
                    # print(filter, patient_df_temp.shape[0])
                if verbose:
                    print(filter)
                    print(patient_df_temp.shape[0])
                if patient_df_temp.shape[0] == 0:
                    continue
                patient_df_temp=remove_invalid_copyforward_spans(patient_df_temp, time_col="upd_aut_local_dttm")

                patient_df_temp['processed_spans'] = patient_df_temp.apply(
                    process_spans, threshold=args.threshold, axis=1
                )

                result = process_patient_notes(patient_df_temp, args.note_identifier)

                result['variables'] = {}
                for variable in args.variables.split(','):
                    result['variables'][variable] = str(item.get(variable, None))

                with open(output_file, 'a') as out:
                    json.dump(result, out)
                    out.write('\n')

                processed_count += 1
        
    print(f"Processed {processed_count} patients from {file_path}")


def main(args):
    print("Processing patients...")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build list of files
    if os.path.isfile(args.notes):
        files = [args.notes]
    else:
        files = [
            os.path.join(args.notes, f)
            for f in os.listdir(args.notes)
            if f.endswith(".jsonl")
        ]

    num_workers = min(cpu_count(), len(files))
    print(f"Using {num_workers} worker processes")
    # read in meta 
    meta=pd.read_csv(args.meta)
    if 'mrn' in meta.columns:
        meta['mrn'] = meta['mrn'].astype(str).apply(lambda x: '{0:0>8}'.format(x))
    else:
        meta[args.meta_identifier] = meta[args.meta_identifier].astype(str)

    tasks = []
    for file_path in files:
        out_file = output_dir / f"{Path(file_path).stem}_processed.jsonl"
        with open(out_file, 'w') as f: # clear the file
            f.write('')

        tasks.append((file_path, args, out_file, meta))

    with Pool(processes=num_workers) as pool:
        pool.starmap(process_file, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # need to add meta in case multiple patients
    parser.add_argument('--meta', type=str, default=None)
    parser.add_argument('--meta_identifier', type=str, default="mrn") # adjustinng for mimic
    parser.add_argument('--note_identifier', type=str, default="pat_mrn_id") # adjusting for mimic
    parser.add_argument('--filter', type=str, default=None)
    parser.add_argument('--filter_offset_start', type=int, default=None) # pph prediction = -8760 hour
    parser.add_argument('--filter_offset_end', type=int, default=None) # pph prediction = -1 hour
    parser.add_argument('--filter_prior', type=str, default=None) # admission
    parser.add_argument('--notes', type=str, required=True)
    parser.add_argument('--threshold', type=int, required=True)
    parser.add_argument('--variables', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    main(args)
