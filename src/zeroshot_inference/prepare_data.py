import json
import argparse
from pathlib import Path
import os
from multiprocessing import Pool, cpu_count
import pandas as pd
from zeroshot_inference.utils import process_spans, process_patient_notes, remove_invalid_copyforward_spans


def process_file(file_path, args, output_file):
    processed_count = 0

    with open(file_path, 'r') as f:
        for l in f:
            item = json.loads(l)

            if 'notes' not in item:
                continue

            patient_df = pd.DataFrame(item['notes'])

            if args.filter:
                patient_df[args.filter] = pd.to_datetime(item.get(args.filter))
                if args.filter_offset:
                    patient_df[args.filter] += pd.offsets.DateOffset(hours=args.filter_offset)
                patient_df = patient_df[
                    patient_df[args.filter] >= patient_df["upd_aut_local_dttm"]
                ]

            if patient_df.shape[0] == 0:
                continue
                
            patient_df =remove_invalid_copyforward_spans(patient_df, time_col="upd_aut_local_dttm")
            patient_df['processed_spans'] = patient_df.apply(
                process_spans, threshold=args.threshold, axis=1
            )
            
            result = process_patient_notes(patient_df)

            result['variables'] = {}
            for variable in args.variables.split(','):
                result['variables'][variable] = item.get(variable, None)
            with open(output_file, 'a') as out:
                json.dump(result, out)
                out.write('\n')
            processed_count += 1
            if processed_count % 50==0:
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

    tasks = []
    for file_path in files:
        out_file = output_dir / f"{Path(file_path).stem}_processed.jsonl"
        with open(out_file, 'w') as f: # clear the file
            f.write('')
        tasks.append((file_path, args, out_file))
        break
    with Pool(processes=num_workers) as pool:
        pool.starmap(process_file, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=str, default=None)
    parser.add_argument('--filter_offset', type=int, default=None)
    parser.add_argument('--notes', type=str, required=True)
    parser.add_argument('--threshold', type=int, required=True)
    parser.add_argument('--variables', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    main(args)
