import json
import argparse
from pathlib import Path
import csv

def read_meta(filename):
    """
    Parse meta file and organize by MRN
    """
    data = {}
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data[str(row['mrn'])] = row
    return data


def main(args):
    notes = {}
    patient_ids = set()
    meta = read_meta(args.meta) 
    # Read stage1 notes
    print("Reading file 1..")
    if args.stage1 != None:
        with open(args.stage1, 'r') as f:
            for l in f:
                item = json.loads(l)
                item['pat_mrn_id'] = str(item['pat_mrn_id'])
                if item['pat_mrn_id'] in meta.keys():
                    notes[item['note_csn_id']] = item
                    patient_ids.add(item['pat_mrn_id'])
    print("Reading file 2..")
    # Read stage2 notes and merge
    if args.stage2 != None:
        with open(args.stage2, 'r') as f:
            for l in f:
                item = json.loads(l) 
                item['pat_mrn_id'] = str(item['pat_mrn_id'])
                if item['pat_mrn_id'] in meta.keys():
                    if args.stage1 == None:
                        notes[item['note_csn_id']] = item
                    notes[item['note_csn_id']]['template_spans_stage2'] = item['template_spans_stage2']
                    notes[item['note_csn_id']]['copyforward_spans_stage2'] = item['copyforward_spans_stage2']
                    patient_ids.add(item['pat_mrn_id'])

    for _, value in notes.items():
        pat_id = str(value['pat_mrn_id'])  
        if "notes" not in meta[pat_id]:  
            meta[pat_id]["notes"] = [] 
        meta[pat_id]["notes"].append(value)
    
    patient_ids = list(patient_ids)
    patient_ids.sort()
    print("Keeping ", len(patient_ids), " patients...")
    
    # Create output directory if needed
    Path(args.output).mkdir(parents=True, exist_ok=True)
    print("Writing to directory..")
    # Write chunks of 1000 patients
    for idx in range(0, len(patient_ids), 1000):
        with open(f'{args.output}/chunk_{idx}.jsonl', 'w') as f:
            chunk_size = min(1000, len(patient_ids) - idx)
            for offset in range(chunk_size):
                patient_data = meta[patient_ids[idx + offset]] 
                f.write(json.dumps(patient_data))
                f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process patient notes with span tagging and removal")
    parser.add_argument('--meta', type=str, required=True, help="CSV metafile of patients, requires `mrn` column")
    parser.add_argument('--stage1', type=str, help="JSONL of stage 1 notes")
    parser.add_argument('--stage2', type=str, help="JSONL of stage 2 notes") 
    parser.add_argument('--output', type=str, required=True, help="Output directory")
    args = parser.parse_args()
    main(args)
