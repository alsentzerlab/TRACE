"""
Analyze Length of Text
Description: Script to analyze how much text was removed from each component and random spans
"""
import json
import argparse
import os

def main(args):
    with open(f'{args.output}', 'w') as ofile:

        ofile.write('mrn,n_notes,length_original,length_removed,length_removed_template,length_removed_copy,length_removed_stage1,length_removed_stage2,length_random\n')

        for file in os.listdir(args.patient_directory):
            if 'jsonl' in file:
                with open(f'{args.patient_directory}/{file}', 'r') as f:
                    for l in f:
                        data = json.loads(l)
                    
                        ofile.write(f'{data['pat_mrn_id']},{data['removed'].count('Note Date: ')},{len(data['original'])},{len(data['removed'])},{len(data['removed_template'])},{len(data['removed_copy'])},{len(data['removed_stage1'])},{len(data['removed_stage2'])},{len(data['random'])}\n')
                    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process patient notes with span tagging and removal")
    parser.add_argument('--patient_directory', 
                        type=str,
                        required=True,
                        help="Input directory for individual patient JSON files")
    parser.add_argument('--output', 
                        type=str,
                        required=True,
                        help="Output directory for analysis file")
    args = parser.parse_args()
    main(args)