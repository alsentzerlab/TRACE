import argparse
import json

from tqdm import tqdm
import os

from run_trace.trace_supervised import SupervisedTRACE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    # Get filename for the tqdm description
    file_label = os.path.basename(args.input)

    # Process line by line to keep memory footprint low
    with open(args.input, 'r', encoding='utf-8') as f_in, \
         open(args.output, 'w', encoding='utf-8') as f_out:
        
        # Count lines for tqdm total
        line_count = sum(1 for _ in open(args.input, 'r'))
        
        for line in tqdm(f_in, total=line_count, desc=f"Processing {file_label}", leave=False, mininterval=10.0):
            line = line.strip()
            if not line:
                continue
                
            item = json.loads(line)
            trace = SupervisedTRACE(item)
            
            if trace.has_trace():
                trace.process_templates()
                trace.process_copyforward()
                # Write immediately to save RAM
            f_out.write(trace.to_string() + '\n')

if __name__ == "__main__":
    main()