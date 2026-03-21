import argparse
import json
from tqdm import tqdm
from run_trace.trace_unsupervised import UnsupervisedTRACE
from run_trace.utils import remove_spans_from_text

def main(args):

    print("\n" + "="*60)
    print("  PREPROCESSING")
    print("="*60 + "\n")
    
    print(f"Loading data from {args.input}...")
    notes = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                note = json.loads(line)
                notes.append(note)
    
    print(f"Loaded {len(notes)} notes\n")
    
    # Check for Stage 1 TRACE spans
    print("TRACE Stage 1: Looking for Copiedand Template Spans")
    has_stage1 = False
    if notes:
        sample_note = notes[0]
        # Check if keys exist and at least one note has non-empty spans
        if 'template_spans' in sample_note and 'copyforward_spans' in sample_note:
            # Check if any note actually has spans
            for note in notes:
                template_spans = note.get('template_spans', [])
                copyforward_spans = note.get('copyforward_spans', [])
                if template_spans or copyforward_spans:
                    has_stage1 = True
                    break
    
    if has_stage1:
        print("TRACE Stage 1: Spans Found")
        print("TRACE Stage 1: Removing Copy Forward and Template Spans")
        
        # Process notes in memory to add stage1_note_text
        for note in tqdm(notes, desc="  Processing", leave=False):
            template_spans = note.get('template_spans', [])
            copyforward_spans = note.get('copyforward_spans', [])
            all_spans = template_spans + copyforward_spans
            full_note_text = note.get('full_note_text', '')
            stage1_note_text = remove_spans_from_text(full_note_text, all_spans)
            # Replace full_note_text with stage1_note_text for unsupervised processing
            note['full_note_text'] = stage1_note_text
        
        print("")
    else:
        print("TRACE Stage 1: Spans Not Found. Moving to Unsupervised TRACE...\n")
    
    ut = UnsupervisedTRACE() # probably want to remove, unnecessary to pipe

    ut.load_notes(notes=notes)
    temp_span_dict, cf_span_dict = ut.run()
    with open(args.output, 'w+') as f:
        for note in tqdm(notes, desc="  Processing", leave=False):
            temp_spans = temp_span_dict.get(note['note_csn_id'], [])
            cf_spans = cf_span_dict.get(note['note_csn_id'], [])
            note['template_spans_stage2'] = temp_spans
            note['copyforward_spans_stage2'] = cf_spans
            json.dump(note, f)
            f.write('\n')

    print(f"Complete. Processed {len(notes)} notes.\n\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', 
                        type=str,
                        required=True,
                        help="Input JSONL file")
    parser.add_argument('--output', 
                        type=str,
                        required=True,
                        help="Output JSONL file")
    args = parser.parse_args()
    main(args)