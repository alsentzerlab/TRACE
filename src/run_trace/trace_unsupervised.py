import re
from collections import defaultdict
from tqdm import tqdm


class UnsupervisedTRACE:
    def __init__(self):
        self.notes = []
        self.note_id_to_note = {} 
    
    def load_notes(self, notes):
        """Load notes from list of dictionaries"""
        self.notes = notes
        # Build fast lookup
        for note in notes:
            self.note_id_to_note[note['note_csn_id']] = note
        print(f"  Loaded {len(self.notes)} notes\n")
    
    def _extract_chunks(self, text):
        """Extract candidate template chunks from text (optimized)"""
        chunks_to_index = defaultdict(list)
        
        # sentence level
        sentences = re.split(r'[.!?]\s+', text)
        
        for sentence in sentences:
            if sum(c.isalpha() for c in sentence) >= 50:
                # get index of sentence
                start = text.find(sentence)
                end = start + len(sentence)
                chunks_to_index[sentence].append([start, end])
        # paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        for para in paragraphs:
            if sum(c.isalpha() for c in para) >= 50:
                # get index of sentence
                start = text.find(para)
                end = start + len(para)
                chunks_to_index[para].append([start, end])

        return chunks_to_index
    
    def _find_frequent_patterns(self, min_notes=1, min_patients=1, max_patients=None):
        """
        Find frequent patterns and record which notes contain them.
        
        Args:
            min_notes: Minimum number of notes required
            min_patients: Minimum number of patients required
            max_patients: Maximum number of patients allowed (None for no limit)
        
        Returns:
            patterns_to_notes = chunks -> [(nid, start, end)]
        """
        chunk_to_notes = defaultdict(set)
        chunk_to_note_list = defaultdict(list)

        print(f"  Threshold:  Min: {min_notes} notes Min: {min_patients} patients" + (f", max {max_patients}" if max_patients else ""))
        for note in tqdm(self.notes, desc=f"  Extracting chunks"):
            nid = note['note_csn_id']
            pid = note.get('pat_mrn_id', '')
            note_date = note.get('upd_aut_local_dttm', '')
            
            for chunk, intervals in self._extract_chunks(note['full_note_text']).items():
                norm = re.sub(r'\s+', ' ', chunk.lower()).strip()
                
                chunk_to_notes[norm].add((nid, pid, note_date, tuple(map(tuple, intervals)), chunk))
                chunk_to_note_list[norm].append((nid, pid, note_date))
        
        pattern_frequencies = {}
        pattern_to_notes = {} 

        for norm, occs in chunk_to_notes.items():
            
            pids = {p for n, p, d, i, c in occs if p}
            
            if len(pids) < min_patients:
                continue
            if max_patients is not None and len(pids) > max_patients:
                continue
            
            # Check note count constraint
            if len(occs) < min_notes:
                continue
            
            # Sort by date and get intervals
            sorted_occs = sorted(occs, key=lambda x: x[2], reverse=False)  # Sort by date
            
            
            if max_patients == 1:
                # Skip first occurrence, keep rest --> mimics copyforward
                if len(sorted_occs) > 1:
                    nids_with_intervals = [(n, list(i), c) for n, p, d, i, c in sorted_occs[1:]]
            else:
                nids_with_intervals = [(n, list(i), c) for n, p, d, i, c in sorted_occs]
            
            if nids_with_intervals:
                pattern_frequencies[norm] = len(nids_with_intervals)
                pattern_to_notes[norm] = [sorted_occs[0][0], nids_with_intervals] # attach the source note id

        print(f"  Found {len(pattern_to_notes.keys())} unique patterns\n")

        return pattern_to_notes
    
    
    def run(self):
        """Main execution pipeline"""
        
        if not self.notes:
            print("No notes loaded.")
            return {}
        
        # Process templates 
        print("Finding templating information...")
        patterns_to_notes = self._find_frequent_patterns(min_notes=1, min_patients=5)
        
        
        # Format into [nid -> [(start,end)]]
        temp_nid_to_intervals = defaultdict(list)
        for pattern, item in patterns_to_notes.items():
            note_id = item[0]
            note_list = item[1]
            for nid, intervals, chunk in note_list:
                for interval in intervals:
                    temp_nid_to_intervals[nid].append({'src': note_id,
                                                      'start': interval[0],
                                                      'end': interval[1],
                                                      'text': chunk})

        # process copyforward
        print("Finding copyforward information...")
        cf_to_notes = self._find_frequent_patterns(min_notes=2, max_patients=1)
        
        
        # Format into [nid -> [(start,end)]]
        cf_nid_to_intervals = defaultdict(list)
        for pattern, item in cf_to_notes.items():
            note_id = item[0]
            note_list = item[1]
            for nid, intervals, chunk in note_list:
                for interval in intervals:
                    cf_nid_to_intervals [nid].append({'src': note_id,
                                                      'start': interval[0],
                                                      'end': interval[1],
                                                      'text': chunk})
        
        return dict(temp_nid_to_intervals), dict(cf_nid_to_intervals)