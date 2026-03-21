

import asyncio
import re
import aiohttp
import json
from typing import Union, Dict, Any, Optional

import pandas as pd
import numpy as np

SEEDS = [1,2,3,4,42]



# Common headers
headers = {
    "Ocp-Apim-Subscription-Key": # your key here,
    "Content-Type": "application/json",
}

# add your URLs
gemini_url = ""
gpt_url = "" 
claude_url = ""

CONTEXT_SIZE = 500000*4
NOTE_PATTERN = re.compile(
    r"(Note ID: .*?\n.*?\n)",
    re.DOTALL
)

def chunk_notes(text, chunk_size=CONTEXT_SIZE):
    """
    Split text into chunks <= chunk_size without splitting notes.
    """

    notes = NOTE_PATTERN.findall(text)

    chunks = []
    current_chunk = ""

    for note in notes:
        # If adding this note would exceed the chunk size
        if current_chunk and len(current_chunk) + len(note) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = note
        else:
            current_chunk += note


    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def combine_spans(interval_list, dest_text, REMOVE_THRESHOLD=50, desc=None):
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


def tag_text(text, intervals):
    """Insert tags around specified intervals using list concatenation."""
    if not intervals:
        return text
    
    # Create events for opening and closing tags
    events = []
    for start, end, label in intervals:
        events.append((start, 'open', label))
        events.append((end, 'close', label))
    
    # Sort events by position
    events.sort(key=lambda x: (x[0], x[1] == 'open'))
    
    # Build result using list (more efficient than string concatenation)
    result = []
    prev_pos = 0
    
    for pos, event_type, label in events:
        result.append(text[prev_pos:pos])
        result.append(f"<{label}>" if event_type == 'open' else f"</{label}>")
        prev_pos = pos
    
    result.append(text[prev_pos:])
    return ''.join(result)

def assign_positions(span_lengths, note_length, rng):
    """
    Assign random positions to spans ensuring no overlaps.
    
    Parameters:
    - span_lengths: list of span lengths (e.g., [150, 30, 200])
    - note_length: total length of the note
    
    Returns:
    - List of [start, end] positions, or None if spans don't fit
    """
    total_span_length = sum(span_lengths)
    
    # remove entire note
    if total_span_length > note_length:
        return [[0,note_length-1]]
    
    # Calculate available space for gaps
    available_space = note_length - total_span_length
    
    # Generate random gaps (including before first and after last span)
    num_gaps = len(span_lengths) + 1
    gaps = sorted(rng.choice(range(available_space + 1), num_gaps - 1))
    
    # Convert to gap sizes
    gap_sizes = []
    gap_sizes.append(gaps[0] if gaps else 0)
    for i in range(1, len(gaps)):
        gap_sizes.append(gaps[i] - gaps[i-1])
    gap_sizes.append(available_space - (gaps[-1] if gaps else 0))
    
    # Place spans
    spans = []
    position = gap_sizes[0]
    
    for i, length in enumerate(span_lengths):
        start = position
        end = start + length
        spans.append([start, end])
        position = end + gap_sizes[i + 1]
    
    return spans

def random_spans(full_note_text, processed_spans, total, seed):
    """
    Generate random spans that maintain realistic distributions:
    - Proportion of notes with/without spans
    - Number of spans per note (when present)
    - Length of individual spans
    - No overlapping spans
    
    Parameters:
    - full_note_text: Series/list of note texts
    - processed_spans: Series/list of span intervals
    
    Returns:
    - List of random spans matching the input distributions
    """
    rng = np.random.default_rng(seed)
    # Analyze the real data
    has_spans = [len(spans) > 0 for spans in processed_spans]
    proportion_with_spans = np.mean(has_spans)
    # Get span counts for notes that have spans
    span_counts = [len(spans) for spans in processed_spans if spans]
    
    
    # Get all span lengths
    span_lengths = []
    for spans in processed_spans:
        if spans:
            for start, end, label in spans:
                span_lengths.append(end - start)
    span_lengths = np.array(span_lengths)
    span_counts = np.array(span_counts)
    
    # Generate random spans
    random_spans = {idx: [] for idx, _ in enumerate(full_note_text)}
    current = 0
    # while current < total-np.mean(span_lengths): # adding buffer
    for idx, note_text in enumerate(full_note_text):
        note_len = len(note_text)

        # Decide if this note should have spans
        if rng.random() > proportion_with_spans:
            continue

        # Sample number of spans from the distribution
        num_spans = rng.choice(span_counts)

        # Generate non-overlapping spans
        span_length_list = []

        for n in range(num_spans):
            span_length_list.append(rng.choice(span_lengths))
            if np.sum(span_length_list) > note_len:
                continue
        new = assign_positions(span_length_list, note_len, rng)
        note_spans = combine_spans(new, note_text)
        # Update the span list
        random_spans[idx] = note_spans
        for span in note_spans:
            current += (span[1]-span[0])
        if current > (total-np.mean(span_lengths)):
            break
    # return the order of the output
    random_span_list = []
    for key in range(len(full_note_text)):
        random_span_list.append([[item[0], item[1], 'R'] for item in random_spans[idx]])
    return random_span_list


def get_all_spans(row):
    """Remove text within specified intervals."""
    text = row['full_note_text']
    intervals = row['processed_spans']
    if not intervals:
        return []
    
    # Sort and merge overlapping intervals
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for start, end, label in sorted_intervals[1:]:
        last_start, last_end, last_label = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end), last_label)
        else:
            merged.append((start, end, label))
    return merged




def remove_text(text, intervals, remove_labels = None):
    """Remove text within specified intervals."""
    if not intervals:
        return text
    
    # Sort and merge overlapping intervals
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for start, end, label in sorted_intervals[1:]:
        if remove_labels != None:
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
def process_patient_notes(patient_df):
    """Process all notes for a single patient efficiently."""

    # Get list of all copyforward notes
    patient_df['parsed'] =  patient_df['copyforward'].apply(parse_json_list)
    copyforward_df = patient_df.explode('parsed')
    mask = copyforward_df['parsed'].notna()
    copyforward_df = pd.json_normalize(copyforward_df[mask]['parsed']).rename(columns={'src_note_csn': 'note_csn_id', 'text': 'full_note_text'})
    
    if copyforward_df.shape[0] != 0: # process copyforward df if there are copyforward entries
        copyforward_df['processed_spans'] = [[] for _ in range(len(copyforward_df))]
        copyforward_df['pat_mrn_id'] = patient_df.iloc[0]['pat_mrn_id']
        copyforward_df["upd_aut_local_dttm"] = pd.to_datetime(copyforward_df["upd_aut_local_dttm"])

        # Concatenate to patient_df and drop duplicates
        patient_df = pd.concat([patient_df, copyforward_df]) 
    patient_df = patient_df.drop_duplicates(subset='note_csn_id', keep='first') # this will keep the trace version, since copyforward is appended at end
    patient_df['upd_aut_local_dttm']  = pd.to_datetime(patient_df["upd_aut_local_dttm"])
    # Sort by date (most to least recent)
    patient_df = patient_df.sort_values(by="upd_aut_local_dttm", ascending=True)
    
    
    
    
    # # Pre-allocate lists for concatenation
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
        
    ret = {
        'pat_mrn_id': int(patient_df.iloc[0]['pat_mrn_id']),
        'original': '\n'.join(original_parts),
        'removed': '\n'.join(removed_parts),
        'removed_template':  '\n'.join(removedT_parts),
        'removed_copy':  '\n'.join(removedC_parts),
        'removed_stage1':  '\n'.join(removed1_parts),
        'removed_stage2':  '\n'.join(removed2_parts),
    }
    # get random spans
    patient_df['all_spans'] = patient_df.apply(get_all_spans, axis=1)
    
    for idx, seed in enumerate(SEEDS):
        patient_df['random_spans'] = random_spans(patient_df['full_note_text'], patient_df['all_spans'], len('\n'.join(original_parts)) - len('\n'.join(removed_parts)), seed )

        random_parts = []
        for _, row in patient_df.iterrows():
            random = row['random_spans']
            random_parts.append(note_header + remove_text(text, random))
        ret[f'random_{idx}'] = '\n'.join(random_parts)
        # Join with newlines once per patient
    return ret


def build_payload(system_prompt: str, user_prompt: str, model_id: str) -> Dict[str, Any]:
    """
    Build request payload based on model type.

    Args:
        system_prompt: System instructions
        user_prompt: Text prompt
        model_id: Model identifier

    Returns:
        dict: Request payload
    """
    if "gpt" in model_id:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        return {
            "model": "gpt-5",
            "messages": messages,
        }

    # Anthropic Claude models (via AWS Bedrock)
    elif "claude" in model_id.lower() or model_id.startswith("anthropic.") or model_id.startswith("arn:aws:bedrock"):
        # Claude uses system parameter separately from messages
        payload = {
            "model_id": "arn:aws:bedrock:us-west-2:679683451337:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "prompt_text": user_prompt
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    # Google Gemini models
    else:
        # Gemini uses systemInstruction for system prompts
        payload = {
                "generationConfig": {
                "maxOutputTokens": 65535,
                },
                "contents" :[{
                    "role": "user",
                    "parts": [
                        {"text": user_prompt}
                    ]
                }]
        }
        if system_prompt:
            payload['system_instruction'] = {
                                            "parts": [
                                                {"text": system_prompt}
                                            ]
                                        }
        return payload


def parse_response(response_json: Dict[str, Any], model_id: str) -> str:
    """
    Parse response based on model type.

    Args:
        response_json: JSON response from API
        model_id: Model identifier

    Returns:
        str: Extracted text content
    """
    # OpenAI models (GPT-4o, GPT-4.1)
    if "gpt" in model_id:
        return response_json["choices"][0]["message"]["content"]

    # Anthropic Claude models
    elif "claude" in model_id.lower() or model_id.startswith("anthropic.") or model_id.startswith("arn:aws:bedrock"):
        return response_json["content"][0]["text"]

    # Google Gemini models
    else:
        content = ""
        for i in response_json:
            content += i['candidates'][0]['content']['parts'][0]['text']
        return content
    
async def get_question(messages: str, model_id: str):
    """
    Helper function to send current conversation to API and get the response (async version).
    
    Args:
        messages: JSON string of the payload
        model_id: Model identifier ('gpt', 'claude', or 'gemini')
    
    Returns:
        str: Parsed response text
    """
    if model_id == "gpt":
        url = gpt_url
    elif model_id == 'claude':
        url = claude_url
    else:
        url = gemini_url
    
    response_json = await post_with_retry(url=url, headers=headers, payload=messages)
    return parse_response(response_json, model_id=model_id)

async def post_with_retry(url, headers, payload, timeout=300, max_retries=8, backoff_factor=1.4):
    """
    Send POST request with retries and exponential backoff (async version).

    Args:
        url (str): endpoint URL
        headers (dict): request headers
        payload (dict/str): JSON or string payload
        timeout (int): request timeout in seconds
        max_retries (int): maximum retry attempts
        backoff_factor (float): multiplier for wait time between retries

    Returns:
        dict: JSON response
    """
    async with aiohttp.ClientSession() as session:
        for attempt in range(max_retries):
            try:
                async with session.post(
                    url, 
                    headers=headers, 
                    data=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 429:  # rate limit
                        raise aiohttp.ClientError("Rate limit hit")
                    response.raise_for_status()
                    return await response.json()
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == max_retries - 1:
                    # Last attempt, re-raise
                    raise RuntimeError(f"POST failed after {max_retries} retries: {e}")
                
                sleep_time = backoff_factor ** attempt
                print(f"Retry {attempt + 1}/{max_retries} after {sleep_time:.1f}s due to: {e}")
                await asyncio.sleep(sleep_time)

    raise RuntimeError(f"POST failed after {max_retries} retries")
    

async def send_single_message(user_prompt: str, 
                               system_instructions: Union[str, None] = None,
                               model_id: str ='gemini'):
    """
    Send a single message to the API (async version).
    
    Args:
        user_prompt: User's prompt text
        system_instructions: Optional system instructions
        model_id: Model to use ('gpt', 'claude', or 'gemini')
    
    Returns:
        str: Model's response
    """
    payload = build_payload(system_instructions, user_prompt, model_id)
    return await get_question(messages=json.dumps(payload), model_id=model_id)

def safe_json_parse(response_text):
    """
    Robustly parse JSON from LLM response.
    Handles various wrapper formats.
    """
    # Try direct parse first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from code blocks
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try removing common prefixes/suffixes
    patterns = [
        (r"^json\s*", r"\s*$"),  # Remove json wrapper
        (r"^```json\s*", r"\s*```$"),  # Remove markdown code block
    ]
    
    for prefix, suffix in patterns:
        cleaned = re.sub(prefix, "", response_text)
        cleaned = re.sub(suffix, "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            continue
    
    raise json.JSONDecodeError(f"Could not parse JSON from response", response_text, 0)


# recover from parsing
def parse_llm_json(response: str, expected_fields: list, 
                   validate_booleans: bool = False) -> Dict[str, Any]:
    """
    Robustly parse JSON from LLM responses that may contain formatting issues.
    
    Args:
        response: The raw response string from the LLM
        expected_fields: Optional list of expected field names for validation
        validate_booleans: If True, ensures all field values are booleans
        
    Returns:
        Parsed JSON as a dictionary with validated schema
        
    Raises:
        ValueError: If JSON cannot be parsed or doesn't match expected schema
    """
    
    # Strategy 1: Try standard JSON parsing first
    try:
        result = json.loads(response)
        if expected_fields:
            validate_schema(result, expected_fields, validate_booleans)
        return result
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 2: Extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            if expected_fields:
                validate_schema(result, expected_fields, validate_booleans)
            return result
        except (json.JSONDecodeError, ValueError):
            response = json_match.group(1)  # Continue with extracted content
    
    # Strategy 3: Find the last complete JSON object in the response
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            if expected_fields:
                validate_schema(result, expected_fields, validate_booleans)
            return result
        except (json.JSONDecodeError, ValueError):
            response = json_match.group(0)
    
    # Strategy 4: Clean common formatting issues
    cleaned = clean_json_string(response)
    try:
        result = json.loads(cleaned)
        if expected_fields:
            validate_schema(result, expected_fields, validate_booleans)
        return result
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 5: Fix common LLM mistakes and retry
    fixed = fix_common_llm_errors(cleaned)
    try:
        result = json.loads(fixed)
        if expected_fields:
            validate_schema(result, expected_fields, validate_booleans)
        return result
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 6: Line-by-line field extraction (last resort)
    if expected_fields:
        result = extract_fields_line_by_line(response, expected_fields)
        if result:
            # Validate before returning
            validate_schema(result, expected_fields, validate_booleans)
            return result
    
    raise ValueError(f"Could not parse JSON after all recovery attempts. Response: {response[:500]}")


def validate_schema(data: Dict[str, Any], expected_fields: list, 
                    validate_booleans: bool = False) -> None:
    """
    Validate that parsed JSON matches expected schema.
    
    Args:
        data: The parsed dictionary to validate
        expected_fields: List of required field names
        validate_booleans: If True, ensures all values are booleans
        
    Raises:
        ValueError: If validation fails
    """
    # Check all expected fields are present
    missing_fields = set(expected_fields) - set(data.keys())
    if missing_fields:
        raise ValueError(f"Missing required fields: {sorted(missing_fields)}")
    
    # Check no extra fields are present
    extra_fields = set(data.keys()) - set(expected_fields)
    if extra_fields:
        raise ValueError(f"Unexpected fields found: {sorted(extra_fields)}")
    
    # Validate all values are booleans if required
    if validate_booleans:
        non_boolean_fields = {
            field: type(value).__name__ 
            for field, value in data.items() 
            if not isinstance(value, bool)
        }
        if non_boolean_fields:
            raise ValueError(
                f"All fields must be boolean. Found non-boolean types: {non_boolean_fields}"
            )


def clean_json_string(text: str) -> str:
    """Remove common artifacts that break JSON parsing."""
    # Remove markdown code fences
    text = re.sub(r'```(?:json)?', '', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Find JSON object boundaries
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end+1]
    
    return text


def fix_common_llm_errors(text: str) -> str:
    """Fix common JSON formatting errors made by LLMs."""
    
    # Fix 1: Remove text before field names (e.g., 'cha"hypertension"' -> '"hypertension"')
    text = re.sub(r'[a-zA-Z_]+("[\w_]+"\s*:)', r'\1', text)
    
    # Fix 2: Remove text after field names (e.g., '"cryptogenic_cirrhosis"rolling_update:' -> '"cryptogenic_cirrhosis":')
    text = re.sub(r'("[^"]+")[\w\s]+(:)', r'\1\2', text)
    
    # Fix 3: Fix missing quotes around field names (e.g., 'hypertension": false' -> '"hypertension": false')
    text = re.sub(r'([,{]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    
    # Fix 4: Remove duplicate field indicators (e.g., '"field": "field": value' -> '"field": value')
    text = re.sub(r'("[^"]+"\s*:\s*)"\1', r'\1', text)
    
    # Fix 5: Fix trailing commas before closing braces
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix 6: Remove text/garbage within field values
    # Look for patterns like: "field": text"value"
    text = re.sub(r':\s*[a-zA-Z_]+\s*"([^"]+)"', r': "\1"', text)
    
    # Fix 7: Remove standalone text between fields
    # Pattern: }, standalone_text "field":
    text = re.sub(r'(,\s*)[a-zA-Z_]+\s*("[\w_]+"\s*:)', r'\1\2', text)
    
    # Fix 8: Fix incomplete entries at the end
    # If the last field doesn't have a value, try to complete it
    text = re.sub(r'("[^"]+"\s*:\s*)$', r'\1false', text)
    
    # Fix 9: Remove newlines within string values that aren't escaped
    lines = text.split('\n')
    fixed_lines = []
    in_string = False
    for line in lines:
        # Simple heuristic: if line doesn't contain : or ends with comma/brace, might be continuation
        stripped = line.strip()
        if in_string or (stripped and not ':' in stripped and not stripped.endswith(('{', '}', ','))):
            if fixed_lines:
                fixed_lines[-1] += ' ' + stripped
        else:
            fixed_lines.append(line)
    text = '\n'.join(fixed_lines)
    
    return text


def extract_fields_line_by_line(text: str, expected_fields: list) -> Optional[Dict[str, Any]]:
    """
    Last resort: extract field values line by line using regex.
    Works even when JSON structure is broken.
    Only returns boolean values for the expected fields.
    """
    result = {}
    
    for field in expected_fields:
        # Look for patterns like: "field": value or field: value
        patterns = [
            rf'"{field}"\s*:\s*(true|false)',
            rf'{field}\s*:\s*(true|false)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value_str = match.group(1).strip().lower()
                result[field] = value_str == 'true'
                break
    
    # Only return if we found ALL expected fields with boolean values
    return result if len(result) == len(expected_fields) else None
