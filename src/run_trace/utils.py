import difflib

def label_spans(src_text, dest_text, min_length=10, autojunk=False, caps=False):
    if caps == False:
        matcher = difflib.SequenceMatcher(None, src_text.lower(), dest_text.lower(), autojunk=autojunk)
    else:
        matcher = difflib.SequenceMatcher(None, src_text, dest_text, autojunk=autojunk)
    opcodes = matcher.get_opcodes()
    
    copied_spans = []
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal' and j2-j1 > min_length:
            # This is matching text - it was copied
            copied_text = dest_text[j1:j2]
            copied_spans.append((j1, j2))
    return copied_spans

def combine_spans(interval_list, dest_text, REMOVE_THRESHOLD=200, desc=None):
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



def remove_spans_from_text(text, spans):
    """Remove specified spans from text."""
    if not spans:
        return text
    span_tuples = [(span['start'], span['end']) for span in spans]
    merged_spans = combine_spans(interval_list=span_tuples,
                                 dest_text=text)
    result = text
    for start, end in reversed(merged_spans):
        start = max(0, min(start, len(result)))
        end = max(start, min(end, len(result)))
        result = result[:start] + result[end:]
    return result


def generate_highlight_html(dest_text, src_texts, mappings, supervised_spans=None):
    """
    Generate HTML to visualize text mappings between destination and source texts.
    
    Args:
        dest_text: The destination text string
        src_texts: List of source text strings
        mappings: List of dicts with keys: 'text', 'start', 'end', 'src'
        supervised_spans: Optional list of supervised span dicts with 'start', 'end' keys
    
    Returns:
        HTML string with highlighted text
    """
    # Color palette for different sources
    colors = [
        '#FFE5E5', '#E5F3FF', '#E5FFE5', '#FFF5E5', '#F5E5FF',
        '#FFE5F5', '#E5FFFF', '#FFFFE5', '#F0F0F0', '#FFE5CC'
    ]
    # Color for supervised spans (different from unsupervised)
    supervised_color = '#FFD700'  # Gold color for supervised TRACE
    
    # Create segments to handle overlapping mappings
    # Collect all boundary points
    boundaries = set([0, len(dest_text)])
    for mapping in mappings:
        boundaries.add(mapping['start'])
        boundaries.add(mapping['end'])
    
    # Add supervised span boundaries
    if supervised_spans:
        for span in supervised_spans:
            boundaries.add(span['start'])
            boundaries.add(span['end'])
    
    boundaries = sorted(boundaries)
    
    # For each segment, determine which sources apply and if it's supervised
    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        
        # Find all mappings that cover this segment
        covering_sources = []
        for mapping in mappings:
            if mapping['start'] <= start and mapping['end'] >= end:
                covering_sources.append(mapping['src'])
        
        # Check if this segment is covered by supervised spans
        is_supervised = False
        if supervised_spans:
            for span in supervised_spans:
                if span['start'] <= start and span['end'] >= end:
                    is_supervised = True
                    break
        
        segments.append({
            'start': start,
            'end': end,
            'text': dest_text[start:end],
            'sources': covering_sources,
            'is_supervised': is_supervised
        })
    
    # Build HTML from segments
    dest_html = []
    for segment in segments:
        if segment['is_supervised']:
            # Supervised span takes priority - use gold color with border
            dest_html.append(f'<mark class="supervised-span" style="background-color: {supervised_color}; border: 2px solid #FFA500;">{segment["text"]}</mark>')
        elif not segment['sources']:
            # No highlighting
            dest_html.append(segment['text'])
        elif len(segment['sources']) == 1:
            # Single source
            src_idx = segment['sources'][0]
            color = colors[src_idx % len(colors)]
            dest_html.append(f'<mark style="background-color: {color};">{segment["text"]}</mark>')
        else:
            # Multiple sources - create layered effect with gradient
            src_colors = [colors[src % len(colors)] for src in segment['sources']]
            src_labels = ', '.join([str(src + 1) for src in segment['sources']])
            gradient = f"linear-gradient(to right, {', '.join(src_colors)})"
            dest_html.append(f'<mark style="background: {gradient};" title="Sources: {src_labels}">{segment["text"]}</mark>')
    
    # Build source texts with highlights
    src_html_parts = []
    for src_idx, src_text in enumerate(src_texts):
        color = colors[src_idx % len(colors)]
        
        # Find all mappings for this source
        src_mappings = [m for m in mappings if m['src'] == src_idx]
        
        # Get the substrings that were used
        highlighted_parts = []
        last_pos = 0
        
        for mapping in sorted(src_mappings, key=lambda x: src_text.find(mapping['text'])):
            substring = mapping['text']
            pos = src_text.find(substring, last_pos)
            
            if pos >= 0:
                # Add text before highlight
                if pos > last_pos:
                    highlighted_parts.append(src_text[last_pos:pos])
                
                # Add highlighted text
                highlighted_parts.append(f'<mark style="background-color: {color}; border-bottom: 2px solid #666;">{substring}</mark>')
                last_pos = pos + len(substring)
        
        # Add remaining text
        if last_pos < len(src_text):
            highlighted_parts.append(src_text[last_pos:])
        
        src_html_parts.append(f'<div class="src-text"><strong>Source {src_idx + 1}:</strong> {"".join(highlighted_parts)}</div>')
    
    # Add legend if supervised spans exist
    legend_html = ""
    if supervised_spans:
        legend_html = """
        <div style="position: fixed; top: 20px; right: 20px; background: white; padding: 15px 20px; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); z-index: 1000;">
            <div style="font-size: 16px; font-weight: 600; margin-bottom: 8px;">Legend</div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <mark class="supervised-span" style="background-color: #FFD700; border: 2px solid #FFA500; padding: 3px 8px; border-radius: 2px;"></mark>
                <span style="font-size: 14px;">Supervised TRACE</span>
            </div>
        </div>
        """
    
    # Combine into final HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        .dest-text {{
            background: #fafafa;
            padding: 20px;
            border-radius: 4px;
            margin-bottom: 30px;
            border-left: 3px solid #333;
        }}
        .src-container {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .src-text {{
            background: #fafafa;
            padding: 15px;
            border-radius: 4px;
            border-left: 3px solid #999;
        }}
        mark {{
            padding: 2px 0;
            border-radius: 2px;
        }}
        mark[title] {{
            cursor: help;
            border-bottom: 2px dotted rgba(0,0,0,0.3);
        }}
        .supervised-span {{
            padding: 2px 0;
            border-radius: 2px;
        }}
        h2 {{
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #666;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    {legend_html}
    <h2>Destination Text</h2>
    <div class="dest-text">{''.join(dest_html)}</div>
    
    <h2>Source Texts</h2>
    <div class="src-container">
        {''.join(src_html_parts)}
    </div>
</body>
</html>"""
    
    return html
