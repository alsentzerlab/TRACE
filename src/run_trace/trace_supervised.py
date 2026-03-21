import json
import ast
import re

from typing import Dict
# Custom
from run_trace.utils import label_spans, generate_highlight_html
# Template pattern for identify_templates
TEMPLATE_PATTERN = (
    r"\[type=[^\]]+\]\s*"
    r"\[index=\d+\]\s*"
    r"\[template_id=\d+\]\s*"
    r"(.*?)"
    r"(?=\n\s*\[type=|\Z)"
)

def process_template_str(template_str):
    """
    Identify the templates after the [type=str] [index=int] [template_id=int] information

    Args:
        template_str (str): list of templates
    
    Returns:
        list[str]: list of templates
    """
    return [t.strip() for t in re.findall(TEMPLATE_PATTERN, template_str, re.DOTALL)]


class SupervisedTRACE:
    """Class for processing and analyzing text traces (templates and copyforward text)."""
    
    def __init__(self, item: Dict, from_processed: bool = False):
        """
        Initialize TRACE object with item data.
        
        Args:
            item: Dictionary containing note data with keys like 'full_note_text',
                  'template_string', 'copyforward', etc.
            from_processed: If True, load from already processed item with existing spans
        """
        self.item = item
        
        if from_processed:
            # Load from already processed item
            self.template_spans = item.get('template_spans', [])
            self.template_string_list = item.get('template_string_list', [])
            self.copyforward_spans = item.get('copyforward_spans', [])
            self.copyforward_string_list = item.get('copyforward_string_list', [])
        else:
            # Initialize empty for new processing
            self.template_spans = []
            self.template_string_list = []
            self.copyforward_spans = []
            self.copyforward_string_list = []
    
    @classmethod
    def from_processed(cls, item: Dict):
        """
        Create a TRACE object from an already processed item.
        
        Args:
            item: Dictionary with existing span data
            
        Returns:
            TRACE object with loaded span data
        """
        return cls(item, from_processed=True)
    
    def has_trace(self) -> bool:
        """Check if the item has any trace data."""
        copyforward = self.item.get('copyforward', '[]')
        if isinstance(copyforward, str):
            copyforward = json.loads(copyforward)
        template_string = self.item.get('template_string', '')
        
        return len(copyforward) > 0 or template_string != ""
    
    def process_templates(self):
        """Process template strings and label spans in the destination text."""
        template_string = self.item.get('template_string', '')
        
        full_note_text = self.item['full_note_text']

        # Parse template string into list of templates
        self.template_string_list = process_template_str(template_string)
        # Label spans for each template
        self.template_spans = []
        for idx, template in enumerate(self.template_string_list):
            spans = label_spans(template, full_note_text, min_length=10, 
                              autojunk=False, caps=False)
            for start, end in spans:
                self.template_spans.append({
                    'text': full_note_text[start:end],
                    'start': start,
                    'end': end,
                    'src': idx
                })
    
    def process_copyforward(self):
        """Process copyforward data and label spans in the destination text."""
        copyforward = self.item.get('copyforward', '[]')
        full_note_text = self.item['full_note_text']
        
        # Parse copyforward if it's a string
        if isinstance(copyforward, str):
            copyforward = ast.literal_eval(copyforward)
        
        # Extract text from copyforward items
        self.copyforward_string_list = [cf['text'] for cf in copyforward]
        
        # Label spans for each copyforward text
        self.copyforward_spans = []
        for idx, cf_text in enumerate(self.copyforward_string_list):
            spans = label_spans(cf_text, full_note_text, min_length=10,
                              autojunk=False, caps=True)
            
            for start, end in spans:
                self.copyforward_spans.append({
                    'text': full_note_text[start:end],
                    'start': start,
                    'end': end,
                    'src': idx
                })
    
    def write_html(self, output_dir: str, base_name: str):
        """
        Generate and write HTML visualization files.
        
        Args:
            output_dir: Directory to write HTML files
            base_name: Base filename (without extension)
        """
        import os
        
        full_note_text = self.item['full_note_text']
        
        # Generate template HTML
        if self.template_string_list:
            html_output = generate_highlight_html(
                full_note_text, 
                self.template_string_list, 
                self.template_spans
            )
            with open(f'{output_dir}/{base_name}_template.html', 'w', 
                     encoding='utf-8') as f:
                f.write(html_output)
        
        # Generate copyforward HTML
        if self.copyforward_string_list:
            html_output = generate_highlight_html(
                full_note_text,
                self.copyforward_string_list,
                self.copyforward_spans
            )
            with open(f'{output_dir}/{base_name}_copyforward.html', 'w',
                     encoding='utf-8') as f:
                f.write(html_output)
    
    def to_string(self) -> str:
        """
        Convert the TRACE object to a JSON string.
        
        Returns:
            JSON string representation of the item with added span data
        """
        output_item = self.item.copy()
        output_item['template_string_list'] = self.template_string_list
        output_item['template_spans'] = self.template_spans
        output_item['copyforward_string_list'] = self.copyforward_string_list
        output_item['copyforward_spans'] = self.copyforward_spans
        
        return json.dumps(output_item)