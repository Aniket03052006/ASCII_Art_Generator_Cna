"""
ASCII Art Result Container

Provides the ASCIIResult dataclass for storing and displaying results,
with support for terminal display, HTML export, and file saving.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime
from PIL import Image
import html


@dataclass
class ASCIIResult:
    """
    Container for ASCII art generation results.
    
    Attributes:
        text: The generated ASCII art string
        source_image: The source image used for generation
        metadata: Generation parameters and metrics
    """
    text: str
    source_image: Optional[Image.Image] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def width(self) -> int:
        """Width in characters."""
        lines = self.text.split('\n')
        return max(len(line) for line in lines) if lines else 0
    
    @property
    def height(self) -> int:
        """Height in lines."""
        return len(self.text.split('\n'))
    
    def display(self, max_width: Optional[int] = None):
        """
        Print the ASCII art to the terminal.
        
        Args:
            max_width: Maximum width to display (truncates if needed)
        """
        if max_width:
            lines = self.text.split('\n')
            for line in lines:
                print(line[:max_width])
        else:
            print(self.text)
    
    def save(self, path: str, format: str = "auto"):
        """
        Save ASCII art to a file.
        
        Args:
            path: Output file path
            format: "txt", "html", or "auto" (detect from extension)
        """
        if format == "auto":
            if path.endswith('.html') or path.endswith('.htm'):
                format = "html"
            else:
                format = "txt"
        
        if format == "html":
            content = self.to_html()
        else:
            content = self.text
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def to_html(
        self,
        font_family: str = "Menlo, Monaco, 'Courier New', monospace",
        font_size: str = "10px",
        bg_color: str = "#1e1e1e",
        fg_color: str = "#d4d4d4",
        title: str = "ASCII Art",
    ) -> str:
        """
        Convert ASCII art to a styled HTML page.
        
        Args:
            font_family: CSS font family
            font_size: CSS font size
            bg_color: Background color
            fg_color: Text color
            title: HTML page title
            
        Returns:
            Complete HTML document string
        """
        escaped_text = html.escape(self.text)
        
        # Generate metadata section
        meta_html = ""
        if self.metadata:
            meta_items = []
            for key, value in self.metadata.items():
                if key != 'source_image':
                    meta_items.append(f"<li><strong>{key}:</strong> {value}</li>")
            if meta_items:
                meta_html = f"""
        <div class="metadata">
            <h3>Generation Details</h3>
            <ul>{''.join(meta_items)}</ul>
        </div>
"""
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            background-color: {bg_color};
            color: {fg_color};
            font-family: {font_family};
            font-size: {font_size};
            line-height: 1.2;
            padding: 20px;
            margin: 0;
        }}
        pre {{
            margin: 0;
            white-space: pre;
            overflow-x: auto;
        }}
        .container {{
            max-width: 100%;
            overflow-x: auto;
        }}
        .metadata {{
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #444;
            font-size: 12px;
        }}
        .metadata ul {{
            list-style: none;
            padding: 0;
        }}
        .metadata li {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <pre>{escaped_text}</pre>
{meta_html}
    </div>
</body>
</html>"""
    
    def to_ansi(self) -> str:
        """
        Convert to ANSI-colored terminal output.
        
        Currently returns plain text; can be extended for color support.
        """
        # TODO: Add ANSI color codes for block characters
        return self.text
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the ASCII art."""
        lines = self.text.split('\n')
        char_count = sum(len(line) for line in lines)
        unique_chars = set(self.text.replace('\n', ''))
        
        return {
            'width': self.width,
            'height': self.height,
            'total_characters': char_count,
            'unique_characters': len(unique_chars),
            'lines': len(lines),
        }
    
    def __repr__(self) -> str:
        return f"ASCIIResult(width={self.width}, height={self.height})"
    
    def __str__(self) -> str:
        return self.text


def create_result(
    text: str,
    source_image: Optional[Image.Image] = None,
    prompt: Optional[str] = None,
    mapper: Optional[str] = None,
    charset: Optional[str] = None,
    **extra_metadata
) -> ASCIIResult:
    """
    Factory function to create an ASCIIResult with standard metadata.
    
    Args:
        text: ASCII art string
        source_image: Source image
        prompt: Generation prompt (if from SD)
        mapper: Mapper used ("aiss" or "random_forest")
        charset: Character set used
        **extra_metadata: Additional metadata
        
    Returns:
        Configured ASCIIResult
    """
    metadata = {
        'generated_at': datetime.now().isoformat(),
    }
    
    if prompt:
        metadata['prompt'] = prompt
    if mapper:
        metadata['mapper'] = mapper
    if charset:
        metadata['charset'] = charset
    
    metadata.update(extra_metadata)
    
    return ASCIIResult(
        text=text,
        source_image=source_image,
        metadata=metadata
    )
