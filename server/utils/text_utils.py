"""
Text utilities for handling Arabic text and numbers.
"""

def to_arabic_number(number: int) -> str:
    """
    Convert a number to its Arabic numeral representation.
    
    Args:
        number: Integer to convert
        
    Returns:
        String containing the number in Arabic numerals
        
    Examples:
        >>> to_arabic_number(123)
        '١٢٣'
        >>> to_arabic_number(456)
        '٤٥٦'
    """
    # Arabic-Indic digits (0-9)
    arabic_numbers = ['٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩']
    
    # Convert each digit to its Arabic equivalent
    return ''.join(arabic_numbers[int(d)] for d in str(number)) 