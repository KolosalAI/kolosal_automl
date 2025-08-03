#!/usr/bin/env python3
"""Debug validation methods"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.security.enhanced_security import AdvancedInputValidator

def debug_validation():
    """Debug validation methods"""
    
    validator = AdvancedInputValidator()
    sql_input = "'; DROP TABLE users; --"
    
    print(f"Testing input: {sql_input}")
    
    # Test instance method
    result = validator.validate_input(sql_input)
    print(f"Instance method result: {result}")
    
    # Test class method
    class_result = AdvancedInputValidator.validate_input_classmethod(sql_input)
    print(f"Class method result: {class_result}")

if __name__ == "__main__":
    debug_validation()
