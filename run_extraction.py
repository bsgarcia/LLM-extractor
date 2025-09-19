#!/usr/bin/env python3
"""
Simple script to run PDF analysis using config.yaml for configuration
"""

import os
import sys
import yaml
import logging
from modules.llm_extractor import LLMExtractor

def load_config(config_file="config/config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        print(f"ERROR: Configuration file '{config_file}' not found")
        print("Please create config.yaml based on config.template.yaml")
        return None
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML in configuration file: {e}")
        return None

def setup_logging(log_level):
    """Setup logging based on configuration"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """Run PDF analysis using configuration file"""
    
    # Load configuration
    config = load_config()
    if not config:
        return 1
    
    # Setup logging
    log_level = config.get('options', {}).get('log_level', 'INFO')
    setup_logging(log_level)
    
    # Get configuration values
    gemini_config = config.get('gemini', {})
    paths_config = config.get('paths', {})
    options_config = config.get('options', {})
    
    # Get API key from config or environment
    api_key = gemini_config.get('api_key') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: No Gemini API key found")
        print("Either set 'api_key' in config.yaml or set GEMINI_API_KEY environment variable")
        print("Example: $env:GEMINI_API_KEY = 'your_api_key_here'")
        return 1
    
    # Get model name
    model_name = gemini_config.get('model', 'gemini-2.0-flash-exp')
    
    # Get file paths
    pdf_dir = paths_config.get('pdf_directory', 'pdfs')
    questions_file = paths_config.get('questions_file', 'config/questions.yaml')
    output_file = paths_config.get('output_file', 'output.md')
    log_file = paths_config.get('log_file', None)  # Get log file from config
    
    # Get options
    confirm_before_processing = options_config.get('confirm_before_processing', True)
    
    print("PDF Analyzer Configuration:")
    print(f"  Model: {model_name}")
    print(f"  PDF Directory: {pdf_dir}")
    print(f"  Questions File: {questions_file}")
    print(f"  Output File: {output_file}")
    print(f"  Log File: {log_file if log_file else 'Console only'}")
    print()
    
    # Check if required files exist
    if not os.path.exists(pdf_dir):
        print(f"ERROR: PDF directory '{pdf_dir}' not found")
        return 1
        
    if not os.path.exists(questions_file):
        print(f"ERROR: Questions file '{questions_file}' not found")
        return 1
    
    # Count PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files to process.")
    
    if len(pdf_files) == 0:
        print("No PDF files found to process")
        return 1
    
    # Ask for confirmation if enabled
    if confirm_before_processing:
        response = input(f"Process {len(pdf_files)} PDF files? This may take a while. (y/N): ")
        if response.lower() != 'y':
            print("Cancelled")
            return 0
    
    try:
        # Create analyzer and run (removed max_text_length parameter)
        print("Starting analysis...")
        analyzer = LLMExtractor(api_key, model_name, log_file=log_file)
        analyzer.process_directory(pdf_dir, questions_file, output_file)
        print(f"Analysis complete! Results saved to: {output_file}")
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())