#!/usr/bin/env python3
"""
PDF Analysis Tool using Gemini AI

This program processes PDF files in a directory, extracts text content,
and uses Gemini AI to answer predefined questions about each document.
Results are saved to a markdown file with organized sections per PDF.
"""

import os
import re
import yaml
import time
import google.generativeai as genai
from pathlib import Path
from typing import List, Dict, Optional
from io import StringIO
import argparse
import logging
from datetime import datetime


class LLMExtractor:
    """Main class for analyzing PDF documents with Gemini AI"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp", log_file: str = None):
        """
        Initialize the PDF analyzer
        
        Args:
            api_key: Google AI API key for Gemini
            model_name: Name of the Gemini model to use
            log_file: Path to log file for output (optional)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.log_file = log_file
        self._setup_logging()
        self._setup_gemini()
    
    def _setup_logging(self):
        """Configure logging for the analyzer"""
        # Create a logger instance for this class
        self.logger = logging.getLogger(__name__)
        
        # Remove any existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Set logging level
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Always add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler if log_file is specified
        if self.log_file:
            # Handle timestamp replacement in log file name
            if '{timestamp}' in self.log_file:
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                actual_log_file = self.log_file.replace('{timestamp}', timestamp)
            else:
                actual_log_file = self.log_file
            
            try:
                file_handler = logging.FileHandler(actual_log_file, encoding='utf-8')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                self.logger.info(f"Logging to file: {actual_log_file}")
            except Exception as e:
                self.logger.warning(f"Failed to setup file logging: {e}")
        
        # Prevent propagation to root logger to avoid duplicate messages
        self.logger.propagate = False
    
    def set_quiet_mode(self, quiet: bool = True):
        """
        Enable or disable quiet mode for logging
        
        Args:
            quiet: If True, only log to file (no console output). If False, log to both console and file.
        """
        # Find and remove/add console handler based on quiet mode
        console_handlers = [h for h in self.logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        
        if quiet:
            # Remove console handlers (keep only file handlers)
            for handler in console_handlers:
                self.logger.removeHandler(handler)
        else:
            # Add console handler if not present
            if not console_handlers and not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in self.logger.handlers):
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
    
    def _format_duration(self, seconds: float) -> str:
        """
        Format duration in seconds to a human-readable string
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{int(minutes)}m {remaining_seconds:.1f}s"
        else:
            hours = seconds // 3600
            remaining_minutes = (seconds % 3600) // 60
            remaining_seconds = seconds % 60
            return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.1f}s"
    
    def _setup_gemini(self):
        """Configure Gemini AI client"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.logger.info(f"Gemini AI configured with model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to configure Gemini AI: {e}")
            raise
    
    def upload_pdf_file(self, pdf_path: str):
        """
        Upload PDF file to Gemini using File API
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Uploaded file object that can be used in generate_content calls
        """
        try:
            pdf_filename = Path(pdf_path).name
            self.logger.info(f"Uploading PDF file: {pdf_filename}")
            upload_start = time.time()
            uploaded_file = genai.upload_file(
                path=pdf_path,
                display_name=pdf_filename
            )
            upload_duration = time.time() - upload_start
            self.logger.info(f"PDF upload completed in {self._format_duration(upload_duration)}")
            self.logger.info(f"Successfully uploaded '{pdf_filename}' as: {uploaded_file.name}")
            return uploaded_file
            
        except Exception as e:
            self.logger.error(f"Failed to upload PDF file {pdf_path}: {e}")
            raise
    
    def load_questions(self, questions_file: str) -> tuple[List[str], str]:
        """
        Load questions and additional instructions from YAML file
        
        Args:
            questions_file: Path to YAML file containing questions
            
        Returns:
            Tuple of (list of questions, additional instructions)
        """
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                questions = data.get('questions', [])
                additional_instructions = data.get('additional_instructions', '')
                self.logger.info(f"Loaded {len(questions)} questions from {questions_file}")
                if additional_instructions:
                    self.logger.info(f"Loaded additional instructions: {additional_instructions[:100]}...")
                return questions, additional_instructions
        except Exception as e:
            self.logger.error(f"Failed to load questions from {questions_file}: {e}")
            raise
    
    def query_gemini(self, uploaded_file, question: str, additional_instructions: str = "") -> str:
        """
        Query Gemini AI with uploaded file and question
        
        Args:
            uploaded_file: Uploaded file object from upload_pdf_file()
            question: Question to ask about the document
            additional_instructions: Additional instructions to append to each question
            
        Returns:
            Gemini's response as string
        """
        start_time = time.time()
        try:
            # Construct the prompt
            base_instructions = "Please provide a concise but comprehensive answer based on the information available in the paper. If the information is not available in the provided document, please state that clearly."
            
            # Add additional instructions if provided
            if additional_instructions:
                full_instructions = f"{base_instructions}\n\nAdditional instructions: {additional_instructions}"
            else:
                full_instructions = base_instructions
            
            prompt = f"""Based on the research paper document, please answer this question: {question}

{full_instructions}"""

            # Use the uploaded file directly with the prompt
            response = self.model.generate_content([uploaded_file, prompt])
            
            end_time = time.time()
            duration = end_time - start_time
            self.logger.info(f"Gemini query completed in {self._format_duration(duration)}")
            
            return response.text.strip()
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            self.logger.error(f"Failed to query Gemini after {self._format_duration(duration)}: {e}")
            return f"Error querying Gemini: {str(e)}"
    
    def get_pdf_title(self, pdf_filename: str) -> str:
        """
        Extract clean title from PDF filename
        
        Args:
            pdf_filename: Name of the PDF file
            
        Returns:
            Clean title for markdown heading
        """
        # Remove .pdf extension
        title = Path(pdf_filename).stem
        
        # Clean up the title (remove extra spaces, normalize)
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title

    def ask_question(self, i: int, n: int, uploaded_file, pdf_name: str, question: str, additional_instructions: str = "") -> str:
        self.logger.info(f"Processing question {i}/{n} for {pdf_name}")
        question_start = time.time()
        answer = self.query_gemini(uploaded_file, question, additional_instructions)
        question_duration = time.time() - question_start
        return answer, question_duration

    def process_pdf(self, pdf_path: str, questions: List[str], additional_instructions: str = "") -> Dict[str, str]:
        """
        Process a single PDF file with all questions using File API
        
        Args:
            pdf_path: Path to PDF file
            questions: List of questions to ask
            additional_instructions: Additional instructions to append to each question
            
        Returns:
            Dictionary mapping questions to answers
        """
        start_time = time.time()
        pdf_name = Path(pdf_path).name
        self.logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Upload PDF file once using File API
            uploaded_file = self.upload_pdf_file(pdf_path)
            
            # Query Gemini for each question using the uploaded file
            results = {}
            total_query_time = 0
            for i, question in enumerate(questions, 1):
                n = len(questions)
                answer, question_duration = self.ask_question(
                    i, n, uploaded_file, pdf_name, question,  additional_instructions)
                total_query_time += question_duration
                results[question] = answer
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            self.logger.info(f"Successfully processed {len(questions)} questions for {pdf_name}")
            self.logger.info(f"Total processing time: {self._format_duration(total_duration)}")
            return results
        except Exception as e:
            end_time = time.time()
            total_duration = end_time - start_time
            self.logger.error(f"Failed to process PDF {pdf_path} after {self._format_duration(total_duration)}: {e}")
            return {}
    
    def update_markdown_file(self, md_path: str, pdf_title: str, qa_results: Dict[str, str]):
        """
        Update markdown file with results for a PDF
        
        Args:
            md_path: Path to markdown output file
            pdf_title: Title of the PDF (used as heading)
            qa_results: Dictionary of questions and answers
        """
        # Read existing content if file exists
        existing_content = ""
        if os.path.exists(md_path):
            with open(md_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        
        # Create new section content
        new_section = f"\n## {pdf_title}\n\n"
        new_section += f"*Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        
        for question, answer in qa_results.items():
            new_section += f"• **{question}**\n\n"
            # Format answer as blockquote to preserve markdown formatting while creating visual separation
            answer_lines = answer.split('\n')
            formatted_answer = '\n'.join(f"> {line}" for line in answer_lines)
            new_section += f"{formatted_answer}\n\n"
        
        # Check if section already exists and replace it
        section_pattern = rf"^## {re.escape(pdf_title)}$.*?(?=^## |\Z)"
        if re.search(section_pattern, existing_content, re.MULTILINE | re.DOTALL):
            # Replace existing section
            new_content = re.sub(section_pattern, new_section.strip(), existing_content, flags=re.MULTILINE | re.DOTALL)
            self.logger.info(f"Updated existing section for {pdf_title}")
        else:
            # Append new section
            new_content = existing_content + new_section
            self.logger.info(f"Added new section for {pdf_title}")
        
        # Write updated content
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    
    def process_directory(self, pdf_dir: str, questions_file: str, output_file: str):
        """
        Process all PDF files in a directory
        
        Args:
            pdf_dir: Directory containing PDF files
            questions_file: Path to YAML file with questions
            output_file: Path to output markdown file
        """
        overall_start_time = time.time()
        self.logger.info(f"Starting PDF analysis process")
        self.logger.info(f"PDF Directory: {pdf_dir}")
        self.logger.info(f"Questions File: {questions_file}")
        self.logger.info(f"Output File: {output_file}")
        
        # Load questions and additional instructions
        questions, additional_instructions = self.load_questions(questions_file)
        if not questions:
            self.logger.error("No questions loaded, exiting")
            return
        
        # Get list of PDF files
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        if not pdf_files:
            self.logger.warning("No PDF files found in directory")
            return
        
        # Initialize output file with header if it doesn't exist
        if not os.path.exists(output_file):
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# Extraction Results\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
        
        # Initialize timing statistics
        processing_times = []
        successful_files = 0
        failed_files = 0
        
        # Process each PDF
        for i, pdf_file in enumerate(pdf_files, 1):
            self.logger.info(f"Processing PDF {i}/{len(pdf_files)}: {pdf_file}")
            
            pdf_path = os.path.join(pdf_dir, pdf_file)
            pdf_title = self.get_pdf_title(pdf_file)
            
            file_start_time = time.time()
            try:
                # Process PDF and get Q&A results
                qa_results = self.process_pdf(pdf_path, questions, additional_instructions)
                
                if qa_results:
                    # Update markdown file
                    self.update_markdown_file(output_file, pdf_title, qa_results)
                    file_duration = time.time() - file_start_time
                    processing_times.append(file_duration)
                    successful_files += 1
                    self.logger.info(f"Successfully processed {pdf_file} in {self._format_duration(file_duration)}")
                else:
                    failed_files += 1
                    self.logger.warning(f"No results generated for {pdf_file}")
                    
            except Exception as e:
                file_duration = time.time() - file_start_time
                failed_files += 1
                self.logger.error(f"Failed to process {pdf_file} after {self._format_duration(file_duration)}: {e}")
                continue
        
        # Calculate and log summary statistics
        overall_duration = time.time() - overall_start_time
        
        self.logger.info("=" * 60)
        self.logger.info("PROCESSING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total files processed: {len(pdf_files)}")
        self.logger.info(f"Successful: {successful_files}")
        self.logger.info(f"Failed: {failed_files}")
        self.logger.info(f"Total processing time: {self._format_duration(overall_duration)}")
        
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            min_time = min(processing_times)
            max_time = max(processing_times)
            self.logger.info(f"Average time per file: {self._format_duration(avg_time)}")
            self.logger.info(f"Fastest file: {self._format_duration(min_time)}")
            self.logger.info(f"Slowest file: {self._format_duration(max_time)}")
            self.logger.info(f"Estimated time per question: {self._format_duration(avg_time/len(questions))}")
        
        self.logger.info("=" * 60)
        self.logger.info("PDF analysis process completed")


def main():
    """Main function with command line interface"""
    # Setup basic logging for the main function
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Analyze PDF files using Gemini AI")
    parser.add_argument("--pdf-dir", default="pdfs", help="Directory containing PDF files")
    parser.add_argument("--questions", default="questions.yaml", help="YAML file with questions")
    parser.add_argument("--output", default="analysis_results.md", help="Output markdown file")
    parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use")
    parser.add_argument("--log-file", help="Log file path (optional)")
    
    args = parser.parse_args()
    
    # Get API key from argument or environment
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("Gemini API key is required. Set GEMINI_API_KEY environment variable or use --api-key argument")
        return
    
    # Convert relative paths to absolute
    pdf_dir = os.path.abspath(args.pdf_dir)
    questions_file = os.path.abspath(args.questions)
    output_file = os.path.abspath(args.output)
    
    # Validate inputs
    if not os.path.exists(pdf_dir):
        logger.error(f"PDF directory not found: {pdf_dir}")
        return
    
    if not os.path.exists(questions_file):
        logger.error(f"Questions file not found: {questions_file}")
        return
    
    # Create analyzer and process files
    try:
        analyzer = LLMExtractor(api_key, args.model, log_file=args.log_file)
        analyzer.process_directory(pdf_dir, questions_file, output_file)
        logger.info(f"Analysis complete. Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")


if __name__ == "__main__":
    main()
