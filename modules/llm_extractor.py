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
import pandas as pd


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
    
    def query_gemini(self, uploaded_file, question: str, additional_instructions: str = "") -> Dict[str, str]:
        """
        Query Gemini AI with uploaded file and question using structured JSON output
        
        Args:
            uploaded_file: Uploaded file object from upload_pdf_file()
            question: Question to ask about the document
            additional_instructions: Additional instructions to append to each question
            
        Returns:
            Dictionary with 'short_answer', 'long_answer', and 'quote' keys
        """
        start_time = time.time()
        try:
            # Construct the prompt for structured output
            base_instructions = "Please provide a structured response with three components based on the information available in the paper."
            
            # Add additional instructions if provided
            if additional_instructions:
                full_instructions = f"{base_instructions}\n\n{additional_instructions}"
            else:
                full_instructions = base_instructions
            
            prompt = f"""Based on the research paper document, please answer this question: {question}

            {full_instructions}

            IMPORTANT FORMATTING INSTRUCTIONS:
            - For statistical data (means, standard deviations, sample sizes), format them clearly with proper spacing: e.g., "M = 45.2, SD = 12.3, n = 120"
            - For task results, organize them with bullet points and highlight task names
            - For group comparisons, use bullet points to separate different groups or conditions
            - When reporting multiple statistical values, organize them in a clear, readable format
            - Always include units and context for numerical values

            Please return your response as a JSON object with exactly these three keys:
            - "short_answer": Brief answer (1-2 sentences max)
            - "long_answer": Detailed explanation with context and properly formatted statistics
            - "quote": Direct quote from the paper with page/section reference, or "NA" if not available"""

            # Use structured JSON response
            response = self.model.generate_content(
                [uploaded_file, prompt],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "short_answer": {"type": "string"},
                            "long_answer": {"type": "string"},
                            "quote": {"type": "string"}
                        },
                        "required": ["short_answer", "long_answer", "quote"]
                    }
                )
            )
            
            end_time = time.time()
            duration = end_time - start_time
            self.logger.info(f"Gemini structured query completed in {self._format_duration(duration)}")
            
            # Parse JSON response
            try:
                import json
                result = json.loads(response.text)
                
                # Validate that we have all required keys
                if all(key in result for key in ["short_answer", "long_answer", "quote"]):
                    return result
                else:
                    self.logger.warning("Response missing required keys, using fallback")
                    return self._create_fallback_response(response.text)
                    
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse JSON response: {e}, using fallback")
                return self._create_fallback_response(response.text)
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            self.logger.error(f"Failed to query Gemini after {self._format_duration(duration)}: {e}")
            return {
                "short_answer": f"Error querying Gemini: {str(e)}",
                "long_answer": f"Error querying Gemini: {str(e)}",
                "quote": "NA"
            }
    
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

    def _create_fallback_response(self, raw_text: str) -> Dict[str, str]:
        """
        Create a fallback structured response from raw text when JSON parsing fails
        
        Args:
            raw_text: Raw response text from Gemini
            
        Returns:
            Dictionary with structured response format
        """
        # Try to extract parts using patterns from additional_instructions format
        short_answer = "NA"
        long_answer = raw_text
        quote = "NA"
        
        # Look for the format specified in additional_instructions
        lines = raw_text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if line_lower.startswith('short answer:'):
                short_answer = line[len('short answer:'):].strip()
            elif line_lower.startswith('quote:'):
                quote = line[len('quote:'):].strip()
            elif line_lower.startswith('long answer:'):
                # Take everything from this line onwards as long answer
                long_answer = '\n'.join(lines[i:]).replace('long answer:', '', 1).strip()
                break
        
        return {
            "short_answer": short_answer,
            "long_answer": long_answer,
            "quote": quote
        }

    def generate_column_names_from_questions(self, questions: List[str]) -> List[str]:
        """
        Generate proper Excel column names from research questions using Gemini AI with structured output
        
        Args:
            questions: List of research questions
            
        Returns:
            List of column names suitable for Excel
        """
        try:
            # Create a prompt to generate column names
            prompt = f"""
            I need to create Excel column names from these research questions. The column names should be:
            - Short and concise (ideally under 25 characters)
            - Clear and descriptive
            - Suitable for Excel column headers
            - Professional and standardized
            - Use underscores instead of spaces
            - Use title case (e.g., "Sample_Size", "Training_Duration")

            Convert each of these questions into appropriate column names:

            {chr(10).join([f"{i+1}. {q}" for i, q in enumerate(questions)])}

            Return a JSON array of column name strings, one for each question in the same order.
            """

            # Use structured JSON response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "array",
                        "items": {"type": "string"}
                    }
                )
            )
            
            # Parse the JSON response
            import json
            column_names = json.loads(response.text)
            
            if isinstance(column_names, list) and len(column_names) == len(questions):
                self.logger.info(f"Generated {len(column_names)} column names using Gemini AI with structured output")
                return column_names
            else:
                self.logger.warning("Invalid column names format or length mismatch, using fallback")
                return self._create_fallback_column_names(questions)
                
        except Exception as e:
            self.logger.error(f"Failed to generate column names with Gemini structured output: {e}")
            return self._create_fallback_column_names(questions)


    def extract_short_answer(self, structured_answer) -> str:
        """
        Extract the short answer from structured response for Excel export
        
        Args:
            structured_answer: Dictionary with structured response or legacy string
            
        Returns:
            Short answer string
        """
        # Handle both structured dict and legacy string formats
        if isinstance(structured_answer, dict):
            return structured_answer.get('short_answer', 'NA')
        else:
            # Legacy format - try to extract short answer from string
            full_answer = str(structured_answer)
            lines = full_answer.split('\n')
            for line in lines:
                line = line.strip()
                if line.lower().startswith('short answer:'):
                    return line[len('short answer:'):].strip()
            
            # If no "short answer:" found, return the first sentence or first 100 chars
            first_line = full_answer.split('\n')[0].strip()
            if len(first_line) <= 100:
                return first_line
            else:
                return first_line[:97] + "..."

    def export_to_xlsx(self, output_xlsx_path: str, all_results: Dict[str, Dict[str, Dict[str, str]]], 
                       questions: List[str], column_names: List[str]):
        """
        Export results to Excel file with short answers in columns
        
        Args:
            output_xlsx_path: Path to output Excel file
            all_results: Dictionary mapping PDF titles to their structured Q&A results
            questions: List of questions used
            column_names: List of column names for the questions
        """
        try:
            # Prepare data for DataFrame
            data_rows = []
            
            for pdf_title, qa_results in all_results.items():
                row = {'PDF_Title': pdf_title}
                
                # Extract short answers for each question
                for i, question in enumerate(questions):
                    column_name = column_names[i] if i < len(column_names) else f"Question_{i+1}"
                    structured_answer = qa_results.get(question, {})
                    short_answer = self.extract_short_answer(structured_answer)
                    row[column_name] = short_answer
                
                data_rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(data_rows)
            
            # Write to Excel
            with pd.ExcelWriter(output_xlsx_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Extraction_Results', index=False)
                
                # Auto-adjust column widths
                worksheet = writer.sheets['Extraction_Results']
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            self.logger.info(f"Successfully exported results to Excel: {output_xlsx_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export to Excel: {e}")
            raise

    def ask_question(self, i: int, n: int, uploaded_file, pdf_name: str, question: str, additional_instructions: str = "") -> tuple:
        self.logger.info(f"Processing question {i}/{n} for {pdf_name}")
        question_start = time.time()
        answer = self.query_gemini(uploaded_file, question, additional_instructions)
        question_duration = time.time() - question_start
        return answer, question_duration

    def process_pdf(self, pdf_path: str, questions: List[str], additional_instructions: str = "") -> Dict[str, Dict[str, str]]:
        """
        Process a single PDF file with all questions using File API
        
        Args:
            pdf_path: Path to PDF file
            questions: List of questions to ask
            additional_instructions: Additional instructions to append to each question
            
        Returns:
            Dictionary mapping questions to structured answers (dict with short_answer, long_answer, quote)
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
    
    def update_markdown_file(self, md_path: str, pdf_title: str, qa_results: Dict[str, Dict[str, str]]):
        """
        Update markdown file with structured results for a PDF
        
        Args:
            md_path: Path to markdown output file
            pdf_title: Title of the PDF (used as heading)
            qa_results: Dictionary of questions mapped to structured answers (dict with short_answer, long_answer, quote)
        """
        # Read existing content if file exists
        existing_content = ""
        if os.path.exists(md_path):
            with open(md_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        
        # Create new section content with structured format
        new_section = f"\n## {pdf_title}\n\n"
        new_section += f"*Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        
        # Load questions to get the original order and numbering
        try:
            with open('config/questions.yaml', 'r', encoding='utf-8') as f:
                questions_data = yaml.safe_load(f)
                questions_list = questions_data.get('questions', [])
        except Exception as e:
            self.logger.warning(f"Could not load questions.yaml for numbering: {e}")
            questions_list = list(qa_results.keys())
        
        for i, question in enumerate(questions_list, 1):
            if question in qa_results:
                structured_answer = qa_results[question]
                
                # Add question with Q prefix
                new_section += f"**Q{i}:** {question}\n\n"
                
                # Handle both old string format and new structured format for compatibility
                if isinstance(structured_answer, dict):
                    # New structured format with improved formatting
                    short_answer = structured_answer.get('short_answer', 'NA')
                    long_answer = structured_answer.get('long_answer', 'NA')
                    quote = structured_answer.get('quote', 'NA')
                    
                    # Format answers in blockquotes with bold headers
                    new_section += f"> **Short Answer:**\n{self._format_for_blockquote(short_answer)}\n\n"
                    new_section += f"> **Long Answer:**\n{self._format_for_blockquote(long_answer)}\n\n"
                    new_section += f"> **Quote:**\n{self._format_for_blockquote(quote)}\n\n"
                else:
                    # Old string format - convert to structured format for compatibility
                    answer_lines = str(structured_answer).split('\n')
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
    
    def _format_for_blockquote(self, content: str) -> str:
        """
        Format multi-line content to maintain blockquote formatting
        
        Args:
            content: Content that may contain multiple lines and bullet points
            
        Returns:
            Formatted content with proper blockquote prefixes
        """
        if not content or content == 'NA':
            return f"> {content}"
        
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.strip():  # Non-empty lines
                formatted_lines.append(f"> {line}")
            else:  # Empty lines
                formatted_lines.append(">")
        
        return '\n'.join(formatted_lines)
    
    def process_directory(self, pdf_dir: str, questions_file: str, output_file: str, xlsx_output: str = None):
        """
        Process all PDF files in a directory
        
        Args:
            pdf_dir: Directory containing PDF files
            questions_file: Path to YAML file with questions
            output_file: Path to output markdown file
            xlsx_output: Path to output Excel file (optional)
        """
        overall_start_time = time.time()
        self.logger.info(f"Starting PDF analysis process")
        self.logger.info(f"PDF Directory: {pdf_dir}")
        self.logger.info(f"Questions File: {questions_file}")
        self.logger.info(f"Output File: {output_file}")
        if xlsx_output:
            self.logger.info(f"Excel Output File: {xlsx_output}")
        
        # Load questions and additional instructions
        questions, additional_instructions = self.load_questions(questions_file)
        if not questions:
            self.logger.error("No questions loaded, exiting")
            return
        
        # Generate column names for Excel export
        column_names = None
        if xlsx_output:
            column_names = self.generate_column_names_from_questions(questions)
        
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
        
        # Initialize timing statistics and results collection
        processing_times = []
        successful_files = 0
        failed_files = 0
        all_results = {}  # Store all results for Excel export
        
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
                    
                    # Store results for Excel export
                    if xlsx_output:
                        all_results[pdf_title] = qa_results
                    
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
        
        # Export to Excel if requested and we have results
        if xlsx_output and all_results and column_names:
            try:
                self.export_to_xlsx(xlsx_output, all_results, questions, column_names)
            except Exception as e:
                self.logger.error(f"Failed to export Excel file: {e}")
        
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
    parser.add_argument("--xlsx-output", help="Output Excel file path (optional)")
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
    xlsx_output = os.path.abspath(args.xlsx_output) if args.xlsx_output else None
    
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
        analyzer.process_directory(pdf_dir, questions_file, output_file, xlsx_output)
        logger.info(f"Analysis complete. Results saved to: {output_file}")
        if xlsx_output:
            logger.info(f"Excel results saved to: {xlsx_output}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")


if __name__ == "__main__":
    main()
