# PDF Extraction Tool

A tool for extracting structured information from PDF documents using Google's Gemini AI API. The tool processes multiple PDFs and answers predefined research questions about each document.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up configuration:**
   ```bash
   cp config/config.template.yaml config/config.yaml
   ```
   Edit `config/config.yaml` and add your Gemini API key (get one at [Google AI Studio](https://aistudio.google.com/app/apikey))

3. **Add your PDFs:**
   Place PDF files in the `pdfs` directory

4. **Run extraction:**
   - **Option A - Python script:** `python run_extraction.py`
   - **Option B - Jupyter notebook:** Open `run_extraction.ipynb`

## Configuration

### Main Configuration File

The tool uses `config/config.yaml` for all settings. Create this file by copying `config/config.template.yaml`:

```yaml
# Gemini AI Configuration
gemini:
  api_key: "your_api_key_here"  # Required: Get from https://aistudio.google.com/app/apikey
  model: "gemini-2.0-flash-lite"  # Model to use

# File Paths
paths:
  pdf_directory: "pdfs"                # Directory containing PDF files
  questions_file: "config/questions.yaml"    # File containing questions to ask
  output_file: "output.md"             # Output markdown file
  log_file: "logs/extraction_{timestamp}.log"  # Log file with timestamp

# Processing Options
options:
  log_level: "INFO"                   # Logging level
  confirm_before_processing: true     # Ask for confirmation before processing
```

### Questions Configuration

Edit `config/questions.yaml` to customize the questions asked about each PDF:

```yaml
questions:
  - "What is the main research question?"
  - "What methodology was used?"
  - "What were the key findings?"

additional_instructions: "Please provide concise answers with quotes and page references."
```

### API Key Setup

You can provide your Gemini API key in two ways:

1. **In config file:** Set `api_key` in `config/config.yaml`
2. **Environment variable:** Set `GEMINI_API_KEY`
   ```bash
   # Windows PowerShell
   $env:GEMINI_API_KEY = "your_api_key_here"
   
   # Linux/Mac
   export GEMINI_API_KEY="your_api_key_here"
   ```

## Running the Extraction

### Method 1: Python Script

Use `run_extraction.py` for command-line execution:

```bash
python run_extraction.py
```

The script will:
- Load configuration from `config/config.yaml`
- Show a summary of settings
- Ask for confirmation (if enabled)
- Process all PDFs in the configured directory
- Save results to the configured output file

### Method 2: Jupyter Notebook

Use `run_extraction.ipynb` for interactive processing:

1. **Open the notebook:** Launch Jupyter and open `run_extraction.ipynb`
2. **Run cells sequentially:** Each cell handles a different step:
   - Cell 1-6: Setup and configuration
   - Cell 7: Basic processing with console output
   - Cell 8: Processing with progress bar (recommended)

The notebook offers two processing options:
- **Cell 7:** Standard processing with detailed console output
- **Cell 8:** Processing with a visual progress bar and quiet logging

## Output

### Results File

Processed results are saved to a markdown file (default: `output.md`) with the following structure:

```markdown
# PDF Analysis Results

Generated on: 2025-01-19 11:10:32
Model used: gemini-2.0-flash-lite
Files to process: 2

---

## Document Title 1

*Processed on: 2025-01-19 11:15:45*

• **Question 1**

> Answer with quotes and page references

• **Question 2**

> Another detailed answer

## Document Title 2

...
```

### Log Files

Detailed logs are saved to the `logs` directory with timestamps:
- Processing status for each PDF
- Question-by-question progress
- Error messages and timing information
- Final summary statistics

## Project Structure

```
├── config/
│   ├── config.yaml          # Main configuration (create from template)
│   ├── config.template.yaml # Template configuration file
│   └── questions.yaml       # Questions to ask about PDFs
├── modules/
│   └── llm_extractor.py     # Main extraction logic
├── pdfs/                    # Place your PDF files here
├── logs/                    # Processing logs (auto-created)
├── run_extraction.py        # Command-line script
├── run_extraction.ipynb     # Jupyter notebook version
├── requirements.txt         # Python dependencies
└── output.md               # Results file (auto-generated)
```

## Features

- **Batch processing:** Handle multiple PDFs automatically
- **Configurable questions:** Customize research questions via YAML
- **Progress tracking:** Visual progress bars in notebook mode
- **Detailed logging:** Comprehensive logs with timing information
- **Error handling:** Graceful handling of API errors and file issues
- **Resume capability:** Can reprocess individual files by updating sections
- **Flexible output:** Structured markdown output with quotes and references

## Dependencies

See `requirements.txt` for the complete list:
- `google-generativeai` - Gemini AI API client
- `PyYAML` - Configuration file parsing
- `tqdm` - Progress bars (notebook only)

## Troubleshooting

**API Key Issues:**
- Ensure your API key is valid and has sufficient quota
- Check both `config/config.yaml` and environment variables

**File Not Found:**
- Verify PDF files are in the configured directory
- Check that `config/questions.yaml` exists

**Processing Errors:**
- Check the log files in `logs` for detailed error messages
- Some PDFs may fail due to format issues or API timeouts

## TODOs
* better output formatting

## License

This project is licensed under the GNU License.