# Module 1: Data Pipeline

## Overview
This module scrapes product data from `books.toscrape.com`, cleans and formats the fields, applies a fixed currency conversion, and stores the structured data in a normalized SQLite relational database (`books.db`).

## Execution
To run the end-to-end pipeline:
```bash
pip install requests beautifulsoup4 pandas
python pipeline.py