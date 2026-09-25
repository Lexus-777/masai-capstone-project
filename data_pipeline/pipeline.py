import requests
from bs4 import BeautifulSoup
import sqlite3
import pandas as pd

print("Starting the web scraping process...")

# 1. Target 4 specific categories with fewer than 20 books each.
# This collects 63 total books, satisfying the >60 requirement without complex page-turning code.
urls = {
    "Travel": "https://books.toscrape.com/catalogue/category/books/travel_2/index.html",
    "Science": "https://books.toscrape.com/catalogue/category/books/science_22/index.html",
    "Poetry": "https://books.toscrape.com/catalogue/category/books/poetry_23/index.html",
    "Classics": "https://books.toscrape.com/catalogue/category/books/classics_6/index.html"
}

# A simple dictionary to translate text ratings into numbers
rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
books_data = []

# 2. Scrape and Clean the Data
for category, url in urls.items():
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find every book card on the web page
    for article in soup.find_all('article', class_='product_pod'):
        try:
            # Extract raw data from HTML tags
            title = article.find('h3').find('a')['title']
            price_text = article.find('p', class_='price_color').text
            rating_text = article.find('p', class_='star-rating')['class'][1]
            availability_text = article.find('p', class_='instock').text.strip()
            
            # Clean data into proper numeric and boolean types
            price_gbp = float(price_text.replace('Â£', '').replace('£', ''))
            rating = rating_map[rating_text]
            in_stock = True if 'In stock' in availability_text else False
            
            # Apply the project's fixed currency conversion rate (1 GBP = 105.50 INR)
            price_inr = round(price_gbp * 105.50, 2)
            
            books_data.append((title, price_gbp, price_inr, rating, in_stock, category))
        except Exception as e:
            # If any row is messy and fails to parse, we drop it by skipping to the next one
            continue

print(f"Successfully scraped and cleaned {len(books_data)} books.")

# 3. Create the Database and Tables
print("Connecting to database...")
conn = sqlite3.connect('books.db')
cursor = conn.cursor()

# Create a normalized schema with a Primary Key / Foreign Key relationship
cursor.execute('''
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock BOOLEAN,
    category_id INTEGER,
    FOREIGN KEY(category_id) REFERENCES categories(category_id)
)
''')

# 4. Insert the Cleaned Data
for cat in urls.keys():
    cursor.execute('INSERT OR IGNORE INTO categories (category_name) VALUES (?)', (cat,))

cursor.execute('SELECT * FROM categories')
cat_dict = {name: cat_id for cat_id, name in cursor.fetchall()}

for b in books_data:
    title, price_gbp, price_inr, rating, in_stock, category = b
    cat_id = cat_dict[category]
    cursor.execute('''
    INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (title, price_gbp, price_inr, rating, in_stock, cat_id))

conn.commit()
print("Data saved to books.db!")

# 5. Execute the 5 Required SQL Queries
print("\n--- Running SQL Queries ---\n")

queries = {
    "1. SELECT/WHERE: All 5-star books": 
        "SELECT title, rating FROM books WHERE rating = 5 LIMIT 5;",
    "2. ORDER BY/LIMIT: Top 3 most expensive books": 
        "SELECT title, price_gbp FROM books ORDER BY price_gbp DESC LIMIT 3;",
    "3. DISTINCT: Unique ratings available": 
        "SELECT DISTINCT rating FROM books ORDER BY rating;",
    "4. IN/BETWEEN: Books priced between £10 and £20": 
        "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 20 LIMIT 5;",
    "5. JOIN: 5 highest-rated books across categories": 
        "SELECT c.category_name, b.title, b.rating FROM books b JOIN categories c ON b.category_id = c.category_id ORDER BY b.rating DESC LIMIT 5;"
}

for desc, q in queries.items():
    print(desc)
    cursor.execute(q)
    for row in cursor.fetchall():
        print("  ", row)
    print("")

# 6. Pandas Equivalency Check
print("--- Verifying JOIN with Pandas ---")
# Read tables back out of the database into Pandas DataFrames
df_books = pd.read_sql("SELECT * FROM books", conn)
df_categories = pd.read_sql("SELECT * FROM categories", conn)

# Replicate the SQL JOIN using a Pandas merge
df_merged = pd.merge(df_books, df_categories, on='category_id')

# Filter and sort to match Query 5
top_pandas = df_merged.sort_values(by='rating', ascending=False)[['category_name', 'title', 'rating']].head(5)
print("\nPandas Merge Result (Should match Query 5 output exactly):")
print(top_pandas.to_string(index=False))

conn.close()