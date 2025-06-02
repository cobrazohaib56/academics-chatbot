import os
import pandas as pd
import sqlite3

def create_db_from_xlsx(xlsx_file_path):
    """
    Create a SQLite database from an XLSX file.
    
    Args:
        xlsx_file_path (str): Path to the XLSX file
        
    Returns:
        sqlite3.Connection: Connection to the SQLite database
    """
    if not os.path.exists(xlsx_file_path):
        raise FileNotFoundError(f"XLSX file not found: {xlsx_file_path}")
    
    print(f"Loading XLSX file: {xlsx_file_path}")
    
    # Read XLSX file
    df = pd.read_excel(xlsx_file_path)
    
    # Replace NaN values with None for SQL compatibility
    df = df.replace({pd.NA: None})
    df = df.where(pd.notnull(df), None)
    
    # Create in-memory SQLite database
    conn = sqlite3.connect("library.db")
    
    # Convert column names to SQL-friendly format (replace spaces with underscores)
    df.columns = [col.replace(' ', '_') for col in df.columns]
    
    # Write to SQLite
    df.to_sql("Library", conn, if_exists="replace", index=False)
    
    # Print info about the database
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(Library)")
    columns = cursor.fetchall()
    
    print(f"\nCreated database table 'Library' with columns:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    print(f"\nLoaded {len(df)} records into the database")
    
    return conn

def main():
    # Path to your Excel file
    xlsx_file_path = "data/raw/books_catalog.xlsx"
    
    # Expand user directory if needed
    xlsx_file_path = os.path.expanduser(xlsx_file_path)
    
    try:
        # Create database and get connection
        conn = create_db_from_xlsx(xlsx_file_path)
        print("\nDatabase created successfully!")
        
        # Print some basic info about the data
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Library")
        count = cursor.fetchone()[0]
        print(f"Total records in database: {count}")
        
        # Print column names
        cursor.execute("PRAGMA table_info(Library)")
        columns = cursor.fetchall()
        print("\nColumns in the database:")
        for col in columns:
            print(f"- {col[1]} ({col[2]})")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Close database connection if it exists
        if 'conn' in locals() and conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    main() 