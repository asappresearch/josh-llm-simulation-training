import sqlite3
import os

def estimate_db_size(db_path):
    # Get the file size
    file_size = os.path.getsize(db_path)
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get the list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    total_rows = 0
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]
        total_rows += row_count
        print(f"Table '{table_name}': {row_count} rows")
    
    conn.close()
    
    print(f"\nDatabase file size: {file_size / (1024 * 1024):.2f} MB")
    print(f"Total number of rows: {total_rows}")
    print(f"Estimated size per row: {file_size / total_rows:.2f} bytes")
    
    # Rough estimate of in-memory size (usually larger due to pandas overhead)
    estimated_memory_usage = total_rows * (file_size / total_rows) * 1.5  # 1.5 is a rough factor for pandas overhead
    print(f"Estimated in-memory size with pandas: {estimated_memory_usage / (1024 * 1024):.2f} MB")

def main():
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    total_estimated_memory = 0
    
    for domain in domains:
        print(f"\nAnalyzing {domain} database:")
        db_path = f'/Users/blattimer/code/josh-llm-simulation-training/db/{domain}-dbase.db'
        if os.path.exists(db_path):
            estimate_db_size(db_path)
            # Add to total estimated memory
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
            table_count = cursor.fetchone()[0]
            conn.close()
            total_estimated_memory += os.path.getsize(db_path) * 1.5 * table_count  # Rough estimate
        else:
            print(f"Database file not found: {db_path}")
    
    print(f"\nTotal estimated memory usage for all domains: {total_estimated_memory / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()