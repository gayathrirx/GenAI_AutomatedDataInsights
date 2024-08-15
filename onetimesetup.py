import psycopg2

import os
import psycopg2
from psycopg2 import sql
import csv
from datetime import datetime

# from google.colab import userdata


DATABASE_URL=os.getenv('roach_url2')

try:
  connection = psycopg2.connect(DATABASE_URL)

  with connection.cursor() as cur:
      cur.execute("SELECT now()")
      res = cur.fetchall()
      connection.commit()
      print(res)

  # Create a cursor object
  mycursor = connection.cursor()
  print("DB Connection Successful")  

except Exception as error:
    print("Error while connecting to CockroachDB", error)

##########START - One time execution to create tables#########################
# try:

#   mycursor.execute("CREATE TABLE IF NOT EXISTS distribution_centers(id INT, name VARCHAR(255), latitude FLOAT, longitude FLOAT)")
#   mycursor.execute("CREATE TABLE IF NOT EXISTS events(id INT, user_id INT, sequence_number INT, session_id VARCHAR(255), created_at TIMESTAMP, ip_address VARCHAR(255), city VARCHAR(255), state VARCHAR(255), postal_code VARCHAR(255), browser VARCHAR(255), traffic_source VARCHAR(255), uri VARCHAR(255), event_type VARCHAR(255))")
#   mycursor.execute("CREATE TABLE IF NOT EXISTS inventory_items(id INT, product_id INT, created_at TIMESTAMP, sold_at TIMESTAMP, cost FLOAT, product_category VARCHAR(255), product_name VARCHAR(255), product_brand VARCHAR(255), product_retail_price FLOAT, product_department VARCHAR(255), product_sku VARCHAR(255), product_distribution_center_id INT)")
#   mycursor.execute("CREATE TABLE IF NOT EXISTS order_items(id INT, order_id INT, user_id INT, product_id INT, inventory_item_id INT, status VARCHAR(255), created_at TIMESTAMP, shipped_at TIMESTAMP, delivered_at TIMESTAMP, returned_at TIMESTAMP, sale_price FLOAT)")
#   mycursor.execute("CREATE TABLE IF NOT EXISTS orders(order_id INT, user_id INT, status VARCHAR(255), gender VARCHAR(255), created_at TIMESTAMP, returned_at TIMESTAMP, shipped_at TIMESTAMP, delivered_at TIMESTAMP, num_of_item INT)")
#   mycursor.execute("CREATE TABLE IF NOT EXISTS products(id INT, cost FLOAT, category VARCHAR(255), name VARCHAR(255), brand VARCHAR(255), retail_price FLOAT, department VARCHAR(255), sku VARCHAR(255), distribution_center_id INT)")
#   mycursor.execute("CREATE TABLE IF NOT EXISTS users(id INT, first_name VARCHAR(255), last_name VARCHAR(255), email VARCHAR(255), age INT, gender VARCHAR(255), state VARCHAR(255), street_address VARCHAR(255), postal_code VARCHAR(255), city VARCHAR(255), country VARCHAR(255), latitude FLOAT, longitude FLOAT, traffic_source VARCHAR(255), created_at TIMESTAMP)")

#   connection.commit()  # Commit the transaction
#   print("Tables created successfully")

# except Exception as error:
#     print("Error while connecting to CockroachDB or creating table:", error)

# finally:
#     # Close the cursor and connection to the database
#     if connection:
#         mycursor.close()
#         connection.close()
#         print("CockroachDB connection closed")

##########END - One time execution to create tables#########################


###########START - One time execution to load data into DB##################
# try:

#   table_names = ["distribution_centers", "events", "inventory_items", "order_items", "orders", "products", "users"]

#   # Function to detect timestamp columns based on data types
#   def detect_timestamp_columns(cursor, table_name):
#       mycursor.execute("Select * FROM events LIMIT 0")
#       columns = [desc[0] for desc in mycursor.description]
#       # cursor.execute(f"DESCRIBE {table_name}")
#       # columns = cursor.fetchall()
#       timestamp_indices = [i for i, column in enumerate(columns) if column[1].startswith('timestamp')]
#       return timestamp_indices


#   for table_name in table_names:
#       timestamp_indices = detect_timestamp_columns(mycursor, table_name)

#       with open(f"data/{table_name}.csv", 'r', encoding='utf-8') as csv_file:  # Specifying utf-8 encoding
#           csv_data = csv.reader(csv_file)
#           next(csv_data)  # Skip headers
#           counter = 0
#           print(f"Currently inserting data into table {table_name}")

#           for row in csv_data:
#               if counter % 10000 == 0:
#                   print(f"Progress is {counter}")

#               row = [None if cell == "" else cell for cell in row]

#               # Correct datetime values if necessary
#               for col_index in timestamp_indices:
#                   if row[col_index] is not None and row[col_index] != '':
#                       try:
#                           datetime.strptime(row[col_index], '%Y-%m-%d %H:%M:%S')
#                       except ValueError:
#                           # Handle invalid datetime value by setting it to None
#                           row[col_index] = None
#                   else:
#                       # Handle None or empty string
#                       row[col_index] = None

#               postfix = ','.join(["%s"] * len(row))
#               query = f"INSERT INTO {table_name} VALUES ({postfix})"
#               try:
#                   mycursor.execute(query, row)
#               except Exception as err:
#                   print(f"Error: {err}")
#                   print(f"Failed row: {row}")

#               counter += 1

#           connection.commit()
#           print(f"Completed inserting data into table {table_name}")

# except Exception as error:
#     print("Error while connecting to CockroachDB or inserting data:", error)

# finally:
    # Close the cursor and connection to the database
    # if connection:
    #     mycursor.close()
    #     connection.close()
    #     print("CockroachDB connection closed")
###########END - One time execution to load data into DB##################