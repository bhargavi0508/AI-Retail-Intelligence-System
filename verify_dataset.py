import csv

file_path = "data/electronics_sales_data.csv"

with open(
    file_path,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.reader(file)

    rows = list(reader)

headers = rows[0]
data = rows[1:]

print()
print("=" * 50)
print("DATASET VERIFICATION")
print("=" * 50)

print(f"Rows    : {len(data)}")
print(f"Columns : {len(headers)}")

print()
print("Columns:")

for column in headers:
    print("-", column)

print()
print("First 5 records:")

for row in data[:5]:
    print(row)

print()
print("=" * 50)
print("VERIFICATION COMPLETED")
print("=" * 50)