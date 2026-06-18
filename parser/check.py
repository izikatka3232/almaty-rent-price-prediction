import sqlite3

conn = sqlite3.connect('db.sqlite')

# Структура таблиц
print("=== FLATS columns ===")
print(conn.execute("PRAGMA table_info(flats)").fetchall())

print("\n=== PRICES columns ===")
print(conn.execute("PRAGMA table_info(prices)").fetchall())

# Пример данных
print("\n=== FLATS sample ===")
for row in conn.execute("SELECT * FROM flats LIMIT 3").fetchall():
    print(row)

conn.close()