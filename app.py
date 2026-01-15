import os
import sqlite3
from datetime import datetime

from flask import Flask, render_template, request, send_from_directory

from shader import shader


def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn


# Initialize the table
with get_db_connection() as conn:
    conn.execute("""CREATE TABLE IF NOT EXISTS conversions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  orig_name TEXT,
                  ascii_path TEXT,
                  upload_time DATETIME)""")

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/convert", methods=["POST"])
def convert():
    file = request.files.get("image")
    if not file:
        return "No file", 400

    # 1. Save the incoming file temporarily
    temp_input_path = os.path.join(
        app.config["UPLOAD_FOLDER"], "temp_" + str(file.filename)
    )
    file.save(temp_input_path)

    # 2. Define the output path
    output_filename = "ascii_" + str(file.filename)
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

    # 3. RUN YOUR SHADER
    try:
        shader(temp_input_path, output_path)
    finally:
        # Clean up the original uploaded file to save space
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

    # LOG TO DATABASE
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO conversions (orig_name, ascii_path, upload_time) VALUES (?, ?, ?)",
            (
                file.filename,
                output_filename,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()

    return render_template("result.html", filename=output_filename)


@app.route("/history")
def history():
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM conversions ORDER BY upload_time DESC"
        ).fetchall()
    return render_template("history.html", conversions=rows)


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    # 1. Connect to the database file (it creates the file if it doesn't exist)
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # 2. Create the table with the exact columns we need
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            orig_name TEXT NOT NULL,
            ascii_path TEXT NOT NULL,
            upload_time DATETIME NOT NULL
        )
    """)

    # 3. Save and close
    conn.commit()
    conn.close()

    # 4. Start the Flask app
    app.run(debug=True)
