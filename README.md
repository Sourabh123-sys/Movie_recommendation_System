# Movie Recommendation System

A simple Movie Recommendation System built with Python and Flask.  
This project allows users to select a movie and get top similar movie recommendations using **cosine similarity**.

---

## Features

- Interactive web interface using **Flask**
- Provides **top 5 movie recommendations** based on the selected movie
- Uses a **preprocessed dataset** (`data.pkl`) and **similarity matrix** (`matrix.pkl`)
- Logs which dataset is used for recommendations

---

## Tech Stack

- Python 3.x
- Flask
- Pandas
- Scikit-learn (cosine similarity)
- HTML/CSS (for frontend)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Sourabh123-sys/Movie_Recommendation_System.git
cd Movie_Recommendation_System

Create and activate a virtual environment:

2. python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux


3.Install dependencies:

pip install -r requirements.txt

Usage

Run the Flask app:

python app.py

Open a browser and go to: http://127.0.0.1:5000

Select a movie from the dropdown to get recommendations.
