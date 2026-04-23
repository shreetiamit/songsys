# SONGSYS — Content-based Song Recommender

A live demo of the content-based song recommender built for CS/DS 3262 (Applied Machine Learning, Vanderbilt, Spring 2026). Trained on a 172-song class survey dataset.

Built by Shreeti Amit.

## What it does

- Type any song in the dataset → returns the 5 most similar songs by cosine similarity over 27 content features (16 multi-hot genres, 9 multi-hot languages, ordinal release era, artist frequency)
- Type a song **not** in the dataset → the app asks for genre(s), language(s), release era, and optionally artist, then builds a synthetic feature vector and finds the best match
- Toggle between cosine similarity and Euclidean distance
- Adjustable top-K (1–15)
- Live autocomplete as you type

## Running locally

Requires Python 3.8+.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Flask app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Files

```
.
├── app.py                       # Flask server + recommender pipeline
├── requirements.txt             # Python deps
├── Song_Recommendations.csv     # The class-sourced dataset (172 rows)
├── templates/
│   └── index.html               # Single-page UI (HTML + CSS + vanilla JS)
└── README.md
```

## How it works (technical)

1. On startup, `app.py` loads the CSV, applies the same preprocessing pipeline used in the notebook (text cleaning, `MultiLabelBinarizer` for genre & language, ordinal encoding for release year, frequency encoding for artist, `MinMaxScaler`), and caches the resulting 172 × 27 feature matrix in memory.
2. The `/api/recommend` endpoint takes a song name, looks it up case-insensitively, and returns the top-K rows of the cosine (or Euclidean) similarity matrix with the query row excluded.
3. If the song isn't found, the same endpoint accepts fallback feature inputs, builds a synthetic feature vector through the exact same preprocessing pipeline (so scaling matches), and returns recommendations against that.
4. The `/api/suggest` endpoint powers the autocomplete dropdown with substring matches.

## Try these queries

- **Levitating** — clean Pop/2020+ match, all 5 hits are Pop/English/2020+
- **Paranoid Android** — classic Rock, era-sensitive (80s–00s)
- **Golden** — tests niche K-pop genre (only 6 songs in the dataset)
- **Pavia Mafia** — only Mariachi song in the dataset; gracefully falls back on language + era
- **Thriller** — not in dataset → triggers the fallback flow. Try: genres `Pop` + `R&B / Soul`, language `English`, era `1980–1999`, artist `Michael Jackson`. The top hit should be Michael Jackson's "P.Y.T." (a real track from the dataset surfaced via the artist-frequency signal).

## License

Course project; dataset is anonymized class survey data.
