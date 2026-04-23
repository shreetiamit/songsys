"""
Song Recommender — Flask demo
CS/DS 3262 course project (Shreeti Amit)

Run:  python app.py
Then open http://127.0.0.1:5000 in your browser.
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

app = Flask(__name__)

# -------------------------------------------------------------
# Load + preprocess (same pipeline as the notebook)
# -------------------------------------------------------------
def load_and_prepare(csv_path='Song_Recommendations.csv'):
    df = pd.read_csv(csv_path)
    df.columns = ['timestamp', 'user_id', 'gender', 'hometown', 'user_language',
                  'song_name', 'artist', 'genre', 'song_language', 'release_year']

    # Strip whitespace
    for col in ['user_id', 'gender', 'hometown', 'user_language',
                'song_name', 'artist', 'genre', 'song_language', 'release_year']:
        df[col] = df[col].astype(str).str.strip()

    # Normalize languages
    def norm_lang(field):
        if pd.isna(field) or field == 'nan':
            return np.nan
        return ';'.join(t.strip().title() for t in field.split(';') if t.strip())
    df['song_language'] = df['song_language'].replace('nan', np.nan).apply(norm_lang)
    df['user_language'] = df['user_language'].replace('nan', np.nan).apply(norm_lang)

    # Collapse "no lyrics" variants
    instrumental_variants = {'No Lyrics', 'No Language', 'Instrumental'}
    df['song_language'] = df['song_language'].apply(
        lambda x: 'Instrumental' if pd.notna(x) and x in instrumental_variants else x)

    # Fill missing
    for col in ['gender', 'hometown', 'song_language']:
        df[col] = df[col].replace('nan', np.nan).fillna('Unknown')

    # Split compound genres
    genre_merge = {'Videogame Music': 'Video Game'}
    def split_genre(g):
        parts = [p.strip() for p in g.split('+')]
        return [genre_merge.get(p, p) for p in parts]
    df['genre_list'] = df['genre'].apply(split_genre)

    # Split multi-valued languages
    def split_semi(x):
        if pd.isna(x) or x == 'Unknown':
            return ['Unknown']
        return [t.strip() for t in x.split(';') if t.strip()]
    df['song_language_list'] = df['song_language'].apply(split_semi)

    # Ordinal release year
    year_order = ['Pre-1960', '1960–1979', '1980–1999', '2000–2009', '2010–2019', '2020+']
    year_to_ord = {y: i for i, y in enumerate(year_order)}
    df['release_year_ord'] = df['release_year'].map(year_to_ord)

    # Multi-hot encode genre and language
    mlb_genre = MultiLabelBinarizer()
    genre_mat = mlb_genre.fit_transform(df['genre_list'])
    genre_cols = [f'genre_{g}' for g in mlb_genre.classes_]
    genre_df = pd.DataFrame(genre_mat, columns=genre_cols, index=df.index)

    mlb_lang = MultiLabelBinarizer()
    lang_mat = mlb_lang.fit_transform(df['song_language_list'])
    lang_cols = [f'lang_{l}' for l in mlb_lang.classes_]
    lang_df = pd.DataFrame(lang_mat, columns=lang_cols, index=df.index)

    # Artist frequency
    artist_counts = df['artist'].value_counts()
    df['artist_freq'] = df['artist'].map(artist_counts)

    # Assemble feature matrix
    num_features = df[['release_year_ord', 'artist_freq']].reset_index(drop=True)
    features = pd.concat([num_features, genre_df.reset_index(drop=True),
                          lang_df.reset_index(drop=True)], axis=1)

    # Min-max scale
    scaler = MinMaxScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index)

    # Song key for lookup
    df['song_key'] = df['song_name'].str.lower().str.strip()

    return {
        'df': df,
        'features': features,
        'features_scaled': features_scaled,
        'X': features_scaled.values,
        'scaler': scaler,
        'year_order': year_order,
        'year_to_ord': year_to_ord,
        'valid_genres': list(mlb_genre.classes_),
        'valid_languages': list(mlb_lang.classes_),
        'artist_counts': artist_counts,
    }

# Load once at startup
STATE = load_and_prepare()
print(f"Loaded {len(STATE['df'])} songs with {STATE['X'].shape[1]} features.")

# -------------------------------------------------------------
# Recommendation functions
# -------------------------------------------------------------
def find_song_indices(song_name):
    key = song_name.lower().strip()
    return STATE['df'].index[STATE['df']['song_key'] == key].tolist()

def songs_matching_prefix(prefix, limit=8):
    """Return suggestions for the autocomplete dropdown."""
    if not prefix:
        return []
    p = prefix.lower().strip()
    df = STATE['df']
    mask = df['song_key'].str.contains(p, regex=False, na=False)
    subset = df[mask].head(limit)
    return [{
        'song_name': r['song_name'],
        'artist': r['artist'],
        'genre': r['genre'],
        'release_year': r['release_year'],
    } for _, r in subset.iterrows()]

def recommend_by_index(song_idx, k=5, metric='cosine'):
    X = STATE['X']
    df = STATE['df']
    if metric == 'cosine':
        scores = cosine_similarity(X[song_idx:song_idx+1], X)[0]
        scores[song_idx] = -np.inf
        top_idx = np.argsort(scores)[::-1][:k]
        top_scores = cosine_similarity(X[song_idx:song_idx+1], X)[0][top_idx]
        score_label = 'cosine'
    else:
        scores = euclidean_distances(X[song_idx:song_idx+1], X)[0]
        scores[song_idx] = np.inf
        top_idx = np.argsort(scores)[:k]
        top_scores = euclidean_distances(X[song_idx:song_idx+1], X)[0][top_idx]
        score_label = 'distance'

    results = []
    for rank, (idx, s) in enumerate(zip(top_idx, top_scores), start=1):
        r = df.iloc[idx]
        results.append({
            'rank': rank,
            'song_name': r['song_name'],
            'artist': r['artist'],
            'genre': r['genre'],
            'release_year': r['release_year'],
            'song_language': r['song_language'],
            'score': round(float(s), 3),
            'score_label': score_label,
        })
    return results

def build_synthetic_vector(genres, languages, release_year, artist=None):
    year_to_ord = STATE['year_to_ord']
    if release_year not in year_to_ord:
        raise ValueError(f"Invalid release_year: {release_year}")
    year_ord = year_to_ord[release_year]

    artist_counts = STATE['artist_counts']
    if artist and artist in artist_counts.index:
        artist_f = artist_counts[artist]
    else:
        artist_f = 1

    valid_genres = STATE['valid_genres']
    valid_langs = STATE['valid_languages']

    genre_vec = np.zeros(len(valid_genres), dtype=int)
    for g in genres:
        if g in valid_genres:
            genre_vec[valid_genres.index(g)] = 1

    lang_vec = np.zeros(len(valid_langs), dtype=int)
    for l in languages:
        if l in valid_langs:
            lang_vec[valid_langs.index(l)] = 1

    raw_vec = np.concatenate([[year_ord, artist_f], genre_vec, lang_vec])
    raw_df = pd.DataFrame(raw_vec.reshape(1, -1), columns=STATE['features'].columns)
    return STATE['scaler'].transform(raw_df)[0]

def recommend_from_vector(vec, k=5, metric='cosine'):
    X = STATE['X']
    df = STATE['df']
    vec = vec.reshape(1, -1)
    if metric == 'cosine':
        scores = cosine_similarity(vec, X)[0]
        top_idx = np.argsort(scores)[::-1][:k]
        score_label = 'cosine'
    else:
        scores = euclidean_distances(vec, X)[0]
        top_idx = np.argsort(scores)[:k]
        score_label = 'distance'

    top_scores = scores[top_idx]
    results = []
    for rank, (idx, s) in enumerate(zip(top_idx, top_scores), start=1):
        r = df.iloc[idx]
        results.append({
            'rank': rank,
            'song_name': r['song_name'],
            'artist': r['artist'],
            'genre': r['genre'],
            'release_year': r['release_year'],
            'song_language': r['song_language'],
            'score': round(float(s), 3),
            'score_label': score_label,
        })
    return results

# -------------------------------------------------------------
# Flask routes
# -------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html',
                           total_songs=len(STATE['df']),
                           valid_genres=STATE['valid_genres'],
                           valid_languages=STATE['valid_languages'],
                           valid_years=STATE['year_order'])

@app.route('/api/suggest')
def api_suggest():
    q = request.args.get('q', '')
    return jsonify(songs_matching_prefix(q))

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    payload = request.get_json()
    song_name = (payload.get('song_name') or '').strip()
    k = int(payload.get('k', 5))
    metric = payload.get('metric', 'cosine')

    if not song_name:
        return jsonify({'error': 'Please enter a song name.'}), 400

    matches = find_song_indices(song_name)
    if matches:
        idx = matches[0]
        q = STATE['df'].iloc[idx]
        results = recommend_by_index(idx, k=k, metric=metric)
        return jsonify({
            'found': True,
            'query': {
                'song_name': q['song_name'],
                'artist': q['artist'],
                'genre': q['genre'],
                'release_year': q['release_year'],
                'song_language': q['song_language'],
            },
            'results': results,
            'metric': metric,
        })

    # Song not found — check if fallback features were supplied
    fb_genres = payload.get('fallback_genres') or []
    fb_langs = payload.get('fallback_languages') or []
    fb_year = payload.get('fallback_year')
    fb_artist = (payload.get('fallback_artist') or '').strip() or None

    if not fb_genres or not fb_langs or not fb_year:
        return jsonify({
            'found': False,
            'needs_fallback': True,
            'message': f'"{song_name}" is not in the dataset. Please provide features so I can find similar songs.',
        })

    try:
        vec = build_synthetic_vector(fb_genres, fb_langs, fb_year, artist=fb_artist)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    results = recommend_from_vector(vec, k=k, metric=metric)
    return jsonify({
        'found': False,
        'used_fallback': True,
        'query': {
            'song_name': song_name,
            'artist': fb_artist or '(unknown)',
            'genre': ', '.join(fb_genres),
            'release_year': fb_year,
            'song_language': ', '.join(fb_langs),
        },
        'results': results,
        'metric': metric,
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
