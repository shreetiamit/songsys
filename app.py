import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

app = Flask(__name__)

def load_and_prepare():
    df = pd.read_csv('spotify_songs.csv')
    
    # Dedupe on (track_name, track_artist), keeping highest popularity
    df.sort_values('track_popularity', ascending=False, inplace=True)
    df.drop_duplicates(subset=['track_name', 'track_artist'], inplace=True)
    
    # Extract year from track_album_release_date
    df['year'] = pd.to_datetime(df['track_album_release_date'], errors='coerce').dt.year
    
    # Scale audio features and year
    scaler = StandardScaler()
    audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                      'speechiness', 'acousticness', 'instrumentalness', 
                      'liveness', 'valence', 'tempo']
    df[audio_features + ['year']] = scaler.fit_transform(df[audio_features + ['year']])

    # One-hot encode playlist_genre
    genre_encoder = OneHotEncoder()
    genre_encoded = genre_encoder.fit_transform(df[['playlist_genre']])
    genre_columns = [f'genre_{g}' for g in genre_encoder.categories_[0]]
    genre_df = pd.DataFrame(genre_encoded.toarray(), columns=genre_columns)
    
    # Multi-hot encode playlist_subgenre
    subgenre_encoder = MultiLabelBinarizer()
    subgenre_encoded = subgenre_encoder.fit_transform(df['playlist_subgenre'].str.split(', '))
    subgenre_columns = [f'subgenre_{s}' for s in subgenre_encoder.classes_]
    subgenre_df = pd.DataFrame(subgenre_encoded, columns=subgenre_columns)

    # Assemble final feature matrix  
    X = pd.concat([df[audio_features + ['year']], genre_df, subgenre_df], axis=1)
    
    return {
        'df': df,
        'X': X,
        'audio_features': audio_features,
        'subgenre_encoder': subgenre_encoder,
    }

STATE = load_and_prepare() 
DF = STATE['df']
X = STATE['X']

@app.route('/')
def index():
    return render_template('index.html', 
        subgenres=sorted(STATE['subgenre_encoder'].classes_))

@app.route('/api/suggest')
def suggest():
    q = request.args.get('q', '')
    matches = DF[DF.apply(lambda r: q.lower() in r['track_name'].lower(), axis=1)]
    matches = matches.head(10).apply(lambda r: f"{r['track_name']} - {r['track_artist']} ({r['year']})", axis=1).tolist()
    return jsonify(matches)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    track = data['track']
    name, artist, year = track.rsplit(' (', 1)[0].rsplit(' - ', 1) + [track.rsplit(' (', 1)[1][:-1]]  
    n = int(data.get('n', 5))
    metric = data.get('metric', 'cosine')
    subgenres = data.get('subgenres', [])
    
    try:
        idx = DF.index[(DF['track_name'] == name) & (DF['track_artist'] == artist) & (DF['year'] == int(year))][0]
    except IndexError:
        return jsonify({
            'found': False,
            'message': f"{track} not found in the dataset."
        }), 404
        
    results = _recommend(idx, X, DF, n, metric)
    
    if subgenres:
        mask = DF[STATE['subgenre_encoder'].transform([subgenres]).toarray().ravel().astype(bool)]
        results = [r for r in results if mask[r['idx']]][:n]
        
    for r in results:
        r['audio_features'] = dict(zip(STATE['audio_features'], X.iloc[r['idx']][STATE['audio_features']].tolist()))
        
    return jsonify({
        'found': True,
        'query': {
            'track': track,
            'name': name,
            'artist': artist,
            'year': year,
        }, 
        'results': results,
        'subgenres': subgenres,
        'metric': metric,
    })

def _recommend(idx, X, df, n, metric):
    scores = cosine_similarity(X.iloc[idx:idx+1], X)[0] if metric == 'cosine' else euclidean_distances(X.iloc[idx:idx+1], X)[0]
    top_idx = np.argsort(scores)[::-1] if metric == 'cosine' else np.argsort(scores)
    results = []
    for i in top_idx:
        if i == idx:
            continue
        r = df.iloc[i]
        results.append({
            'idx': i,
            'track_name': r['track_name'], 
            'track_artist': r['track_artist'],
            'year': r['year'],
            'score': scores[i],
        })
        if len(results) == n:
            break
    return results
        
if __name__ == '__main__':
    app.run(debug=True)
