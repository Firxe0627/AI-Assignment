import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #aaaaaa;
    margin-bottom: 30px;
}

.movie-card {
    padding: 15px;
    border-radius: 10px;
    background-color: #16191f;
    margin-bottom: 20px;
}

.similarity {
    font-size: 22px;
    font-weight: 600;
}

.movie-title {
    font-size: 22px;
    font-weight: 600;
}

.genre {
    color: #aaaaaa;
    font-size: 14px;
}

.description {
    color: #cccccc;
    font-size: 15px;
    line-height: 1.5;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎬 Movie Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Item-Based Collaborative Filtering</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter a movie title and the system will recommend similar movies "
    "using Item-Based Collaborative Filtering."
)


# ============================================================
# LOAD MODEL FILES
# ============================================================

@st.cache_resource
def load_model():

    # Item similarity information
    item_distances = np.load(
        "models/item_distances.npy"
    )

    item_indices = np.load(
        "models/item_indices.npy"
    )

    # Movie ID -> matrix index
    with open(
        "models/movie_to_index.pkl",
        "rb"
    ) as f:
        movie_to_index = pickle.load(f)

    # Matrix index -> Movie ID
    with open(
        "models/index_to_movie.pkl",
        "rb"
    ) as f:
        index_to_movie = pickle.load(f)

    # Movie metadata
    movie_metadata = pd.read_csv(
        "models/movie_metadata.csv"
    )

    return (
        item_distances,
        item_indices,
        movie_to_index,
        index_to_movie,
        movie_metadata
    )


try:

    (
        item_distances,
        item_indices,
        movie_to_index,
        index_to_movie,
        movie_metadata
    ) = load_model()

    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error(
        f"Unable to load recommendation model: {e}"
    )

    st.stop()


# ============================================================
# LOAD LINKS.CSV
# ============================================================

@st.cache_data
def load_links():

    links = pd.read_csv(
        "data/links.csv"
    )

    # Make sure IDs are numeric
    links["movieId"] = pd.to_numeric(
        links["movieId"],
        errors="coerce"
    )

    links["tmdbId"] = pd.to_numeric(
        links["tmdbId"],
        errors="coerce"
    )

    return links


links = load_links()


# ============================================================
# ADD TMDB ID TO MOVIE METADATA
# ============================================================

# Avoid duplicate tmdbId if metadata already contains it

if "tmdbId" not in movie_metadata.columns:

    movie_metadata = movie_metadata.merge(
        links[["movieId", "tmdbId"]],
        on="movieId",
        how="left"
    )


# ============================================================
# MODEL STATUS
# ============================================================

st.success(
    "✅ Recommendation model loaded successfully"
)


# ============================================================
# MODEL STATISTICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Movies",
        len(movie_to_index)
    )

with col2:

    st.metric(
        "Users",
        128075
    )

with col3:

    st.metric(
        "Recommendation Method",
        "Item-Based CF"
    )


st.divider()


# ============================================================
# TMDB CONFIGURATION
# ============================================================

TMDB_TOKEN = None

try:

    TMDB_TOKEN = st.secrets["TMDB_TOKEN"]

except Exception:

    TMDB_TOKEN = None


# ============================================================
# TMDB CONFIGURATION
# ============================================================

TMDB_TOKEN = None

try:
    TMDB_TOKEN = st.secrets["TMDB_TOKEN"]
except Exception:
    TMDB_TOKEN = None


# ============================================================
# TMDB REQUEST HEADERS
# ============================================================

def tmdb_headers():

    return {
        "Authorization": f"Bearer {TMDB_TOKEN}",
        "accept": "application/json"
    }


# ============================================================
# GET MOVIE BY TMDB ID
# ============================================================

@st.cache_data(show_spinner=False)
def get_tmdb_movie(tmdb_id):

    if TMDB_TOKEN is None:
        return None

    if pd.isna(tmdb_id):
        return None

    try:

        url = (
            f"https://api.themoviedb.org/3/movie/"
            f"{int(tmdb_id)}"
        )

        response = requests.get(
            url,
            headers=tmdb_headers(),
            timeout=10
        )

        if response.status_code == 200:

            return response.json()

    except Exception:
        pass

    return None


# ============================================================
# SEARCH MOVIE ON TMDB
# ============================================================

@st.cache_data(show_spinner=False)
def search_tmdb_movie(title, year=None):

    if TMDB_TOKEN is None:
        return None

    try:

        url = (
            "https://api.themoviedb.org/3/search/movie"
        )

        params = {
            "query": title,
            "include_adult": "false",
            "language": "en-US"
        }

        # Add year if available
        if year is not None:

            try:
                params["year"] = int(year)
            except Exception:
                pass

        response = requests.get(
            url,
            headers=tmdb_headers(),
            params=params,
            timeout=10
        )

        if response.status_code != 200:
            return None

        data = response.json()

        results = data.get(
            "results",
            []
        )

        if not results:
            return None

        # Return the first matching result
        return results[0]

    except Exception:
        return None


# ============================================================
# GET MOVIE INFORMATION WITH FALLBACK
# ============================================================

def get_movie_information(
    tmdb_id,
    title
):

    # --------------------------------------------------------
    # Method 1: TMDB ID
    # --------------------------------------------------------

    movie = get_tmdb_movie(
        tmdb_id
    )

    if movie is not None:
        return movie


    # --------------------------------------------------------
    # Method 2: TMDB title search
    # --------------------------------------------------------

    year = None

    try:

        title_string = str(title)

        if "(" in title_string and ")" in title_string:

            year_text = title_string[
                title_string.rfind("(") + 1:
                title_string.rfind(")")
            ]

            if year_text.isdigit():

                year = int(year_text)

    except Exception:
        pass


    # Remove year from title
    clean_title = str(title)

    if year is not None:

        clean_title = clean_title[
            :clean_title.rfind("(")
        ].strip()


    movie = search_tmdb_movie(
        clean_title,
        year
    )

    return movie


# ============================================================
# POSTER URL
# ============================================================

def get_poster_url(poster_path):

    if not poster_path:
        return None

    return (
        "https://image.tmdb.org/t/p/w500"
        + poster_path
    )


# ============================================================
# FIND MOVIES
# ============================================================

def search_movies(search_text):

    if not search_text:

        return pd.DataFrame()

    search_text = search_text.strip().lower()

    matches = movie_metadata[
        movie_metadata["title"]
        .astype(str)
        .str.lower()
        .str.contains(
            search_text,
            regex=False,
            na=False
        )
    ].copy()

    return matches.head(20)


# ============================================================
# ITEM-BASED RECOMMENDER
# ============================================================

def recommend_similar_movies(
    movie_id,
    n=10
):

    # Check movie exists
    if movie_id not in movie_to_index:

        return pd.DataFrame()

    movie_index = movie_to_index[movie_id]

    # Get precomputed similar movies
    distances = item_distances[
        movie_index
    ]

    indices = item_indices[
        movie_index
    ]

    recommendations = []

    for distance, similar_index in zip(
        distances,
        indices
    ):

        similar_index = int(similar_index)

        # Don't recommend the selected movie itself
        if similar_index == movie_index:
            continue

        # Convert matrix index -> movieId
        try:

            similar_movie_id = index_to_movie[
                similar_index
            ]

        except Exception:

            continue

        # Convert distance to similarity
        similarity = 1 - float(distance)

        recommendations.append({
            "movieId": similar_movie_id,
            "similarity": similarity
        })

        if len(recommendations) >= n:

            break

    recommendations_df = pd.DataFrame(
        recommendations
    )

    if recommendations_df.empty:

        return recommendations_df

    # Add metadata
    recommendations_df = recommendations_df.merge(
        movie_metadata[
            [
                "movieId",
                "title",
                "genres",
                "tmdbId"
            ]
        ],
        on="movieId",
        how="left"
    )

    return recommendations_df


# ============================================================
# SEARCH SECTION
# ============================================================

st.header("🔎 Find a Movie")

search_text = st.text_input(
    "Enter movie name",
    placeholder="e.g. Spider-Man, Toy Story, Titanic..."
)


# ============================================================
# SEARCH RESULTS
# ============================================================

matching_movies = search_movies(
    search_text
)


if search_text:

    st.write(
        f"Found {len(matching_movies)} matching movie(s)."
    )


# ============================================================
# MOVIE SELECTION
# ============================================================

if not matching_movies.empty:

    movie_options = {}

    for _, row in matching_movies.iterrows():

        movie_options[
            f"{row['title']}"
        ] = row["movieId"]

    selected_title = st.selectbox(
        "Select a movie",
        list(movie_options.keys())
    )

    selected_movie_id = movie_options[
        selected_title
    ]

    selected_movie = matching_movies[
        matching_movies["movieId"]
        == selected_movie_id
    ].iloc[0]


    st.info(
        f"Selected: {selected_movie['title']}"
    )


    # ========================================================
    # NUMBER OF RECOMMENDATIONS
    # ========================================================

    number_of_recommendations = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10
    )


    # ========================================================
    # RECOMMEND BUTTON
    # ========================================================

    if st.button(
        "🎯 Recommend Similar Movies",
        type="primary"
    ):

        with st.spinner(
            "Finding similar movies..."
        ):

            recommendations = recommend_similar_movies(
                selected_movie_id,
                number_of_recommendations
            )


        if recommendations.empty:

            st.error(
                "No recommendations found for this movie."
            )

        else:

            st.success(
                f"Found {len(recommendations)} similar movies."
            )


            # ==================================================
            # SELECTED MOVIE
            # ==================================================

            st.subheader(
                f"🎬 Movies Similar to {selected_movie['title']}"
            )


            # ==================================================
            # DISPLAY RECOMMENDATIONS
            # ==================================================

            for rank, (_, movie) in enumerate(
                recommendations.iterrows(),
                start=1
            ):

                movie_title = movie["title"]

                genres = movie["genres"]

                similarity = movie["similarity"]

                tmdb_id = movie["tmdbId"]


                # Get TMDB information
                tmdb_movie = get_movie_information(
                    tmdb_id,
                    movie_title
                )


                poster_url = None

                overview = None


                if tmdb_movie:

                    poster_url = get_poster_url(
                        tmdb_movie.get(
                            "poster_path"
                        )
                    )

                    overview = tmdb_movie.get(
                        "overview"
                    )


                # ==============================================
                # MOVIE CARD
                # ==============================================

                st.markdown(
                    '<div class="movie-card">',
                    unsafe_allow_html=True
                )


                col_poster, col_info, col_score = st.columns(
                    [1, 4, 1]
                )


                # ==============================================
                # POSTER
                # ==============================================

                with col_poster:

                    if poster_url:

                        st.image(
                            poster_url,
                            width=140
                        )

                    else:

                        st.markdown(
                            "🎬\n\nPoster unavailable"
                        )


                # ==============================================
                # MOVIE INFORMATION
                # ==============================================

                with col_info:

                    st.markdown(
                        f"""
                        <div class="movie-title">
                        {rank}. {movie_title}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    st.markdown(
                        f"""
                        <div class="genre">
                        Genres: {genres}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    if overview:

                        st.markdown(
                            f"""
                            <div class="description">
                            {overview}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            """
                            <div class="description">
                            Description unavailable.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                # ==============================================
                # SIMILARITY
                # ==============================================

                with col_score:

                    st.markdown(
                        "Similarity"
                    )

                    st.markdown(
                        f"""
                        <div class="similarity">
                        {similarity:.4f}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


else:

    if search_text:

        st.warning(
            "No matching movies found. "
            "Try another movie title."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Movie Recommendation System — "
    "Item-Based Collaborative Filtering"
)
