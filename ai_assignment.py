import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import html
import os
import re
import random

from sklearn.metrics.pairwise import cosine_similarity
from streamlit_searchbox import st_searchbox
from scipy.sparse import load_npz


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 30px 0 20px 0;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 750;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #a7adb8;
        margin-bottom: 5px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .movie-section-title {
        font-size: 23px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .movie-card-title {
        font-size: 14px;
        font-weight: 600;
        margin-top: 7px;
        line-height: 1.3;
        min-height: 36px;
    }

    .movie-card-year {
        font-size: 12px;
        color: #8f96a3;
        margin-top: 3px;
        min-height: 16px;
    }

    .method-description {
        color: #9da3ae;
        font-size: 14px;
        margin-top: -5px;
        margin-bottom: 15px;
    }

    .recommendation-card {
        background: #151820;
        border: 1px solid #252b36;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 15px;
    }

    .recommendation-title {
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .recommendation-genres {
        color: #9da3ae;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .recommendation-description {
        color: #c5c9d0;
        font-size: 14px;
        line-height: 1.5;
    }

    .rank {
        font-size: 18px;
        font-weight: 700;
        color: #ff4b4b;
    }

    .selected-movie-card {
        background: #151820;
        border: 1px solid #252b36;
        border-radius: 14px;
        padding: 18px;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .selected-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .selected-genres {
        color: #9da3ae;
        font-size: 14px;
    }

    .footer {
        text-align: center;
        color: #747b87;
        font-size: 13px;
        padding: 30px 0 10px 0;
    }

        /* ========================================================
       NETFLIX STYLE HERO BANNER
       ======================================================== */

    .netflix-hero {
        position: relative;
        width: 100%;
        height: 520px;
        border-radius: 18px;
        overflow: hidden;
        margin: 10px 0 18px 0;
        background: #11141b;
        border: 1px solid #252b36;
    }
    
    .netflix-hero-background {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center;
    }

    .netflix-hero-overlay {
        position: absolute;
        inset: 0;
        background:
            linear-gradient(
                90deg,
                rgba(10, 12, 16, 0.96) 0%,
                rgba(10, 12, 16, 0.82) 32%,
                rgba(10, 12, 16, 0.40) 62%,
                rgba(10, 12, 16, 0.15) 100%
            ),
            linear-gradient(
                0deg,
                rgba(10, 12, 16, 0.95) 0%,
                rgba(10, 12, 16, 0.05) 45%
            );
    }

    .netflix-hero-content {
        position: relative;
        z-index: 2;
        height: 520px;
        width: 62%;
        padding: 70px 0 60px 45px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .netflix-hero-label {
        font-size: 13px;
        font-weight: 700;
        color: #d9dce2;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .netflix-hero-title {
        font-size: 48px;
        line-height: 1.08;
        font-weight: 800;
        color: white;
        margin-bottom: 12px;
    }

    .netflix-hero-meta {
        font-size: 14px;
        color: #d1d5dc;
        margin-bottom: 18px;
    }

    .netflix-hero-overview {
        font-size: 15px;
        line-height: 1.55;
        color: #d5d8de;
        max-width: 560px;
        margin-bottom: 18px;
    }

    .netflix-hero-fade-bottom {
        position: absolute;
        left: 0;
        right: 0;
        bottom: 0;
        height: 90px;
        z-index: 2;
        background: linear-gradient(
            transparent,
            #0e1015
        );
        pointer-events: none;
    }

    .hero-button-row {
        margin-top: -65px;
        position: relative;
        z-index: 5;
        margin-bottom: 18px;
    }

    .hero-info-label {
        color: #9da3ae;
        font-size: 12px;
        margin-top: 4px;
    }

    @media (max-width: 900px) {

        .netflix-hero {
            min-height: 390px;
        }

        .netflix-hero-content {
            min-height: 390px;
            width: 75%;
            padding: 55px 25px 45px 25px;
        }

        .netflix-hero-title {
            font-size: 32px;
        }

        .netflix-hero-overview {
            font-size: 13px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None

if "searchbox_version" not in st.session_state:
    st.session_state.searchbox_version = 0

if "recommend_clicked" not in st.session_state:
    st.session_state.recommend_clicked = False

if "selected_recommendation_method" not in st.session_state:
    st.session_state.selected_recommendation_method = (
        "🤝 Collaborative Filtering"
    )

if "recommendation_results" not in st.session_state:
    st.session_state.recommendation_results = None

if "results_method" not in st.session_state:
    st.session_state.results_method = (
        "🤝 Collaborative Filtering"
    )

if "page" not in st.session_state:
    st.session_state.page = "home"

if "details_movie_id" not in st.session_state:
    st.session_state.details_movie_id = None

if "details_tmdb_id" not in st.session_state:
    st.session_state.details_tmdb_id = None

if "details_title" not in st.session_state:
    st.session_state.details_title = ""

if "details_genres" not in st.session_state:
    st.session_state.details_genres = ""

if "details_return_page" not in st.session_state:
    st.session_state.details_return_page = "home"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            🎬 Movie Recommendation System
        </div>
        <div class="hero-subtitle">
            Discover movies using intelligent recommendation techniques
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD COLLABORATIVE FILTERING MODEL
# ============================================================

def get_model_version():

    model_files = [
        "models/item_distances.npy",
        "models/item_indices.npy",
        "models/movie_to_index.pkl",
        "models/index_to_movie.pkl",
        "models/movie_metadata.csv"
    ]

    return tuple(
        os.path.getmtime(file)
        for file in model_files
    )


@st.cache_resource
def load_cf_model(model_version):

    item_distances = np.load(
        "models/item_distances.npy"
    )

    item_indices = np.load(
        "models/item_indices.npy"
    )

    with open(
        "models/movie_to_index.pkl",
        "rb"
    ) as f:
        movie_to_index = pickle.load(f)

    with open(
        "models/index_to_movie.pkl",
        "rb"
    ) as f:
        index_to_movie = pickle.load(f)

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
    ) = load_cf_model(
        get_model_version()
    )

except Exception as e:

    st.error(
        f"Unable to load recommendation model: {e}"
    )

    st.stop()


# ============================================================
# LOAD LINKS
# ============================================================

def load_links():

    links = pd.read_csv(
        "data/links.csv"
    )

    links["movieId"] = pd.to_numeric(
        links["movieId"],
        errors="coerce"
    )

    links["tmdbId"] = pd.to_numeric(
        links["tmdbId"],
        errors="coerce"
    )

    return links


try:

    links = load_links()

except Exception as e:

    st.error(
        f"Unable to load links.csv: {e}"
    )

    st.stop()


# ============================================================
# ADD TMDB ID TO COLLABORATIVE METADATA
# ============================================================

if "tmdbId" not in movie_metadata.columns:

    movie_metadata = movie_metadata.merge(
        links[
            [
                "movieId",
                "tmdbId"
            ]
        ],
        on="movieId",
        how="left"
    )


# ============================================================
# TMDB TOKEN
# ============================================================

TMDB_TOKEN = None

try:

    TMDB_TOKEN = st.secrets["TMDB_TOKEN"]

except Exception:

    TMDB_TOKEN = None


# ============================================================
# TMDB API
# ============================================================

@st.cache_data(
    show_spinner=False
)
def get_tmdb_movie(tmdb_id):

    if tmdb_id is None:
        return None

    if pd.isna(tmdb_id):
        return None

    if TMDB_TOKEN is None:
        return None

    try:

        tmdb_id = int(tmdb_id)

    except Exception:

        return None

    headers = {
        "Authorization": f"Bearer {TMDB_TOKEN}",
        "accept": "application/json"
    }

    # --------------------------------------------------------
    # Try Movie
    # --------------------------------------------------------

    movie_url = (
        f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    )

    try:

        movie_response = requests.get(
            movie_url,
            headers=headers,
            timeout=10
        )

    except Exception:

        movie_response = None

    if (
        movie_response is not None
        and movie_response.status_code == 200
    ):

        data = movie_response.json()

        data["_media_type"] = "movie"

        return data

    # --------------------------------------------------------
    # Try TV
    # --------------------------------------------------------

    tv_url = (
        f"https://api.themoviedb.org/3/tv/{tmdb_id}"
    )

    try:

        tv_response = requests.get(
            tv_url,
            headers=headers,
            timeout=10
        )

    except Exception:

        tv_response = None

    if (
        tv_response is not None
        and tv_response.status_code == 200
    ):

        data = tv_response.json()

        data["title"] = data.get(
            "name",
            data.get(
                "original_name",
                ""
            )
        )

        data["release_date"] = data.get(
            "first_air_date",
            ""
        )

        data["_media_type"] = "tv"

        return data

    return None


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
# SEARCH FUNCTION
# ============================================================

def search_movies(search_text):

    if not search_text:
        return []

    search_text = (
        str(search_text)
        .strip()
        .lower()
    )

    if not search_text:
        return []

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

    matches = matches.head(20)

    results = []

    for _, movie in matches.iterrows():

        title = str(
            movie["title"]
        )

        genres = str(
            movie.get(
                "genres",
                ""
            )
        )

        display_text = format_movie_title(
            title
        )

        if (
            genres
            and genres != "nan"
        ):

            display_text = (
                f"{display_text}  ·  {genres}"
            )

        results.append(
            (
                display_text,
                int(movie["movieId"])
            )
        )

    return results


# ============================================================
# COLLABORATIVE FILTERING
# ============================================================

def recommend_cf(
    movie_id,
    n=10
):

    if movie_id is None:
        return pd.DataFrame()

    if movie_id not in movie_to_index:
        return pd.DataFrame()

    movie_index = movie_to_index[
        movie_id
    ]

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

        similar_index = int(
            similar_index
        )

        if similar_index == movie_index:
            continue

        try:

            similar_movie_id = (
                index_to_movie[
                    similar_index
                ]
            )

        except Exception:

            continue

        recommendations.append(
            {
                "movieId": similar_movie_id
            }
        )

        if len(recommendations) >= n:
            break

    recommendations_df = pd.DataFrame(
        recommendations
    )

    if recommendations_df.empty:
        return recommendations_df

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

    return recommendations_df.head(n)


# ============================================================
# CONTENT-BASED MODEL
# ============================================================

CONTENT_MODEL_DIR = "content_models"

GENRE_WEIGHT = 0.1
TAG_WEIGHT = 0.9


def get_content_model_version():

    model_files = [
        "content_models/genre_matrix.npz",
        "content_models/tag_matrix.npz",
        "content_models/movie_to_index.pkl",
        "content_models/index_to_movie.pkl",
        "content_models/movie_metadata.csv"
    ]

    return tuple(
        os.path.getmtime(file)
        for file in model_files
    )


@st.cache_resource
def load_content_model(
    model_version
):

    genre_matrix = load_npz(
        os.path.join(
            CONTENT_MODEL_DIR,
            "genre_matrix.npz"
        )
    )

    tag_matrix = load_npz(
        os.path.join(
            CONTENT_MODEL_DIR,
            "tag_matrix.npz"
        )
    )

    with open(
        os.path.join(
            CONTENT_MODEL_DIR,
            "movie_to_index.pkl"
        ),
        "rb"
    ) as f:

        content_movie_to_index = pickle.load(f)

    with open(
        os.path.join(
            CONTENT_MODEL_DIR,
            "index_to_movie.pkl"
        ),
        "rb"
    ) as f:

        content_index_to_movie = pickle.load(f)

    content_metadata = pd.read_csv(
        os.path.join(
            CONTENT_MODEL_DIR,
            "movie_metadata.csv"
        )
    )

    return (
        genre_matrix,
        tag_matrix,
        content_movie_to_index,
        content_index_to_movie,
        content_metadata
    )


try:

    (
        content_genre_matrix,
        content_tag_matrix,
        content_movie_to_index,
        content_index_to_movie,
        content_metadata
    ) = load_content_model(
        get_content_model_version()
    )

except Exception as e:

    st.error(
        f"Unable to load Content-Based model: {e}"
    )

    st.stop()


# ============================================================
# CONTENT-BASED RECOMMENDATION
# ============================================================

def recommend_content(
    movie_id,
    n=10
):

    if movie_id is None:
        return pd.DataFrame()

    if movie_id not in content_movie_to_index:
        return pd.DataFrame()

    movie_index = (
        content_movie_to_index[
            movie_id
        ]
    )

    # --------------------------------------------------------
    # Genre Similarity
    # --------------------------------------------------------

    genre_similarity = cosine_similarity(
        content_genre_matrix[
            movie_index
        ],
        content_genre_matrix
    ).flatten()

    # --------------------------------------------------------
    # Tag Similarity
    # --------------------------------------------------------

    tag_similarity = cosine_similarity(
        content_tag_matrix[
            movie_index
        ],
        content_tag_matrix
    ).flatten()

    # --------------------------------------------------------
    # Weighted Similarity
    # Genre = 10%
    # Tag   = 90%
    # --------------------------------------------------------

    final_similarity = (
        GENRE_WEIGHT
        * genre_similarity
        +
        TAG_WEIGHT
        * tag_similarity
    )

    final_similarity[
        movie_index
    ] = -1

    # --------------------------------------------------------
    # Top N
    # --------------------------------------------------------

    top_indices = np.argsort(
        final_similarity
    )[::-1][:n]

    recommendations = (
        content_metadata
        .iloc[
            top_indices
        ][
            [
                "movieId",
                "title",
                "genres"
            ]
        ]
        .copy()
    )

    recommendations[
        "similarity"
    ] = np.round(
        final_similarity[
            top_indices
        ],
        6
    )

    # --------------------------------------------------------
    # Add TMDB ID
    # --------------------------------------------------------

    recommendations = recommendations.merge(
        links[
            [
                "movieId",
                "tmdbId"
            ]
        ],
        on="movieId",
        how="left"
    )

    recommendations.reset_index(
        drop=True,
        inplace=True
    )

    return recommendations


# ============================================================
# MOVIE TITLE FUNCTIONS
# ============================================================

def get_year(title):

    title = str(title).strip()

    match = re.search(
        r"\((\d{4})\)\s*$",
        title
    )

    if match:
        return match.group(1)

    return ""


def format_movie_title(title):

    title = str(title).strip()

    # --------------------------------------------------------
    # Convert MovieLens article suffix:
    #
    # "Godfather, The (1972)"
    # -> "The Godfather (1972)"
    #
    # "Lord of the Rings: The Fellowship of the Ring, The (2001)"
    # -> "The Lord of the Rings: The Fellowship of the Ring (2001)"
    # --------------------------------------------------------

    # Normalize whitespace first.
    title = re.sub(r"\s+", " ", title)

    pattern = (
        r"^(.*?),\s*"
        r"(The|A|An)\s*"
        r"(\(\d{4}\))$"
    )

    match = re.match(
        pattern,
        title
    )

    if match:

        main_title = (
            match.group(1)
            .strip()
        )

        article = (
            match.group(2)
            .strip()
        )

        year = (
            match.group(3)
            .strip()
        )

        return (
            f"{article} "
            f"{main_title} "
            f"{year}"
        )

    return title


# ============================================================
# GENRE MOVIES
# ============================================================

def get_genre_movies(
    genre,
    n=8
):

    data = movie_metadata.copy()

    data["genres"] = (
        data["genres"]
        .fillna("")
        .astype(str)
    )

    matches = data[
        data["genres"]
        .str.contains(
            genre,
            case=False,
            regex=False,
            na=False
        )
    ].copy()

    return matches.head(n)


# ============================================================
# SHOW HOMEPAGE MOVIE ROW
# ============================================================

def show_movie_row(
    section_title,
    movies
):

    if movies.empty:
        return

    # ========================================================
    # SECTION TITLE
    # ========================================================

    st.markdown(
        f"""
        <div class="movie-section-title">
            {html.escape(section_title)}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # HORIZONTAL NETFLIX-STYLE CONTAINER
    # ========================================================

    with st.container(
        horizontal=True,
        wrap=False,
        gap="small"
    ):

        for _, movie in movies.iterrows():

            movie_id = int(
                movie["movieId"]
            )

            title = format_movie_title(
                str(movie["title"])
            )

            tmdb_id = movie.get(
                "tmdbId"
            )

            genres = str(
                movie.get(
                    "genres",
                    ""
                )
            )

            # ------------------------------------------------
            # MOVIE CARD
            # ------------------------------------------------

            with st.container(
                width=155
            ):

                # ------------------------------------------------
                # POSTER
                # ------------------------------------------------

                tmdb_movie = get_tmdb_movie(
                    tmdb_id
                )

                poster_url = None

                if tmdb_movie:

                    poster_url = get_poster_url(
                        tmdb_movie.get(
                            "poster_path"
                        )
                    )

                if poster_url:

                    st.image(
                        poster_url,
                        width=155
                    )

                else:

                    st.markdown(
                        """
                        <div style="
                            width:155px;
                            height:225px;
                            background:#151820;
                            border:1px solid #252b36;
                            border-radius:7px;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            color:#9da3ae;
                            text-align:center;
                        ">
                            🎬<br>
                            Poster unavailable
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ------------------------------------------------
                # TITLE
                # ------------------------------------------------

                st.markdown(
                    f"""
                    <div class="movie-card-title">
                        {html.escape(title)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ------------------------------------------------
                # YEAR
                # ------------------------------------------------

                year = get_year(
                    title
                )

                st.markdown(
                    f"""
                    <div class="movie-card-year">
                        {html.escape(year)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ------------------------------------------------
                # VIEW DETAILS
                # ------------------------------------------------

                if st.button(
                    "View Details",
                    key=(
                        f"home_details_"
                        f"{section_title}_"
                        f"{movie_id}"
                    ),
                    use_container_width=True
                ):

                    st.session_state.details_movie_id = (
                        movie_id
                    )

                    st.session_state.details_tmdb_id = (
                        tmdb_id
                    )

                    st.session_state.details_title = (
                        title
                    )

                    st.session_state.details_genres = (
                        genres
                    )

                    st.session_state.details_return_page = (
                        "home"
                    )
                # Dialog opens at the end of this run; no rerun.

# ============================================================
# NETFLIX STYLE HERO
# ============================================================

def show_hero_movie():

    if movie_metadata.empty:
        return

    # ========================================================
    # RANDOM HERO MOVIE
    # Select from the ENTIRE movie dataset.
    # The selected movie stays stable during the current
    # Streamlit session.
    # ========================================================

    hero_candidates = movie_metadata[
        movie_metadata["title"].notna()
    ].copy()

    if hero_candidates.empty:
        return

    # --------------------------------------------------------
    # Select a random movie only when there is no current Hero
    # --------------------------------------------------------

    if (
        "hero_movie_id" not in st.session_state
        or st.session_state.hero_movie_id is None
    ):

        hero_movie = hero_candidates.sample(
            n=1
        ).iloc[0]

        st.session_state.hero_movie_id = int(
            hero_movie["movieId"]
        )

    # --------------------------------------------------------
    # Get the saved Hero movie
    # --------------------------------------------------------

    hero_matches = hero_candidates[
        hero_candidates["movieId"]
        == st.session_state.hero_movie_id
    ]

    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    if hero_matches.empty:

        hero_movie = hero_candidates.sample(
            n=1
        ).iloc[0]

        st.session_state.hero_movie_id = int(
            hero_movie["movieId"]
        )

    else:

        hero_movie = hero_matches.iloc[0]

    # ========================================================
    # HERO MOVIE INFORMATION
    # ========================================================

    hero_movie_id = int(
        hero_movie["movieId"]
    )

    hero_title = format_movie_title(
        str(hero_movie["title"])
    )

    hero_genres = str(
        hero_movie.get(
            "genres",
            ""
        )
    )

    hero_tmdb_id = hero_movie.get(
        "tmdbId"
    )

    tmdb_movie = get_tmdb_movie(
        hero_tmdb_id
    )

    backdrop_url = None
    hero_overview = None
    hero_release_date = ""

    # ========================================================
    # TMDB INFORMATION
    # ========================================================

    if tmdb_movie:

        backdrop_path = tmdb_movie.get(
            "backdrop_path"
        )

        if backdrop_path:

            backdrop_url = (
                "https://image.tmdb.org/t/p/original"
                + backdrop_path
            )

        hero_overview = (
            tmdb_movie.get(
                "overview"
            )
        )

        hero_release_date = (
            tmdb_movie.get(
                "release_date",
                ""
            )
        )

        if not hero_release_date:

            hero_release_date = (
                tmdb_movie.get(
                    "first_air_date",
                    ""
                )
            )

    # ========================================================
    # FALLBACK TO POSTER
    # ========================================================

    if not backdrop_url:

        if tmdb_movie:

            backdrop_url = get_poster_url(
                tmdb_movie.get(
                    "poster_path"
                )
            )

    # ========================================================
    # YEAR
    # ========================================================

    hero_year = get_year(
        hero_title
    )

    if not hero_year and hero_release_date:

        hero_year = str(
            hero_release_date
        )[:4]

    if hero_genres == "nan":

        hero_genres = ""

    # ========================================================
    # OVERVIEW
    # ========================================================

    if hero_overview:

        hero_overview = str(
            hero_overview
        ).strip()

        if len(hero_overview) > 360:

            hero_overview = (
                hero_overview[:360]
                + "..."
            )

    else:

        hero_overview = (
            "Discover this movie and explore "
            "similar recommendations."
        )

    # ========================================================
    # HERO HTML
    # ========================================================

    background_html = ""

    if backdrop_url:

        background_html = (
            '<img '
            'class="netflix-hero-background" '
            'src="'
            + html.escape(
                backdrop_url,
                quote=True
            )
            + '" '
            'alt="'
            + html.escape(
                hero_title,
                quote=True
            )
            + '">'
        )

    hero_html = (
        '<div class="netflix-hero">'
        + background_html
        + '<div class="netflix-hero-overlay"></div>'
        + '<div class="netflix-hero-content">'

        + '<div class="netflix-hero-label">'
        + '⭐ Featured Movie'
        + '</div>'

        + '<div class="netflix-hero-title">'
        + html.escape(hero_title)
        + '</div>'

        + '<div class="netflix-hero-meta">'
        + html.escape(hero_year)

        + (
            ' &nbsp;•&nbsp; '
            if hero_year and hero_genres
            else ''
        )

        + html.escape(hero_genres)
        + '</div>'

        + '<div class="netflix-hero-overview">'
        + html.escape(hero_overview)
        + '</div>'

        + '</div>'

        + '<div class="netflix-hero-fade-bottom"></div>'

        + '</div>'
    )

    # ========================================================
    # DISPLAY HERO
    # ========================================================

    st.markdown(
        hero_html,
        unsafe_allow_html=True
    )

    # ========================================================
    # HERO BUTTONS
    # ========================================================

    st.markdown(
        '<div class="hero-button-row">',
        unsafe_allow_html=True
    )

    hero_col1, hero_col2, hero_col3 = st.columns(
        [1.2, 1.2, 7]
    )

    # ========================================================
    # RECOMMEND SIMILAR
    # ========================================================

    with hero_col1:

        if st.button(
            "✨ Recommend Similar",
            key="hero_recommend_button"
        ):

            hero_method = st.session_state.get(
                "selected_recommendation_method",
                "🤝 Collaborative Filtering"
            )

            st.session_state.selected_movie_id = (
                hero_movie_id
            )

            st.session_state.selected_recommendation_method = (
                hero_method
            )

            st.session_state.results_method = (
                hero_method
            )

            with st.spinner(
                "Generating recommendations..."
            ):

                if hero_method == (
                    "🤝 Collaborative Filtering"
                ):

                    hero_recommendations = (
                        recommend_cf(
                            hero_movie_id,
                            n=10
                        )
                    )

                else:

                    hero_recommendations = (
                        recommend_content(
                            hero_movie_id,
                            n=10
                        )
                    )

            st.session_state.recommendation_results = (
                hero_recommendations
            )

            st.session_state.recommend_clicked = True

            st.session_state.page = "results"

            st.rerun()

    # ========================================================
    # MORE INFO
    # ========================================================

    with hero_col2:

        if st.button(
            "ⓘ More Info",
            key="hero_details_button"
        ):

            st.session_state.details_movie_id = (
                hero_movie_id
            )

            st.session_state.details_tmdb_id = (
                hero_tmdb_id
            )

            st.session_state.details_title = (
                hero_title
            )

            st.session_state.details_genres = (
                hero_genres
            )

            st.session_state.details_return_page = (
                "home"
            )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-info-label">'
        'Featured movie selected from the movie dataset'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # HERO HTML
    # ========================================================

    background_html = ""

    if backdrop_url:

        background_html = (
            '<img '
            'class="netflix-hero-background" '
            'src="'
            + html.escape(
                backdrop_url,
                quote=True
            )
            + '" '
            'alt="'
            + html.escape(
                hero_title,
                quote=True
            )
            + '">'
        )

    hero_html = (
        '<div class="netflix-hero">'
        + background_html
        + '<div class="netflix-hero-overlay"></div>'
        + '<div class="netflix-hero-content">'

        + '<div class="netflix-hero-label">'
        + '⭐ Featured Movie'
        + '</div>'

        + '<div class="netflix-hero-title">'
        + html.escape(hero_title)
        + '</div>'

        + '<div class="netflix-hero-meta">'
        + html.escape(hero_year)

        + (
            ' &nbsp;•&nbsp; '
            if hero_year and hero_genres
            else ''
        )

        + html.escape(hero_genres)
        + '</div>'

        + '<div class="netflix-hero-overview">'
        + html.escape(hero_overview)
        + '</div>'

        + '</div>'

        + '<div class="netflix-hero-fade-bottom"></div>'

        + '</div>'
    )

    # ========================================================
    # IMPORTANT
    # Do NOT indent the HTML string.
    # This prevents Streamlit from interpreting it as code.
    # ========================================================

    st.markdown(
        hero_html,
        unsafe_allow_html=True
    )

    # ========================================================
    # HERO BUTTONS
    # ========================================================
    st.markdown(
        '<div class="hero-button-row">',
        unsafe_allow_html=True
    )
    
    hero_col1, hero_col2, hero_col3 = st.columns(
        [1.2, 1.2, 7]
    )
    with hero_col1:

        if st.button(
            "✨ Recommend Similar",
            key="hero_recommend_button"
        ):

            hero_method = st.session_state.get(
                "selected_recommendation_method",
                "🤝 Collaborative Filtering"
            )

            st.session_state.selected_movie_id = (
                hero_movie_id
            )

            st.session_state.selected_recommendation_method = (
                hero_method
            )

            st.session_state.results_method = (
                hero_method
            )

            with st.spinner(
                "Generating recommendations..."
            ):

                if hero_method == (
                    "🤝 Collaborative Filtering"
                ):

                    hero_recommendations = (
                        recommend_cf(
                            hero_movie_id,
                            n=10
                        )
                    )

                else:

                    hero_recommendations = (
                        recommend_content(
                            hero_movie_id,
                            n=10
                        )
                    )

            st.session_state.recommendation_results = (
                hero_recommendations
            )

            st.session_state.recommend_clicked = True

            st.session_state.page = "results"

            st.rerun()

    with hero_col2:

        if st.button(
            "ⓘ More Info",
            key="hero_details_button"
        ):

            st.session_state.details_movie_id = (
                hero_movie_id
            )

            st.session_state.details_tmdb_id = (
                hero_tmdb_id
            )

            st.session_state.details_title = (
                hero_title
            )

            st.session_state.details_genres = (
                hero_genres
            )

            st.session_state.details_return_page = (
                "home"
            )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-info-label">'
        'Featured movie selected from the movie dataset'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# HOMEPAGE MOVIE DATA
# ============================================================

featured_movies = (
    movie_metadata
    .dropna(
        subset=["title"]
    )
    .head(8)
)

action_movies = get_genre_movies(
    "Action",
    8
)

comedy_movies = get_genre_movies(
    "Comedy",
    8
)

drama_movies = get_genre_movies(
    "Drama",
    8
)

sci_fi_movies = get_genre_movies(
    "Sci-Fi",
    8
)


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    # ========================================================
    # NETFLIX STYLE HERO
    # ========================================================

    show_hero_movie()

    # ========================================================
    # RECOMMENDATION METHOD
    # ========================================================

    st.divider()

    # ========================================================
    # RECOMMENDATION METHOD
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🧠 Recommendation Method'
        '</div>',
        unsafe_allow_html=True
    )

    method = st.radio(
        "Choose how recommendations should be generated:",
        [
            "🤝 Collaborative Filtering",
            "🎯 Content-Based Filtering"
        ],
        horizontal=True,
        key="recommendation_method"
    )

    # Keep current method synchronized
    st.session_state.selected_recommendation_method = (
        method
    )

    if method == "🤝 Collaborative Filtering":

        st.markdown(
            '<div class="method-description">'
            'Recommends movies based on relationships between '
            'movies derived from user rating behaviour.'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="method-description">'
            'Recommends movies based on content similarity using '
            'movie genres and user-generated tags.'
            '</div>',
            unsafe_allow_html=True
        )

    # ========================================================
    # FIND A MOVIE
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🔎 Find a Movie'
        '</div>',
        unsafe_allow_html=True
    )

    selected_search_movie = st_searchbox(
        search_movies,
        placeholder=(
            "Search for a movie, e.g. Spider-Man, "
            "Toy Story, Titanic..."
        ),
        label=None,
        key=f"movie_searchbox_{st.session_state.searchbox_version}",
        debounce=200,
        rerun_on_update=True,
        clear_on_submit=False,
        edit_after_submit="option"
    )

    # ========================================================
    # HANDLE MOVIE SELECTION
    # ========================================================

    if selected_search_movie is not None:

        try:

            selected_movie_id = int(
                selected_search_movie
            )

            if (
                selected_movie_id
                != st.session_state.selected_movie_id
            ):

                st.session_state.selected_movie_id = (
                    selected_movie_id
                )

                st.session_state.recommend_clicked = (
                    False
                )

                st.session_state.recommendation_results = (
                    None
                )

        except Exception:

            pass

    # ========================================================
    # SHOW SELECTED MOVIE
    # ========================================================

    if st.session_state.selected_movie_id is not None:

        selected_matches = movie_metadata[
            movie_metadata["movieId"]
            == st.session_state.selected_movie_id
        ]

        if not selected_matches.empty:

            selected_movie = (
                selected_matches.iloc[0]
            )

            selected_title = format_movie_title(
                str(selected_movie["title"])
            )

            selected_genres = str(
                selected_movie.get(
                    "genres",
                    ""
                )
            )

            st.markdown(
                f'<div class="selected-movie-card">'
                f'<div class="selected-title">'
                f'🎬 {html.escape(selected_title)}'
                f'</div>'
                f'<div class="selected-genres">'
                f'{html.escape(selected_genres)}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ========================================================
    # RECOMMEND BUTTON
    # ========================================================

    if st.button(
        "✨ Recommend Top 10 Movies",
        type="primary",
        use_container_width=True,
        key="recommend_button"
    ):

        # ----------------------------------------------------
        # Validate selection
        # ----------------------------------------------------

        if st.session_state.selected_movie_id is None:

            st.warning(
                "Please search and select a movie first."
            )

        else:

            # ------------------------------------------------
            # LOCK METHOD
            # ------------------------------------------------

            st.session_state.selected_recommendation_method = (
                method
            )

            st.session_state.results_method = (
                method
            )

            # ------------------------------------------------
            # GENERATE RECOMMENDATIONS
            # ------------------------------------------------

            with st.spinner(
                "Generating recommendations..."
            ):

                if method == "🤝 Collaborative Filtering":

                    recommendations = recommend_cf(
                        st.session_state.selected_movie_id,
                        n=10
                    )

                else:

                    recommendations = recommend_content(
                        st.session_state.selected_movie_id,
                        n=10
                    )

            # ------------------------------------------------
            # SAVE RESULTS
            # ------------------------------------------------

            st.session_state.recommendation_results = (
                recommendations
            )

            st.session_state.recommend_clicked = True

            # ------------------------------------------------
            # GO TO RESULTS PAGE
            # ------------------------------------------------

            st.session_state.page = "results"

            st.rerun()

    # ========================================================
    # NETFLIX-STYLE MOVIE ROWS
    # ========================================================

    st.divider()

    show_movie_row(
        "⭐ Featured Movies",
        featured_movies
    )

    show_movie_row(
        "🔥 Action",
        action_movies
    )

    show_movie_row(
        "😂 Comedy",
        comedy_movies
    )

    show_movie_row(
        "🎭 Drama",
        drama_movies
    )

    show_movie_row(
        "👽 Sci-Fi",
        sci_fi_movies
    )


# ============================================================
# RESULTS PAGE
# ============================================================

if (
    st.session_state.page == "results"
    and st.session_state.recommend_clicked
    and st.session_state.selected_movie_id is not None
):

    # ========================================================
    # TOP NAVIGATION
    # ========================================================

    if st.button(
        "🏠 Back to Home",
        key="back_to_home",
        use_container_width=False
    ):
    
        # ========================================================
        # CLEAR SELECTED MOVIE
        # ========================================================
    
        st.session_state.selected_movie_id = None

        st.session_state.hero_movie_id = None
    
        # ========================================================
        # CLEAR RECOMMENDATION RESULTS
        # ========================================================
    
        st.session_state.recommend_clicked = False
    
        st.session_state.recommendation_results = None
    
        # ========================================================
        # RESET RECOMMENDATION METHOD
        # ========================================================
    
        st.session_state.results_method = (
            "🤝 Collaborative Filtering"
        )
    
        st.session_state.selected_recommendation_method = (
            "🤝 Collaborative Filtering"
        )
    
        # ========================================================

        # Reset the actual radio widget state too.
        st.session_state.recommendation_method = (
            "🤝 Collaborative Filtering"
        )

        # Clear temporary movie-detail state.
        st.session_state.details_movie_id = None
        st.session_state.details_tmdb_id = None
        st.session_state.details_title = ""
        st.session_state.details_genres = ""
        st.session_state.details_return_page = "home"

        # RESET SEARCHBOX
        # ========================================================
    
        st.session_state.searchbox_version += 1
    
        # ========================================================
        # RETURN TO HOME
        # ========================================================
    
        st.session_state.page = "home"
    
        st.rerun()

    st.divider()

    # ========================================================
    # SELECTED MOVIE
    # ========================================================

    selected_movie_id = (
        st.session_state.selected_movie_id
    )

    selected_matches = movie_metadata[
        movie_metadata["movieId"]
        == selected_movie_id
    ]

    if not selected_matches.empty:

        selected_movie = (
            selected_matches.iloc[0]
        )

        selected_display_title = format_movie_title(
            str(selected_movie["title"])
        )

        selected_display_genres = str(
            selected_movie.get(
                "genres",
                ""
            )
        )

        st.markdown(
            f'<div class="selected-movie-card">'
            f'<div class="selected-title">'
            f'🎬 {html.escape(selected_display_title)}'
            f'</div>'
            f'<div class="selected-genres">'
            f'{html.escape(selected_display_genres)}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ========================================================
    # LOAD SAVED METHOD
    # ========================================================

    method = st.session_state.get(
        "results_method",
        "🤝 Collaborative Filtering"
    )

    # IMPORTANT:
    # Do NOT regenerate recommendations here.
    # The recommendation method is locked when the user
    # clicked the Recommend button.

    recommendations = (
        st.session_state.get(
            "recommendation_results"
        )
    )

    if recommendations is None:

        st.error(
            "Recommendation results are unavailable."
        )

        st.stop()

    # ========================================================
    # SHOW METHOD USED
    # ========================================================

    if method == "🤝 Collaborative Filtering":

        st.caption(
            "Recommendation method: Collaborative Filtering"
        )

    else:

        st.caption(
            "Recommendation method: Content-Based Filtering"
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    if recommendations.empty:

        st.error(
            "Sorry, no recommendations are available "
            "for this movie."
        )

    else:

        recommendations = (
            recommendations
            .head(10)
            .reset_index(drop=True)
        )

        st.success(
            f"{len(recommendations)} "
            "recommendations generated."
        )

        st.markdown(
            '<div class="section-title">'
            '✨ Recommended Movies'
            '</div>',
            unsafe_allow_html=True
        )

        # ====================================================
        # RECOMMENDATION CARDS
        # ====================================================

        for rank, (_, movie) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            movie_id = int(
                movie["movieId"]
            )

            movie_title = format_movie_title(
                str(movie["title"])
            )

            genres = str(
                movie.get(
                    "genres",
                    ""
                )
            )

            tmdb_id = movie.get(
                "tmdbId"
            )

            # ------------------------------------------------
            # TMDB DETAILS
            # ------------------------------------------------

            tmdb_movie = get_tmdb_movie(
                tmdb_id
            )

            poster_url = None
            overview = None

            if tmdb_movie:

                poster_url = get_poster_url(
                    tmdb_movie.get(
                        "poster_path"
                    )
                )

                overview = (
                    tmdb_movie.get(
                        "overview"
                    )
                )

            # ------------------------------------------------
            # CARD
            # ------------------------------------------------

            st.markdown(
                '<div class="recommendation-card">',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(
                [1.2, 5, 1.5]
            )

            # ------------------------------------------------
            # POSTER
            # ------------------------------------------------

            with col1:

                if poster_url:

                    st.image(
                        poster_url,
                        width=120
                    )

                else:

                    st.markdown(
                        """
                        🎬

                        **Poster unavailable**
                        """
                    )

            # ------------------------------------------------
            # INFORMATION
            # ------------------------------------------------

            with col2:

                safe_title = html.escape(
                    movie_title
                )

                safe_genres = html.escape(
                    genres
                )

                st.markdown(
                    f"""
                    <div class="rank">
                        #{rank}
                    </div>

                    <div class="recommendation-title">
                        {safe_title}
                    </div>

                    <div class="recommendation-genres">
                        {safe_genres}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if overview:

                    short_description = (
                        overview[:220]
                        + "..."
                        if len(overview) > 220
                        else overview
                    )

                    st.markdown(
                        f"""
                        <div class="recommendation-description">
                            {html.escape(
                                short_description
                            )}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div class="recommendation-description">
                            Movie description is unavailable.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # ------------------------------------------------
            # DETAILS BUTTON
            # ------------------------------------------------

            with col3:

                st.write("")

                # IMPORTANT:
                # Method is intentionally NOT used in the key.
                # This prevents state/key conflicts between
                # Collaborative and Content-Based results.

                details_key = (
                    f"results_details_{movie_id}"
                )

                if st.button(
                    "View Details",
                    key=details_key,
                    use_container_width=True
                ):

                    st.session_state.details_movie_id = (
                        movie_id
                    )

                    st.session_state.details_tmdb_id = (
                        tmdb_id
                    )

                    st.session_state.details_title = (
                        movie_title
                    )

                    st.session_state.details_genres = (
                        genres
                    )

                    # Remember that the dialog came from
                    # the Results Page.

                    st.session_state.details_return_page = (
                        "results"
                    )

                # Dialog opens at the end of this run; no rerun.

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


# ============================================================
# MOVIE DETAILS DIALOG
# ============================================================

@st.dialog(
    "Movie Details"
)
def show_movie_details():

    movie_id = (
        st.session_state.details_movie_id
    )

    tmdb_id = (
        st.session_state.get(
            "details_tmdb_id"
        )
    )

    fallback_title = (
        st.session_state.get(
            "details_title",
            "Movie"
        )
    )

    fallback_genres = (
        st.session_state.get(
            "details_genres",
            ""
        )
    )

    tmdb_details = get_tmdb_movie(
        tmdb_id
    )

    # ========================================================
    # TMDB AVAILABLE
    # ========================================================

    if tmdb_details:

        poster = get_poster_url(
            tmdb_details.get(
                "poster_path"
            )
        )

        if poster:

            st.image(
                poster,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        tmdb_title = tmdb_details.get(
            "title"
        )

        if not tmdb_title:

            tmdb_title = tmdb_details.get(
                "name",
                fallback_title
            )

        st.subheader(
            tmdb_title
        )

        # ----------------------------------------------------
        # Release Date
        # ----------------------------------------------------

        if (
            tmdb_details.get(
                "_media_type"
            )
            == "tv"
        ):

            release_date = (
                tmdb_details.get(
                    "first_air_date",
                    ""
                )
            )

        else:

            release_date = (
                tmdb_details.get(
                    "release_date",
                    ""
                )
            )

        if release_date:

            st.caption(
                f"Release date: {release_date}"
            )

        # ----------------------------------------------------
        # Genres
        # ----------------------------------------------------

        genres_data = (
            tmdb_details.get(
                "genres",
                []
            )
        )

        if genres_data:

            genre_text = ", ".join(
                genre["name"]
                for genre in genres_data
            )

            st.caption(
                f"Genres: {genre_text}"
            )

        # ----------------------------------------------------
        # Overview
        # ----------------------------------------------------

        overview = (
            tmdb_details.get(
                "overview"
            )
        )

        if overview:

            st.write(
                overview
            )

        else:

            st.info(
                "Description unavailable."
            )

    # ========================================================
    # FALLBACK
    # ========================================================

    else:

        st.subheader(
            fallback_title
        )

        st.caption(
            f"Genres: {fallback_genres}"
        )

        st.info(
            "Additional movie details are "
            "currently unavailable."
        )


# ============================================================
# OPEN DETAILS DIALOG
# ============================================================

if (
    st.session_state.details_movie_id
    is not None
):

    show_movie_details()

    # Clear only the temporary dialog trigger.
    #
    # This does NOT change:
    # - page
    # - results_method
    # - recommendation_results
    # - selected_movie_id

    st.session_state.details_movie_id = None


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Movie Recommendation System ·
        Collaborative Filtering & Content-Based Filtering
    </div>
    """,
    unsafe_allow_html=True
)
