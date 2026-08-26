import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import html
import os

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

    /* ========================================================
       Netflix-style Movie Rows
       ======================================================== */

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
    }

    .movie-card-year {
        font-size: 12px;
        color: #8f96a3;
        margin-top: 3px;
    }

    /* ========================================================
       Method Description
       ======================================================== */

    .method-description {
        color: #9da3ae;
        font-size: 14px;
        margin-top: -5px;
        margin-bottom: 15px;
    }

    /* ========================================================
       Selected Movie
       ======================================================== */

    .selected-card {
        background: linear-gradient(
            135deg,
            #151922,
            #1c2330
        );
        border: 1px solid #2c3442;
        border-radius: 14px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .selected-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .selected-genres {
        color: #9da3ae;
        font-size: 14px;
    }

    /* ========================================================
       Recommendation Cards
       ======================================================== */

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

    /* ========================================================
       Footer
       ======================================================== */

    .footer {
        text-align: center;
        color: #747b87;
        font-size: 13px;
        padding: 30px 0 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


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
# SESSION STATE
# ============================================================

if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None

if "recommend_clicked" not in st.session_state:
    st.session_state.recommend_clicked = False

if "details_movie_id" not in st.session_state:
    st.session_state.details_movie_id = None

if "details_tmdb_id" not in st.session_state:
    st.session_state.details_tmdb_id = None

if "details_title" not in st.session_state:
    st.session_state.details_title = ""

if "details_genres" not in st.session_state:
    st.session_state.details_genres = ""


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
        search_text
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

        display_text = title

        if (
            genres
            and genres != "nan"
        ):

            display_text = (
                f"{title}  ·  {genres}"
            )

        results.append(
            (
                display_text,
                movie["movieId"]
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

        # Do not recommend selected movie
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
    #
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

    # Do not recommend selected movie
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
# HOMEPAGE MOVIE FUNCTIONS
# ============================================================

def get_year(title):

    title = str(title)

    if (
        "(" in title
        and ")" in title
    ):

        year = title[-5:-1]

        if year.isdigit():

            return year

    return ""


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
    ]

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

    st.markdown(
        f"""
        <div class="movie-section-title">
            {section_title}
        </div>
        """,
        unsafe_allow_html=True
    )

    columns = st.columns(
        len(movies)
    )

    for column, (_, movie) in zip(
        columns,
        movies.iterrows()
    ):

        movie_id = movie[
            "movieId"
        ]

        title = str(
            movie["title"]
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

        with column:

            # ------------------------------------------------
            # Poster
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
                    use_container_width=True
                )

            else:

                st.markdown(
                    """
                    🎬

                    **Poster unavailable**
                    """
                )

            # ------------------------------------------------
            # Title
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
            # Year
            # ------------------------------------------------

            year = get_year(
                title
            )

            if year:

                st.markdown(
                    f"""
                    <div class="movie-card-year">
                        {year}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ------------------------------------------------
            # Details Button
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

                st.rerun()


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
# HOMEPAGE MOVIE ROWS
# ============================================================

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


st.divider()


# ============================================================
# RECOMMENDATION METHOD
# ============================================================

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
    horizontal=True
)


if method == "🤝 Collaborative Filtering":

    st.markdown(
        """
        <div class="method-description">
        Recommends movies based on relationships between
        movies derived from user rating behaviour.
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="method-description">
        Recommends movies based on content similarity using
        movie genres and user-generated tags.
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# ============================================================
# SEARCH SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔎 Find a Movie'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LIVE AUTOCOMPLETE SEARCH
# ============================================================

selected_search_movie = st_searchbox(
    search_movies,
    placeholder=(
        "Search for a movie, e.g. Spider-Man, "
        "Toy Story, Titanic..."
    ),
    label=None,
    key="movie_searchbox",
    debounce=200,
    rerun_on_update=True,
    clear_on_submit=False,
    edit_after_submit="option"
)


# ============================================================
# HANDLE MOVIE SELECTION
# ============================================================

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

    except Exception:

        pass


# ============================================================
# SELECTED MOVIE
# ============================================================

selected_movie_id = (
    st.session_state.selected_movie_id
)


if selected_movie_id is not None:

    selected_matches = movie_metadata[
        movie_metadata["movieId"]
        == selected_movie_id
    ]

    if not selected_matches.empty:

        selected_movie = (
            selected_matches.iloc[0]
        )

        selected_title = html.escape(
            str(
                selected_movie["title"]
            )
        )

        selected_genres = html.escape(
            str(
                selected_movie["genres"]
            )
        )

        st.markdown(
            '<div class="selected-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="selected-title">
                🎬 {selected_title}
            </div>

            <div class="selected-genres">
                {selected_genres}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # Recommend Button
        # ----------------------------------------------------

        if st.button(
            "✨ Recommend Top 10 Movies",
            type="primary",
            use_container_width=True,
            key="recommend_button"
        ):

            st.session_state.recommend_clicked = (
                True
            )


# ============================================================
# RECOMMENDATIONS
# ============================================================

if (
    st.session_state.recommend_clicked
    and st.session_state.selected_movie_id
    is not None
):

    selected_movie_id = (
        st.session_state.selected_movie_id
    )

    with st.spinner(
        "Generating recommendations..."
    ):

        # ----------------------------------------------------
        # Collaborative Filtering
        # ----------------------------------------------------

        if method == "🤝 Collaborative Filtering":

            recommendations = recommend_cf(
                selected_movie_id,
                n=10
            )

        # ----------------------------------------------------
        # Content-Based Filtering
        # ----------------------------------------------------

        else:

            recommendations = recommend_content(
                selected_movie_id,
                n=10
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

            movie_id = movie[
                "movieId"
            ]

            movie_title = str(
                movie["title"]
            )

            genres = str(
                movie["genres"]
            )

            tmdb_id = movie.get(
                "tmdbId"
            )

            # ------------------------------------------------
            # TMDB
            # ------------------------------------------------

            tmdb_movie = get_tmdb_movie(
                tmdb_id
            )

            poster_url = None

            overview = None

            release_date = ""

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

                if (
                    tmdb_movie.get(
                        "_media_type"
                    )
                    == "tv"
                ):

                    release_date = (
                        tmdb_movie.get(
                            "first_air_date",
                            ""
                        )
                    )

                else:

                    release_date = (
                        tmdb_movie.get(
                            "release_date",
                            ""
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
            # Poster
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
            # Information
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
            # Details Button
            # ------------------------------------------------

            with col3:

                st.write("")

                details_key = (
                    f"details_"
                    f"{method}_"
                    f"{movie_id}"
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

                    st.rerun()

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

    st.session_state.details_movie_id = None


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        Movie Recommendation System
        · Collaborative Filtering & Content-Based Filtering
    </div>
    """,
    unsafe_allow_html=True
)
