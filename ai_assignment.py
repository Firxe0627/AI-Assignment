import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from streamlit_searchbox import st_searchbox


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

    /* ========================================================
       MAIN
       ======================================================== */

    .main {
        padding-top: 1rem;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        padding: 30px 0 25px 0;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 750;
        letter-spacing: -1px;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #a7adb8;
    }


    /* ========================================================
       SECTION TITLE
       ======================================================== */

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 12px;
    }


    /* ========================================================
       METHOD
       ======================================================== */

    .method-description {
        color: #9da3ae;
        font-size: 14px;
        margin-top: -5px;
        margin-bottom: 15px;
    }


    /* ========================================================
       SELECTED MOVIE
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
       RECOMMENDATION CARD
       ======================================================== */

    .recommendation-card {
        background: #151820;
        border: 1px solid #252b36;
        border-radius: 14px;

        padding: 16px;

        margin-bottom: 15px;
    }

    .rank {
        font-size: 17px;
        font-weight: 700;
        color: #ff4b4b;

        margin-bottom: 4px;
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


    /* ========================================================
       FOOTER
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
# SESSION STATE
# ============================================================

if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None

if "selected_movie_title" not in st.session_state:
    st.session_state.selected_movie_title = None

if "recommend_clicked" not in st.session_state:
    st.session_state.recommend_clicked = False

if "details_movie_id" not in st.session_state:
    st.session_state.details_movie_id = None


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

@st.cache_resource
def load_cf_model():

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
    ) = load_cf_model()

except Exception as e:

    st.error(
        f"Unable to load recommendation model: {e}"
    )

    st.stop()


# ============================================================
# LOAD LINKS
# ============================================================

@st.cache_data
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


links = load_links()


# ============================================================
# ADD TMDB ID
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
# CLEAN MOVIE METADATA
# ============================================================

movie_metadata["movieId"] = pd.to_numeric(
    movie_metadata["movieId"],
    errors="coerce"
)

movie_metadata["tmdbId"] = pd.to_numeric(
    movie_metadata["tmdbId"],
    errors="coerce"
)

movie_metadata["title"] = (
    movie_metadata["title"]
    .fillna("Unknown Movie")
    .astype(str)
)

movie_metadata["genres"] = (
    movie_metadata["genres"]
    .fillna("")
    .astype(str)
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

    if pd.isna(tmdb_id):

        return None

    if TMDB_TOKEN is None:

        return None

    try:

        url = (
            "https://api.themoviedb.org/3/movie/"
            f"{int(tmdb_id)}"
        )

        headers = {
            "Authorization":
                f"Bearer {TMDB_TOKEN}",

            "accept":
                "application/json"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:

            return None

        return response.json()

    except Exception:

        return None


# ============================================================
# TMDB POSTER
# ============================================================

def get_poster_url(
    poster_path
):

    if not poster_path:

        return None

    return (
        "https://image.tmdb.org/t/p/w500"
        + poster_path
    )


# ============================================================
# SEARCH MOVIES
# ============================================================

def search_movies(
    search_text
):

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
        .str.lower()
        .str.contains(
            search_text,
            regex=False,
            na=False
        )
    ].copy()

    # Maximum 20 suggestions
    matches = matches.head(20)

    # streamlit-searchbox supports
    # tuples:
    #
    # (display text, returned value)

    results = []

    for _, movie in matches.iterrows():

        movie_id = int(
            movie["movieId"]
        )

        movie_title = str(
            movie["title"]
        )

        results.append(
            (
                movie_title,
                movie_id
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

        # Do not recommend the selected movie
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
                "movieId":
                    similar_movie_id
            }
        )

        if len(recommendations) >= n:

            break

    recommendations_df = pd.DataFrame(
        recommendations
    )

    if recommendations_df.empty:

        return recommendations_df

    recommendations_df = (
        recommendations_df
        .merge(
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
    )

    return recommendations_df


# ============================================================
# CONTENT-BASED MODEL
# ============================================================

@st.cache_resource
def build_content_model():

    content_data = (
        movie_metadata.copy()
        .reset_index(drop=True)
    )

    content_data["genres"] = (
        content_data["genres"]
        .fillna("")
        .astype(str)
        .str.replace(
            "|",
            " ",
            regex=False
        )
    )

    vectorizer = TfidfVectorizer(
        lowercase=True
    )

    genre_matrix = (
        vectorizer.fit_transform(
            content_data["genres"]
        )
    )

    return (
        content_data,
        vectorizer,
        genre_matrix
    )


(
    content_data,
    content_vectorizer,
    content_matrix
) = build_content_model()


# ============================================================
# CONTENT-BASED RECOMMENDATION
# ============================================================

def recommend_content(
    movie_id,
    n=10
):

    matches = content_data[
        content_data["movieId"]
        == movie_id
    ]

    if matches.empty:

        return pd.DataFrame()

    movie_index = (
        matches.index[0]
    )

    similarity_scores = (
        cosine_similarity(
            content_matrix[
                movie_index
            ],
            content_matrix
        )
        .flatten()
    )

    ranked_indices = (
        similarity_scores
        .argsort()[::-1]
    )

    recommendations = []

    for index in ranked_indices:

        if index == movie_index:

            continue

        movie = content_data.iloc[
            index
        ]

        recommendations.append(
            {
                "movieId":
                    movie["movieId"],

                "title":
                    movie["title"],

                "genres":
                    movie["genres"],

                "tmdbId":
                    movie["tmdbId"]
            }
        )

        if len(recommendations) >= n:

            break

    return pd.DataFrame(
        recommendations
    )


# ============================================================
# METHOD SELECTION
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
        Recommends movies with similar content characteristics,
        using movie genres as the content representation.
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# ============================================================
# LIVE MOVIE SEARCH
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔎 Find a Movie'
    '</div>',
    unsafe_allow_html=True
)


selected_search_movie = st_searchbox(
    search_movies,
    placeholder="Start typing a movie title...",
    label=None,
    key="movie_searchbox",
    debounce=150,
    rerun_on_update=True,
    clear_on_submit=False
)


# ============================================================
# HANDLE SEARCH SELECTION
# ============================================================

if selected_search_movie is not None:

    try:

        selected_movie_id = int(
            selected_search_movie
        )

        st.session_state.selected_movie_id = (
            selected_movie_id
        )

        selected_matches = movie_metadata[
            movie_metadata["movieId"]
            == selected_movie_id
        ]

        if not selected_matches.empty:

            st.session_state.selected_movie_title = (
                selected_matches.iloc[0]["title"]
            )

        st.session_state.recommend_clicked = False

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

        st.markdown(
            '<div class="selected-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="selected-title">
                🎬 {selected_movie["title"]}
            </div>

            <div class="selected-genres">
                {selected_movie["genres"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # RECOMMEND BUTTON
        # ====================================================

        if st.button(
            "✨ Recommend Top 10 Movies",
            type="primary",
            use_container_width=True,
            key="recommend_button"
        ):

            st.session_state.recommend_clicked = True


# ============================================================
# MOVIE DETAILS DIALOG
# ============================================================

@st.dialog(
    "🎬 Movie Details"
)
def show_movie_details(
    movie_id
):

    movie_matches = movie_metadata[
        movie_metadata["movieId"]
        == movie_id
    ]

    if movie_matches.empty:

        st.error(
            "Movie information is unavailable."
        )

        return

    movie = (
        movie_matches.iloc[0]
    )

    tmdb_movie = get_tmdb_movie(
        movie["tmdbId"]
    )

    poster_url = None
    description = None
    release_date = None

    if tmdb_movie:

        poster_url = get_poster_url(
            tmdb_movie.get(
                "poster_path"
            )
        )

        description = (
            tmdb_movie.get(
                "overview"
            )
        )

        release_date = (
            tmdb_movie.get(
                "release_date"
            )
        )

    # --------------------------------------------------------
    # POSTER
    # --------------------------------------------------------

    if poster_url:

        st.image(
            poster_url,
            use_container_width=True
        )

    else:

        st.info(
            "Poster unavailable."
        )


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.subheader(
        movie["title"]
    )


    # --------------------------------------------------------
    # GENRES
    # --------------------------------------------------------

    if movie["genres"]:

        st.caption(
            movie["genres"].replace(
                "|",
                " • "
            )
        )


    # --------------------------------------------------------
    # RELEASE DATE
    # --------------------------------------------------------

    if release_date:

        st.caption(
            f"Release date: {release_date}"
        )


    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    if description:

        st.write(
            description
        )

    else:

        st.info(
            "Movie description is unavailable."
        )


    # --------------------------------------------------------
    # TMDB FALLBACK
    # --------------------------------------------------------

    if TMDB_TOKEN is None:

        st.caption(
            "TMDB information is unavailable because "
            "the TMDB API token has not been configured."
        )


# ============================================================
# RECOMMENDATIONS
# ============================================================

if (
    st.session_state.recommend_clicked
    and selected_movie_id is not None
):

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '✨ Recommended Movies'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # GENERATE RECOMMENDATIONS
    # --------------------------------------------------------

    with st.spinner(
        "Generating recommendations..."
    ):

        if method == "🤝 Collaborative Filtering":

            recommendations = recommend_cf(
                selected_movie_id,
                n=10
            )

        else:

            recommendations = recommend_content(
                selected_movie_id,
                n=10
            )


    # --------------------------------------------------------
    # EMPTY RESULT
    # --------------------------------------------------------

    if recommendations.empty:

        st.error(
            "We could not generate recommendations "
            "for this movie."
        )

    else:

        st.success(
            f"{len(recommendations)} recommendations generated."
        )


        # ====================================================
        # RECOMMENDATION CARDS
        # ====================================================

        for rank, (_, movie) in enumerate(
            recommendations.head(10).iterrows(),
            start=1
        ):

            movie_id = int(
                movie["movieId"]
            )

            movie_title = str(
                movie["title"]
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
            # TMDB INFORMATION
            # ------------------------------------------------

            tmdb_movie = get_tmdb_movie(
                tmdb_id
            )

            poster_url = None
            overview = None

            if tmdb_movie:

                poster_url = (
                    get_poster_url(
                        tmdb_movie.get(
                            "poster_path"
                        )
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

            with st.container(
                border=True
            ):

                col1, col2 = st.columns(
                    [1.2, 5]
                )


                # ============================================
                # POSTER
                # ============================================

                with col1:

                    if poster_url:

                        st.image(
                            poster_url,
                            use_container_width=True
                        )

                    else:

                        st.markdown(
                            """
                            <div style="
                                height:180px;
                                display:flex;
                                align-items:center;
                                justify-content:center;
                                background:#1c212b;
                                border-radius:10px;
                                color:#8f96a3;
                                text-align:center;
                            ">
                                🎬<br>
                                Poster unavailable
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                # ============================================
                # INFORMATION
                # ============================================

                with col2:

                    st.markdown(
                        f"""
                        <div class="rank">
                            #{rank}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    # ----------------------------------------
                    # CLICKABLE MOVIE TITLE
                    # ----------------------------------------

                    if st.button(
                        movie_title,
                        key=f"movie_details_{method}_{movie_id}",
                        type="tertiary"
                    ):

                        st.session_state.details_movie_id = (
                            movie_id
                        )

                        st.rerun()


                    # ----------------------------------------
                    # GENRES
                    # ----------------------------------------

                    if genres and genres != "nan":

                        st.markdown(
                            f"""
                            <div class="recommendation-genres">
                                {genres.replace("|", " • ")}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                    # ----------------------------------------
                    # DESCRIPTION
                    # ----------------------------------------

                    if overview:

                        short_description = (
                            overview[:280]
                            + "..."
                            if len(overview) > 280
                            else overview
                        )

                        st.markdown(
                            f"""
                            <div class="recommendation-description">
                                {short_description}
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


# ============================================================
# OPEN DETAILS DIALOG
# ============================================================

if (
    st.session_state.details_movie_id
    is not None
):

    details_movie_id = (
        st.session_state.details_movie_id
    )

    # Clear state before opening dialog
    st.session_state.details_movie_id = None

    show_movie_details(
        details_movie_id
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">

        Movie Recommendation System
        · Collaborative Filtering
        · Content-Based Filtering

    </div>
    """,
    unsafe_allow_html=True
)
