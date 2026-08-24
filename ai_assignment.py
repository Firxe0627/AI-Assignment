import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

st.markdown("""
<style>

    /* ---------- Main page ---------- */

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

    /* ---------- Method cards ---------- */

    .method-description {
        color: #9da3ae;
        font-size: 14px;
        margin-top: -5px;
        margin-bottom: 15px;
    }

    /* ---------- Selected movie ---------- */

    .selected-card {
        background: linear-gradient(
            135deg,
            #151922,
            #1c2330
        );
        border: 1px solid #2c3442;
        border-radius: 14px;
        padding: 20px;
        margin-top: 10px;
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

    /* ---------- Recommendation cards ---------- */

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

    /* ---------- Info ---------- */

    .info-box {
        background: #131720;
        border: 1px solid #282f3a;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 15px;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #747b87;
        font-size: 13px;
        padding: 30px 0 10px 0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="hero">
<div class="hero-title">🎬 Movie Recommendation System</div>
<div class="hero-subtitle">Discover movies using intelligent recommendation techniques</div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
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
            "accept": "application/json"
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
# POSTER
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
        return pd.DataFrame()

    search_text = (
        search_text
        .strip()
        .lower()
    )

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

        recommendations.append({
            "movieId":
                similar_movie_id
        })

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

    return recommendations_df


# ============================================================
# CONTENT-BASED FILTERING
# ============================================================

@st.cache_resource
def build_content_model():

    content_data = movie_metadata.copy()

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

    genre_matrix = vectorizer.fit_transform(
        content_data["genres"]
    )

    return (
        content_data,
        vectorizer,
        genre_matrix
    )


content_data, content_vectorizer, content_matrix = (
    build_content_model()
)


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

    movie_index = matches.index[0]

    similarity_scores = cosine_similarity(
        content_matrix[movie_index],
        content_matrix
    ).flatten()

    ranked_indices = (
        similarity_scores
        .argsort()[::-1]
    )

    recommendations = []

    for index in ranked_indices:

        if index == movie_index:
            continue

        recommendations.append(
            content_data.iloc[index][
                [
                    "movieId",
                    "title",
                    "genres",
                    "tmdbId"
                ]
            ]
        )

        if len(recommendations) >= n:
            break

    if not recommendations:
        return pd.DataFrame()

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
# SEARCH SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔎 Find a Movie'
    '</div>',
    unsafe_allow_html=True
)

search_text = st.text_input(
    "Movie title",
    placeholder=(
        "Search for a movie, e.g. Spider-Man, "
        "Toy Story, Titanic..."
    ),
    label_visibility="collapsed"
)


# ============================================================
# SEARCH RESULTS
# ============================================================

matching_movies = search_movies(
    search_text
)


if search_text:

    if matching_movies.empty:

        st.warning(
            "No movies found. "
            "Try a different title or spelling."
        )

    else:

        st.caption(
            f"{len(matching_movies)} movie(s) found"
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


    # ========================================================
    # SELECTED MOVIE CARD
    # ========================================================

    st.markdown(
        '<div class="selected-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="selected-title">
            🎬 {selected_movie['title']}
        </div>

        <div class="selected-genres">
            {selected_movie['genres']}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # RECOMMEND BUTTON
    # ========================================================

    recommend_clicked = st.button(
        "✨ Recommend Top 10 Movies",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    if recommend_clicked:

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


        if recommendations.empty:

            st.error(
                "We could not generate recommendations "
                "for this movie."
            )

        else:

            st.success(
                "10 recommendations generated."
            )

            st.markdown(
                '<div class="section-title">'
                '✨ Recommended Movies'
                '</div>',
                unsafe_allow_html=True
            )


            # ==================================================
            # RECOMMENDATION CARDS
            # ==================================================

            for rank, (_, movie) in enumerate(
                recommendations.iterrows(),
                start=1
            ):

                movie_title = movie["title"]

                genres = movie["genres"]

                tmdb_id = movie["tmdbId"]


                # ----------------------------------------------
                # TMDB INFORMATION
                # ----------------------------------------------

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


                # ----------------------------------------------
                # CARD
                # ----------------------------------------------

                st.markdown(
                    '<div class="recommendation-card">',
                    unsafe_allow_html=True
                )


                col1, col2, col3 = st.columns(
                    [1.2, 5, 1.5]
                )


                # ----------------------------------------------
                # POSTER
                # ----------------------------------------------

                with col1:

                    if poster_url:

                        st.image(
                            poster_url,
                            width=120
                        )

                    else:

                        st.markdown(
                            "🎬\n\n"
                            "**Poster unavailable**"
                        )


                # ----------------------------------------------
                # MOVIE INFORMATION
                # ----------------------------------------------

                with col2:

                    st.markdown(
                        f"""
                        <div class="rank">
                            #{rank}
                        </div>

                        <div class="recommendation-title">
                            {movie_title}
                        </div>

                        <div class="recommendation-genres">
                            {genres}
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


                # ----------------------------------------------
                # DETAILS BUTTON
                # ----------------------------------------------

                with col3:

                    st.write("")

                    details_key = (
                        f"details_"
                        f"{method}_"
                        f"{movie['movieId']}"
                    )


                    if st.button(
                        "View Details",
                        key=details_key,
                        use_container_width=True
                    ):

                        @st.dialog(
                            movie_title
                        )
                        def show_movie_details():

                            tmdb_details = (
                                get_tmdb_movie(
                                    tmdb_id
                                )
                            )

                            if tmdb_details:

                                poster = (
                                    get_poster_url(
                                        tmdb_details.get(
                                            "poster_path"
                                        )
                                    )
                                )

                                if poster:

                                    st.image(
                                        poster,
                                        use_container_width=True
                                    )

                                st.subheader(
                                    tmdb_details.get(
                                        "title",
                                        movie_title
                                    )
                                )

                                st.caption(
                                    tmdb_details.get(
                                        "release_date",
                                        ""
                                    )
                                )

                                st.write(
                                    tmdb_details.get(
                                        "overview",
                                        "Description unavailable."
                                    )
                                )

                            else:

                                st.subheader(
                                    movie_title
                                )

                                st.write(
                                    f"Genres: {genres}"
                                )

                                st.info(
                                    "Additional movie details "
                                    "are currently unavailable."
                                )


                        show_movie_details()


                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )


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
