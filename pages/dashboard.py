import streamlit as st
from datetime import datetime
import pandas as pd
from typing import Optional
# import plotly.express as px
import pydeck as pdk
# import altair as alt
from urllib.error import HTTPError


@st.cache
def fetch_data(date: datetime,
               country: Optional[str] = None,
               nrows: Optional[int] = None) -> pd.DataFrame:
    DATA_URL = (
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master"
        "/csse_covid_19_data/"
        f"csse_covid_19_daily_reports/{date.strftime('%m-%d-%Y')}.csv")
    if country == "US":
        DATA_URL = (
            "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master"
            "/csse_covid_19_data/"
            f"csse_covid_19_daily_reports_us/{date.strftime('%m-%d-%Y')}.csv")

    df = pd.read_csv(DATA_URL, nrows=nrows)
    df = df.rename(columns=str.lower)
    df = df.rename(columns={"long_": "lon"})
    # df = df.drop(columns=["province_state", "combined_key", "incident_rate"])
    df["last_update"] = pd.to_datetime(df["last_update"]).dt.strftime(
        '%m-%d-%Y')
    df = df.dropna(subset=["lat", "lon"])
    df = df.fillna(0)
    return df


def app():
    st.title("COVID-19 Dashboard")
    st.write("""
        The raw data is updated and maintained by Johns Hopkins University [1].
    """)

    try:
        st.header("Raw Data")
        date = st.date_input("Filter by Date:")
        df = fetch_data(date)
        selected_countries = st.multiselect("Filter by Country or Region:",
                                            df["country_region"].unique())
        st.dataframe(df.loc[df["country_region"].isin(
            selected_countries)] if selected_countries else df)

        st.subheader("Summary")
        n_confirmed = df["confirmed"].sum() / (10 ** 6)
        n_confirmed_biggest = df.groupby("country_region").agg(
            {"confirmed": "sum"}).nlargest(1, "confirmed")
        n_deaths = df["deaths"].sum() / (10 ** 6)
        n_recovered = df["recovered"].sum() / (10 ** 6)
        st.markdown(f"""
            - The COVID-19 pandemic has spreded to **{len(df["country_region"].unique())}** countries and regions.
            - About **{n_confirmed:.2f}M** people have been infected.
            - About **{n_deaths:.2f}M** people have died due to the pandemic.
            - About **{n_recovered:.2f}M** people have recovered.
            - {n_confirmed_biggest.index.values[0]} is the most-affected country
            with **{n_confirmed_biggest["confirmed"].values[0]}** COVID-19 cases.
        """)

        st.header("Daily Statistics")
        metric = st.selectbox("Metric:",
                              ["confirmed", "deaths", "recovered", "active"])

        # Set viewport for the deckgl map
        view = pdk.ViewState(latitude=0, longitude=0, zoom=0.2, )

        # Create the scatter plot layer
        covidLayer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            pickable=False,
            opacity=0.3,
            stroked=True,
            filled=True,
            radius_scale=1,
            radius_min_pixels=5,
            radius_max_pixels=60,
            line_width_min_pixels=1,
            get_position=["lon", "lat"],
            get_radius=metric,
            get_fill_color=[252, 136, 3],
            get_line_color=[255, 0, 0],
            tooltip="test test",
        )
        # covidLayer.data = df[df["last_update"] == date.isoformat()]

        # Create the deck.gl map
        r = pdk.Deck(
            layers=[covidLayer],
            initial_view_state=view,
            map_style="mapbox://styles/mapbox/light-v10",
        )
        map = st.pydeck_chart(r)

        st.header("References")
        st.write("""
            [1] Dong E, Du H, Gardner L. An interactive web-based dashboard to track
            COVID-19 in real time. Lancet Inf Dis. 20(5):533-534.
            doi: 10.1016/S1473-3099(20)30120-1
        """)
    except HTTPError:
        st.error("Data is not available.")
    except Exception as e:
        st.error(f"Error: {e}")
