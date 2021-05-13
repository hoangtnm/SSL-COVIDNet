import streamlit as st

from pages import home, dashboard, inference

PAGES = {
    "Home": home,
    "Dashboard": dashboard,
    "COVID-19 Diagnosis": inference,
}

sidebar = st.sidebar
selection = sidebar.selectbox("Navigate",
                              ["Home", "Dashboard", "COVID-19 Diagnosis"])
# sidebar.header("Keywords")
# sidebar.info("""COVID-19, Deep Learning, Self-Supervised Learning,
# Medical Imaging, Computer Tomography.""")
sidebar.header("About")
sidebar.info("""
    The application is maintained by Tran N.M. Hoang.
    Contact: [hoangtnm](https://github.com/hoangtnm)
""")
sidebar.header("Contributing")
sidebar.info("""
    Contribution is appreciated. To contribute to this project, please
    make pull requests at the [SSL-COVIDNet](https://github.com/hoangtnm/SSL-COVIDNet) repository.
""")
sidebar.header("License")
sidebar.info("""
    This application has a [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.
""")
page = PAGES[selection]
page.app()
