import streamlit as st

class MultiPage:
    """
    Class for generating and configuring additional streamlit app pages using object orientation
    """
    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name
        st.set_page_config(
            page_title = self.app_name,
            page_icon = '📄' # Sourced from the following website: https://twemoji.maxcdn.com/v/latest/svg/1f4c4.svg
        )
    
    # Function for adding pages
    def add_page(self, title, func) -> None:
        self.pages.append({"title": title, "function": func})
    
    # Creates a new instance in the sidebar with the page title
    def run(self):
        st.title(self.app_name)
        page = st.sidebar.radio('Contents', self.pages, format_func = lambda page: page['title'])
        page['function']()
