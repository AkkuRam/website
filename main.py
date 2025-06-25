import streamlit as st

def main():
    st.title("Welcome to My Streamlit App")

    st.header("About")
    st.write("This is a basic Streamlit application. You can use it to create interactive web apps with Python.")

    st.sidebar.title("Navigation")
    st.sidebar.write("Use the sidebar to explore different sections of this app.")

    if st.button("Click Me"):
        st.success("Button clicked! Welcome to the app.")

    name = st.text_input("What's your name?")
    if name:
        st.write(f"Hello, {name}! Thanks for visiting.")

if __name__ == "__main__":
    main()