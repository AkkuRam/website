import streamlit as st

# Set up the page title and icon
# st.set_page_config(page_title="My Portfolio", page_icon=":wave:", layout="wide")

# Main section
st.title("Akhilesh Ramesh")

st.write("""
I'm Akhilesh Ramesh, a student with passion in coding in Rust and Python
focusing on projects related to system utils and data science. 
This portfolio showcases my skills, projects, and contact information.
""")

# About Section
st.header("About Me")

st.write("""
I come from Germany, where I speak the languages: English, German and Kannada. 
I have completed my bachelor's degree in Data Science and Artificial Intelligence at Maastricht University. 
Currently, I am studying the Data Science for Decision Making masters at Maastricht University.
In my free time, I enjoy programming and playing chess. 
The main languages I code in are Python and Rust, however I have experience coding in other languages as well (i.e. Java, MATLAB, R, Golang).
""")

# Skills Section
st.header("Skills")
st.write("""
- **Programming Languages**: Python, Rust, Java, MATLAB, R, Golang
- **Courses**: Streamlit, Flask, React
- **Tools**: Git, Docker, VS Code
- **Other**: Problem-solving, Teamwork, Communication
""")

# Projects Section
st.header("Projects")
st.subheader("Disk Analyzer")

st.image("disk_analyzer.png", use_container_width=True)

st.write("""
This project was done in python, where the main objective was to display system utils. Essentially a system tool
similar to a task manager, that depicts different statistics (i.e. CPU usage, network speed, device storage, etc). Here the python 
library used for this was "Rich". 
""")
st.markdown("[Github: Disk Analyzer](https://github.com/AkkuRam/disk-analyzer)")

st.subheader("2. Project Two")
st.write("""
Description of your second project. You can add more projects in a similar format.
""")
st.markdown("[View Project](https://github.com/yourusername/project-two)")

# Contact Section
st.header("Contact Me")
st.write("""
Feel free to reach out via email or connect with me on LinkedIn.
""")
st.write("📧 Email: [reachakhil10@gmail.com](mailto:your.email@example.com)")
st.write("🔗 LinkedIn: [www.linkedin.com/in/akhilesh-ramesh](https://linkedin.com/in/yourusername)")
st.write("🔗 Github: [https://github.com/Akkuram](https://github.com/Akkuram)")

# Footer
st.write("---")
st.write("© 2025 Akhilesh Ramesh | Built with Streamlit")
