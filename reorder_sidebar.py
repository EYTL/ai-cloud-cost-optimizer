import re

file_path = 'app.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add containers exactly after st.sidebar.title
container_setup = '''st.sidebar.title("Dashboard Controls")

source_container = st.sidebar.container()
filter_container = st.sidebar.container()
creds_container = st.sidebar.container()
'''
content = content.replace('st.sidebar.title("Dashboard Controls")', container_setup)

# 2. Replace st.sidebar calls with the specific containers based on section
# Data Source Selection goes to source_container
content = content.replace('source = st.sidebar.radio(', 'source = source_container.radio(')

# Credentials go to creds_container
content = content.replace('st.sidebar.markdown("### Upload Your Data")', 'creds_container.markdown("### Upload Your Data")')
content = content.replace('st.sidebar.file_uploader(', 'creds_container.file_uploader(')
content = content.replace('st.sidebar.caption(', 'creds_container.caption(')

content = content.replace('st.sidebar.markdown("### AWS Credentials")', 'creds_container.markdown("### AWS Credentials")')
content = content.replace('st.sidebar.expander("Enter AWS Credentials"', 'creds_container.expander("Enter AWS Credentials"')

# For GCP and Azure (they might have been removed, but just in case of residue)
content = content.replace('st.sidebar.markdown("### GCP Credentials")', 'creds_container.markdown("### GCP Credentials")')
content = content.replace('st.sidebar.expander("Enter GCP Details"', 'creds_container.expander("Enter GCP Details"')

content = content.replace('st.sidebar.markdown("### Azure Credentials")', 'creds_container.markdown("### Azure Credentials")')
content = content.replace('st.sidebar.expander("Enter Azure Details"', 'creds_container.expander("Enter Azure Details"')

# Filters go to filter_container
content = content.replace('st.sidebar.slider(', 'filter_container.slider(')
content = content.replace('st.sidebar.selectbox(', 'filter_container.selectbox(')

# Remove the hacky 300px div we put earlier
content = content.replace('''# Force the selectbox dropdown to open downwards by ensuring there is empty space below it in the sidebar\nst.sidebar.markdown("<div style='height: 300px;'></div>", unsafe_allow_html=True)''', '')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Sidebar layout restructured to force downward opening dropdown!")
