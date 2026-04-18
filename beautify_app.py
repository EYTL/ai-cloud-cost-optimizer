import os

file_path = 'app.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the old CSS config block
new_css = '''st.set_page_config(page_title="AI Cloud Cost Optimizer", layout="wide")

st.markdown("""
<style>
    /* Gradient Title */
    h1 {
        padding-bottom: 2rem !important;
        background: -webkit-linear-gradient(45deg, #0ea5e9, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    /* Section Header Partitions */
    h3 {
        margin-top: 3.5rem !important;
        margin-bottom: 2rem !important;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.3);
        color: #0ea5e9;
        font-weight: 600 !important;
    }
    
    /* Subtle Dividers */
    hr {
        margin-top: 3rem !important;
        margin-bottom: 3rem !important;
        border-color: rgba(128, 128, 128, 0.2) !important;
    }
    
    /* Floating Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.1);
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Make metric values pop */
    div[data-testid="stMetricValue"] {
        font-weight: 700 !important;
        color: #0ea5e9 !important;
    }
</style>
""", unsafe_allow_html=True)'''

# Using string logic to extract the old CSS and replace it
import re

# Find the block between st.set_page_config and the next `# ───` comment
pattern = r'st\.set_page_config\(.*?\)(.*?)(?=# ───)'
content = re.sub(pattern, lambda m: new_css + "\n\n", content, flags=re.DOTALL, count=1)

# Convert all st.subheader("...") to st.markdown("### ...")
content = re.sub(r'st\.subheader\("(.*?)"\)', r'st.markdown("### \1")', content)

# Remove all st.divider() because our new underlined h3 headers act as better partitions
content = content.replace("st.divider()", 'st.write("")')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("UI Beautified!")
