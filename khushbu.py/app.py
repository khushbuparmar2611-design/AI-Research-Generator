import streamlit as st
import json
import os
import io
import zipfile
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

try:
    import fitz
except ImportError:
    fitz = None

try:
    import bibtexparser
except ImportError:
    bibtexparser = None

import requests

st.set_page_config(
    page_title="AI Research Generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding: 2rem 2.5rem;
        max-width: 1200px;
    }

    .stSidebar {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    .stSidebar [data-testid="stSidebarNav"] {
        padding-top: 1rem;
    }

    .sidebar-title {
        color: #f8fafc;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 1rem 1rem 0.5rem;
        border-bottom: 1px solid #334155;
        margin-bottom: 0.5rem;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }

    .section-sub {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    .badge-complete {
        background: #dcfce7;
        color: #15803d;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .badge-missing {
        background: #fef9c3;
        color: #854d0e;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .stButton > button {
        background: #1e40af;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: background 0.2s;
    }

    .stButton > button:hover {
        background: #1d4ed8;
    }

    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        font-size: 0.875rem;
    }

    .generated-content {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #1e40af;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.875rem;
        line-height: 1.6;
    }

    h1 { color: #0f172a; }
    h2 { color: #1e293b; font-size: 1.2rem; font-weight: 600; }
    h3 { color: #334155; font-size: 1rem; font-weight: 600; }

    .step-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session():
    defaults = {
        "project": {
            "title": "", "domain": "", "research_type": "Review Paper",
            "citation_style": "IEEE", "journal_template": "Generic Academic",
            "num_pages": 8, "keywords": ""
        },
        "authors": [],
        "references": [],
        "bibtex": "",
        "gemini_key": "",
        "planner": {},
        "sections": {
            "abstract": "", "introduction": "", "literature_review": "",
            "methodology": "", "results": "", "discussion": "", "conclusion": ""
        },
        "figures": [],
        "charts": [],
        "active_page": "Project Setup"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def call_gemini(prompt: str, api_key: str) -> str:
    if not api_key:
        return "ERROR: No Gemini API key provided."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite-preview-06-17:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048}
    }
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.HTTPError as e:
        return f"ERROR: API request failed ({e.response.status_code}). Check your API key."
    except Exception as e:
        return f"ERROR: {str(e)}"


def extract_pdf_metadata(uploaded_file) -> dict:
    if fitz is None:
        return {"title": uploaded_file.name, "authors": "Unknown", "abstract": "", "raw_text": "PyMuPDF not installed"}
    try:
        data = uploaded_file.read()
        doc = fitz.open(stream=data, filetype="pdf")
        meta = doc.metadata or {}
        title = meta.get("title", "") or uploaded_file.name.replace(".pdf", "")
        authors = meta.get("author", "") or "Unknown"
        first_page_text = ""
        if len(doc) > 0:
            first_page_text = doc[0].get_text()[:1500]
        abstract = ""
        lower = first_page_text.lower()
        if "abstract" in lower:
            idx = lower.index("abstract")
            abstract = first_page_text[idx:idx+800].strip()
        doc.close()
        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "raw_text": first_page_text,
            "filename": uploaded_file.name
        }
    except Exception as e:
        return {"title": uploaded_file.name, "authors": "Unknown", "abstract": "", "raw_text": str(e), "filename": uploaded_file.name}


def generate_bibtex(references: list) -> str:
    entries = []
    for i, ref in enumerate(references):
        key = f"ref{i+1}"
        title = ref.get("title", f"Reference {i+1}").replace("{", "").replace("}", "")
        authors = ref.get("authors", "Unknown Author")
        entry = f"""@article{{{key},
  title = {{{title}}},
  author = {{{authors}}},
  year = {{2024}},
  journal = {{Unknown Journal}}
}}"""
        entries.append(entry)
    return "\n\n".join(entries)


def build_latex(project, authors, sections, bibtex_content) -> str:
    author_str = " \\and ".join([a.get("name", "Author") for a in authors]) if authors else "Author Name"
    affil = authors[0].get("affiliation", "University") if authors else "University"

    def safe(text):
        if not text:
            return "This section has not been generated yet."
        return text.replace("&", "\\&").replace("%", "\\%").replace("$", "\\$").replace("#", "\\#").replace("_", "\\_").replace("^", "\\^{}")

    latex = f"""\\documentclass[12pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\usepackage{{natbib}}
\\usepackage{{geometry}}
\\usepackage{{hyperref}}
\\usepackage{{amsmath}}
\\usepackage{{booktabs}}
\\usepackage{{setspace}}
\\geometry{{margin=1in}}
\\onehalfspacing

\\title{{{safe(project.get('title', 'Research Paper'))}}}
\\author{{{safe(author_str)} \\\\ \\small {safe(affil)}}}
\\date{{\\today}}

\\bibliographystyle{{plain}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{safe(sections.get('abstract', ''))}
\\end{{abstract}}

\\textbf{{Keywords:}} {safe(project.get('keywords', ''))}

\\newpage
\\tableofcontents
\\newpage

\\section{{Introduction}}
{safe(sections.get('introduction', ''))}

\\section{{Literature Review}}
{safe(sections.get('literature_review', ''))}

\\section{{Methodology}}
{safe(sections.get('methodology', ''))}

\\section{{Results}}
{safe(sections.get('results', ''))}

\\section{{Discussion}}
{safe(sections.get('discussion', ''))}

\\section{{Conclusion}}
{safe(sections.get('conclusion', ''))}

\\bibliography{{references}}

\\end{{document}}
"""
    return latex


def create_project_zip(latex_content, bibtex_content, figures, charts) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project/main.tex", latex_content)
        zf.writestr("project/references.bib", bibtex_content or "% No references added")
        zf.writestr("project/images/.gitkeep", "")
        zf.writestr("project/charts/.gitkeep", "")
        for i, fig in enumerate(figures):
            try:
                zf.writestr(f"project/images/figure_{i+1}.png", fig["data"])
            except Exception:
                pass
        for i, chart in enumerate(charts):
            try:
                zf.writestr(f"project/charts/chart_{i+1}.png", chart["data"])
            except Exception:
                pass
    return buf.getvalue()


def integrity_check(project, authors, references, planner, sections) -> dict:
    checks = {
        "Research Title": bool(project.get("title", "").strip()),
        "Authors Added": len(authors) > 0,
        "References Added": len(references) > 0,
        "Research Objective": bool(planner.get("objective", "")),
        "Abstract Generated": bool(sections.get("abstract", "").strip()),
        "Introduction Generated": bool(sections.get("introduction", "").strip()),
        "Literature Review Generated": bool(sections.get("literature_review", "").strip()),
        "Methodology Generated": bool(sections.get("methodology", "").strip()),
        "Conclusion Generated": bool(sections.get("conclusion", "").strip()),
    }
    return checks


def page_project_setup():
    st.markdown('<div class="section-header">📋 Project Setup</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Define your research paper\'s core details.</div>', unsafe_allow_html=True)

    p = st.session_state.project
    col1, col2 = st.columns(2)
    with col1:
        p["title"] = st.text_input("Research Title", value=p["title"], placeholder="e.g., Machine Learning in Healthcare")
        p["domain"] = st.text_input("Research Domain", value=p["domain"], placeholder="e.g., Computer Science, Biology")
        p["research_type"] = st.selectbox("Research Type", ["Review Paper", "Survey Paper", "Case Study", "Experimental Research"], index=["Review Paper", "Survey Paper", "Case Study", "Experimental Research"].index(p["research_type"]))
    with col2:
        p["citation_style"] = st.selectbox("Citation Style", ["APA", "MLA", "IEEE"], index=["APA", "MLA", "IEEE"].index(p["citation_style"]))
        p["journal_template"] = st.selectbox("Journal Template", ["Generic Academic", "IEEE", "ACM"], index=["Generic Academic", "IEEE", "ACM"].index(p["journal_template"]))
        p["num_pages"] = st.number_input("Target Number of Pages", min_value=4, max_value=50, value=p["num_pages"])

    p["keywords"] = st.text_input("Keywords (comma-separated)", value=p["keywords"], placeholder="machine learning, neural networks, healthcare")

    st.session_state.project = p

    if p["title"]:
        st.success(f"✅ Project '{p['title']}' configured successfully.")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Type", p["research_type"])
    with col2:
        st.metric("Citation Style", p["citation_style"])
    with col3:
        st.metric("Target Pages", p["num_pages"])


def page_authors():
    st.markdown('<div class="section-header">👥 Authors</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Add all authors contributing to this paper.</div>', unsafe_allow_html=True)

    with st.expander("➕ Add New Author", expanded=len(st.session_state.authors) == 0):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", key="new_author_name")
            affiliation = st.text_input("Institution / University", key="new_author_affil")
        with col2:
            department = st.text_input("Department", key="new_author_dept")
            email = st.text_input("Email Address", key="new_author_email")

        if st.button("Add Author"):
            if name.strip():
                st.session_state.authors.append({
                    "name": name, "affiliation": affiliation,
                    "department": department, "email": email
                })
                st.success(f"Author '{name}' added.")
                st.rerun()
            else:
                st.warning("Please enter the author's name.")

    if st.session_state.authors:
        st.markdown(f"**{len(st.session_state.authors)} author(s) added:**")
        df = pd.DataFrame(st.session_state.authors)
        st.dataframe(df, use_container_width=True)

        remove_idx = st.number_input("Remove author by row number (0-indexed)", min_value=0, max_value=max(0, len(st.session_state.authors)-1), step=1, key="remove_author_idx")
        if st.button("Remove Selected Author"):
            if 0 <= remove_idx < len(st.session_state.authors):
                removed = st.session_state.authors.pop(remove_idx)
                st.success(f"Removed '{removed['name']}'.")
                st.rerun()
    else:
        st.info("No authors added yet. Use the form above to add authors.")


def page_references():
    st.markdown('<div class="section-header">📚 References</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload PDF papers to build your reference library.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload PDF references", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")

    if uploaded:
        new_names = {f.name for f in uploaded}
        existing_names = {r.get("filename", "") for r in st.session_state.references}
        for f in uploaded:
            if f.name not in existing_names:
                with st.spinner(f"Extracting metadata from {f.name}..."):
                    meta = extract_pdf_metadata(f)
                    st.session_state.references.append(meta)
        st.session_state.bibtex = generate_bibtex(st.session_state.references)

    if st.session_state.references:
        st.markdown(f"**Reference Library ({len(st.session_state.references)} papers)**")
        for i, ref in enumerate(st.session_state.references):
            with st.expander(f"📄 [{i+1}] {ref.get('title', 'Unknown Title')[:80]}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**Authors:** {ref.get('authors', 'Unknown')}")
                    if ref.get("abstract"):
                        st.write(f"**Abstract preview:** {ref['abstract'][:300]}...")
                with col2:
                    st.write(f"**File:** {ref.get('filename', 'N/A')}")

        with st.expander("📋 Generated BibTeX"):
            st.code(st.session_state.bibtex, language="bibtex")
            st.download_button("Download references.bib", st.session_state.bibtex, "references.bib", "text/plain")
    else:
        st.info("Upload PDF files above to extract references automatically.")


def page_research_planner():
    st.markdown('<div class="section-header">🗺️ Research Planner</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Use AI to generate your research structure and objectives.</div>', unsafe_allow_html=True)

    api_key = st.text_input("Gemini API Key", value=st.session_state.gemini_key, type="password", placeholder="AIza...")
    if api_key:
        st.session_state.gemini_key = api_key

    if not st.session_state.project.get("title"):
        st.warning("Please complete Project Setup first.")
        return

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**What the AI will generate:**")
        st.write("- Clear research objective\n- 3–5 focused research questions\n- Refined keyword list\n- Suggested paper structure")

    if st.button("🚀 Generate Research Plan", use_container_width=False):
        if not st.session_state.gemini_key:
            st.error("Please enter your Gemini API key.")
        else:
            p = st.session_state.project
            prompt = f"""You are an academic research assistant helping a first-year college student.

Research title: {p['title']}
Domain: {p['domain']}
Type: {p['research_type']}
Keywords: {p['keywords']}

Generate a research plan as valid JSON with these exact keys:
- objective: one clear sentence
- research_questions: list of 3-5 strings
- keywords: list of 6-10 strings
- structure: list of section names with one-sentence descriptions

Respond ONLY with valid JSON. No markdown, no explanation."""

            with st.spinner("Generating research plan..."):
                result = call_gemini(prompt, st.session_state.gemini_key)

            if result.startswith("ERROR:"):
                st.error(result)
            else:
                try:
                    clean = result.strip().replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(clean)
                    st.session_state.planner = parsed
                    st.success("Research plan generated!")
                except Exception:
                    st.session_state.planner = {"raw": result}
                    st.warning("Could not parse JSON. Showing raw output.")

    if st.session_state.planner:
        planner = st.session_state.planner
        if "raw" in planner:
            st.text_area("Raw AI Output", planner["raw"], height=200)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Research Objective**")
                st.info(planner.get("objective", ""))
                st.markdown("**Research Questions**")
                for q in planner.get("research_questions", []):
                    st.write(f"• {q}")
            with col2:
                st.markdown("**Keywords**")
                keywords = planner.get("keywords", [])
                st.write(" · ".join([f"`{k}`" for k in keywords]))
                st.markdown("**Suggested Structure**")
                for sec in planner.get("structure", []):
                    if isinstance(sec, dict):
                        for k, v in sec.items():
                            st.write(f"**{k}:** {v}")
                    else:
                        st.write(f"• {sec}")


def page_section_generator():
    st.markdown('<div class="section-header">✍️ Section Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Generate each section individually with focused AI prompts.</div>', unsafe_allow_html=True)

    if not st.session_state.gemini_key:
        st.warning("Enter your Gemini API key in the Research Planner section first.")
        return

    if not st.session_state.project.get("title"):
        st.warning("Complete Project Setup first.")
        return

    p = st.session_state.project
    planner = st.session_state.planner
    sections = st.session_state.sections
    objective = planner.get("objective", p.get("title", ""))
    refs_context = "; ".join([r.get("title", "") for r in st.session_state.references[:5]])

    section_configs = [
        ("abstract", "Abstract", f"Write a 200-word academic abstract for a {p['research_type']} titled '{p['title']}' in {p['domain']}. Research objective: {objective}. Keywords: {p.get('keywords','')}. Use formal academic English. No fake citations."),
        ("introduction", "Introduction", f"Write an introduction section (300-400 words) for a {p['research_type']} titled '{p['title']}' in {p['domain']}. Objective: {objective}. Include: background, problem statement, paper organization. Academic tone. No fabricated statistics."),
        ("literature_review", "Literature Review", f"Write a literature review section (400-500 words) for '{p['title']}'. Domain: {p['domain']}. Referenced papers: {refs_context if refs_context else 'none provided'}. Discuss research themes without fabricating specific paper details or fake citations. Academic tone."),
        ("methodology", "Methodology", f"Write a methodology section (300-400 words) for a {p['research_type']} titled '{p['title']}'. Research type: {p['research_type']}. Objective: {objective}. Describe research approach, data collection concept, and analysis method. No fabricated experiments."),
        ("results", "Results Template", f"Create a results section template (250-300 words) for '{p['title']}'. Include placeholder structure, table headings, and figure references that the student can fill in. Label placeholders clearly with [INSERT ...]."),
        ("discussion", "Discussion", f"Write a discussion section (300-400 words) for '{p['title']}'. Objective: {objective}. Discuss implications, limitations, and future work conceptually. Academic tone. No fabricated data."),
        ("conclusion", "Conclusion", f"Write a conclusion section (200-250 words) for '{p['title']}'. Objective: {objective}. Summarize key findings conceptually and state research contributions. Academic tone."),
    ]

    tabs = st.tabs([cfg[1] for cfg in section_configs])

    for i, (key, label, prompt) in enumerate(section_configs):
        with tabs[i]:
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button(f"Generate", key=f"gen_{key}"):
                    with st.spinner(f"Generating {label}..."):
                        result = call_gemini(prompt, st.session_state.gemini_key)
                    if result.startswith("ERROR:"):
                        st.error(result)
                    else:
                        sections[key] = result
                        st.session_state.sections = sections
                        st.success(f"{label} generated!")
                        st.rerun()
            with col2:
                status = "✅ Generated" if sections.get(key, "").strip() else "⏳ Not yet generated"
                st.caption(status)

            sections[key] = st.text_area(
                f"Edit {label}",
                value=sections.get(key, ""),
                height=280,
                key=f"edit_{key}",
                placeholder=f"Click 'Generate' to create the {label}, then edit as needed."
            )
            st.session_state.sections[key] = sections[key]


def page_figures_charts():
    st.markdown('<div class="section-header">📊 Figures & Charts</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload figures and generate charts from your data.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📷 Upload Figures", "📈 Generate Charts from CSV"])

    with tab1:
        uploaded_imgs = st.file_uploader("Upload figures (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="fig_uploader")
        if uploaded_imgs:
            for img_file in uploaded_imgs:
                data = img_file.read()
                if not any(f["name"] == img_file.name for f in st.session_state.figures):
                    st.session_state.figures.append({"name": img_file.name, "data": data})
            st.success(f"{len(uploaded_imgs)} figure(s) uploaded.")

        if st.session_state.figures:
            st.markdown(f"**{len(st.session_state.figures)} figure(s) stored:**")
            cols = st.columns(min(3, len(st.session_state.figures)))
            for i, fig in enumerate(st.session_state.figures):
                with cols[i % 3]:
                    st.image(fig["data"], caption=fig["name"], use_column_width=True)

    with tab2:
        csv_file = st.file_uploader("Upload CSV data file", type=["csv"], key="csv_uploader")
        if csv_file:
            try:
                df = pd.read_csv(csv_file)
                st.dataframe(df.head(10), use_container_width=True)
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                all_cols = df.columns.tolist()

                if not numeric_cols:
                    st.warning("No numeric columns found in CSV.")
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Pie Chart"])
                    with col2:
                        x_col = st.selectbox("X-axis / Labels", all_cols)
                    with col3:
                        y_col = st.selectbox("Y-axis / Values", numeric_cols)

                    chart_title = st.text_input("Chart Title", "Research Data Chart")

                    if st.button("Generate Chart"):
                        fig, ax = plt.subplots(figsize=(8, 5))
                        fig.patch.set_facecolor("#f8fafc")
                        ax.set_facecolor("#ffffff")

                        try:
                            if chart_type == "Bar Chart":
                                ax.bar(df[x_col].astype(str), df[y_col], color="#1e40af", edgecolor="white", linewidth=0.5)
                                ax.set_xlabel(x_col, fontsize=11)
                                ax.set_ylabel(y_col, fontsize=11)
                                plt.xticks(rotation=45, ha="right")
                            elif chart_type == "Line Chart":
                                ax.plot(df[x_col].astype(str), df[y_col], color="#1e40af", linewidth=2, marker="o", markersize=5)
                                ax.set_xlabel(x_col, fontsize=11)
                                ax.set_ylabel(y_col, fontsize=11)
                                plt.xticks(rotation=45, ha="right")
                            elif chart_type == "Pie Chart":
                                ax.pie(df[y_col], labels=df[x_col].astype(str), autopct="%1.1f%%", colors=plt.cm.Blues(range(50, 250, 200 // max(len(df), 1))))
                                ax.axis("equal")

                            ax.set_title(chart_title, fontsize=13, fontweight="bold", pad=15)
                            plt.tight_layout()

                            img_buf = io.BytesIO()
                            plt.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
                            img_buf.seek(0)
                            chart_data = img_buf.read()
                            plt.close()

                            st.session_state.charts.append({"name": chart_title, "data": chart_data})
                            st.image(chart_data, caption=chart_title, use_column_width=True)
                            st.success(f"Chart '{chart_title}' generated and saved.")
                        except Exception as e:
                            st.error(f"Chart generation failed: {e}")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")


def page_integrity_check():
    st.markdown('<div class="section-header">🔍 Research Integrity Check</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Review your paper\'s completeness before exporting.</div>', unsafe_allow_html=True)

    checks = integrity_check(
        st.session_state.project,
        st.session_state.authors,
        st.session_state.references,
        st.session_state.planner,
        st.session_state.sections
    )

    completed = sum(checks.values())
    total = len(checks)
    pct = int((completed / total) * 100)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Completed Items", f"{completed}/{total}")
    with col2:
        st.metric("Completion", f"{pct}%")
    with col3:
        status = "🟢 Ready to Export" if pct == 100 else ("🟡 Almost Ready" if pct >= 60 else "🔴 Needs Work")
        st.metric("Status", status)

    st.progress(pct / 100)
    st.markdown("---")
    st.markdown("**Checklist:**")

    for item, done in checks.items():
        if done:
            st.markdown(f'✅ &nbsp; <span class="badge-complete">Complete</span> &nbsp; **{item}**', unsafe_allow_html=True)
        else:
            st.markdown(f'⚠️ &nbsp; <span class="badge-missing">Missing</span> &nbsp; **{item}**', unsafe_allow_html=True)

    if pct < 100:
        missing = [k for k, v in checks.items() if not v]
        st.warning(f"Still missing: {', '.join(missing)}")
    else:
        st.success("🎉 Your research paper is complete and ready for export!")


def page_export():
    st.markdown('<div class="section-header">📦 Export Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Download your complete LaTeX project for Overleaf.</div>', unsafe_allow_html=True)

    checks = integrity_check(
        st.session_state.project, st.session_state.authors,
        st.session_state.references, st.session_state.planner, st.session_state.sections
    )
    pct = int((sum(checks.values()) / len(checks)) * 100)

    if pct < 40:
        st.warning("Your paper is less than 40% complete. We recommend finishing more sections before exporting.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**What's included in the ZIP:**")
        st.write("- `main.tex` — Full LaTeX document\n- `references.bib` — BibTeX references\n- `images/` — Uploaded figures\n- `charts/` — Generated charts")
    with col2:
        st.markdown("**How to use with Overleaf:**")
        st.write("1. Download the ZIP file\n2. Go to [overleaf.com](https://overleaf.com)\n3. New Project → Upload Project\n4. Upload the ZIP file\n5. Click Compile")

    st.markdown("---")

    with st.expander("Preview LaTeX (main.tex)", expanded=False):
        latex_preview = build_latex(
            st.session_state.project,
            st.session_state.authors,
            st.session_state.sections,
            st.session_state.bibtex
        )
        st.code(latex_preview, language="latex")

    if st.button("📦 Generate & Download ZIP", use_container_width=False):
        with st.spinner("Building your project ZIP..."):
            latex_content = build_latex(
                st.session_state.project,
                st.session_state.authors,
                st.session_state.sections,
                st.session_state.bibtex
            )
            zip_bytes = create_project_zip(
                latex_content,
                st.session_state.bibtex,
                st.session_state.figures,
                st.session_state.charts
            )

        project_name = st.session_state.project.get("title", "research_paper").replace(" ", "_")[:40]
        st.download_button(
            label="⬇️ Download project.zip",
            data=zip_bytes,
            file_name=f"{project_name}_latex.zip",
            mime="application/zip",
            use_container_width=False
        )
        st.success("ZIP created successfully! Click the button above to download.")


def main():
    init_session()

    with st.sidebar:
        st.markdown('<div class="sidebar-title">🎓 AI Research Generator</div>', unsafe_allow_html=True)
        st.markdown("")

        pages = {
            "📋 Project Setup": "Project Setup",
            "👥 Authors": "Authors",
            "📚 References": "References",
            "🗺️ Research Planner": "Research Planner",
            "✍️ Section Generator": "Section Generator",
            "📊 Figures & Charts": "Figures & Charts",
            "🔍 Integrity Check": "Integrity Check",
            "📦 Export": "Export"
        }

        selected = st.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
        st.session_state.active_page = pages[selected]

        st.markdown("---")
        checks = integrity_check(
            st.session_state.project, st.session_state.authors,
            st.session_state.references, st.session_state.planner, st.session_state.sections
        )
        pct = int((sum(checks.values()) / len(checks)) * 100)
        st.caption(f"Overall progress: {pct}%")
        st.progress(pct / 100)

        st.markdown("---")
        st.caption("Built for first-year researchers 📘")
        st.caption("Powered by Gemini 2.5 Flash Lite")

    page = st.session_state.active_page

    if page == "Project Setup":
        page_project_setup()
    elif page == "Authors":
        page_authors()
    elif page == "References":
        page_references()
    elif page == "Research Planner":
        page_research_planner()
    elif page == "Section Generator":
        page_section_generator()
    elif page == "Figures & Charts":
        page_figures_charts()
    elif page == "Integrity Check":
        page_integrity_check()
    elif page == "Export":
        page_export()


if __name__ == "__main__":
    main()
