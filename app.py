"""
app.py — Golden RAG Evaluation Studio

A production-quality Gradio UI for exploring RAG evaluation on deep learning
content from 3Blue1Brown's neural network series.

Tabs:
  📊 Dataset Viewer — QA evaluation set with filtering
  🔍 Ask Questions — RAG-powered Q&A simulation
  📄 Transcript Explorer — Timestamped transcript browser
"""

import gradio as gr
from transcript_extractor import fetch_transcripts, get_transcript_for_display, VIDEOS
from qa_generator import generate_qa_pairs, get_qa_table_data, get_methodology
from rag_engine import RAGEngine

# ─── Initialize Engine (cached for fast startup) ──────────────────────────────

print("🚀 Starting Golden RAG Evaluation Studio...\n")
transcripts = fetch_transcripts()
engine = RAGEngine()
qa_pairs = generate_qa_pairs()
stats = engine.get_stats()

# ─── Custom CSS ────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ─── FORCE ALL TEXT TO BE VISIBLE ────────────────────────────────────── */
/* This is the critical fix — Gradio defaults to dark text which is invisible
   on our dark background. We override EVERYTHING to white/light. */

.gradio-container, .gradio-container *,
.gradio-container p, .gradio-container span,
.gradio-container label, .gradio-container td,
.gradio-container th, .gradio-container li,
.gradio-container h1, .gradio-container h2,
.gradio-container h3, .gradio-container h4,
.gradio-container h5, .gradio-container h6 {
    color: #e2e8f0 !important;
}

/* Links should be a bright accent color */
.gradio-container a {
    color: #818cf8 !important;
}
.gradio-container a:hover {
    color: #a5b4fc !important;
}

/* Table text specifically */
.gradio-container table td,
.gradio-container table th,
.gradio-container .table-wrap td,
.gradio-container .table-wrap th,
.gradio-container .dataframe td,
.gradio-container .dataframe th,
.gradio-container [class*="table"] td,
.gradio-container [class*="table"] th {
    color: #e2e8f0 !important;
    border-color: rgba(99, 102, 241, 0.2) !important;
}

/* Table header row */
.gradio-container table thead th,
.gradio-container .table-wrap thead th {
    color: #c7d2fe !important;
    background: rgba(99, 102, 241, 0.15) !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    font-size: 0.8em !important;
    letter-spacing: 0.05em !important;
}

/* Table row hover */
.gradio-container table tbody tr:hover {
    background: rgba(99, 102, 241, 0.08) !important;
}

/* Input text */
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .input-container input,
.gradio-container [data-testid] input {
    color: #f1f5f9 !important;
    caret-color: #818cf8 !important;
}

/* Placeholder text */
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #64748b !important;
}

/* Dropdown text */
.gradio-container .dropdown-container,
.gradio-container [class*="dropdown"] span,
.gradio-container .secondary-wrap span {
    color: #e2e8f0 !important;
}

/* Markdown rendered content */
.gradio-container .markdown-text,
.gradio-container .markdown-text p,
.gradio-container .markdown-text li,
.gradio-container .markdown-text td,
.gradio-container .markdown-text th,
.gradio-container .markdown-text blockquote,
.gradio-container .prose,
.gradio-container .prose * {
    color: #cbd5e1 !important;
}

.gradio-container .markdown-text h1,
.gradio-container .markdown-text h2,
.gradio-container .markdown-text h3,
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {
    color: #e0e7ff !important;
}

.gradio-container .markdown-text strong,
.gradio-container .prose strong {
    color: #c7d2fe !important;
}

.gradio-container .markdown-text code,
.gradio-container .prose code {
    color: #67e8f9 !important;
    background: rgba(6, 182, 212, 0.1) !important;
    padding: 1px 6px !important;
    border-radius: 4px !important;
}

/* Blockquotes */
.gradio-container blockquote {
    border-left: 3px solid #6366f1 !important;
    background: rgba(99, 102, 241, 0.06) !important;
    color: #cbd5e1 !important;
    padding: 8px 16px !important;
    margin: 8px 0 !important;
    border-radius: 0 8px 8px 0 !important;
}

/* Tab Labels */
.gradio-container button[role="tab"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 1em !important;
}
.gradio-container button[role="tab"][aria-selected="true"] {
    color: #818cf8 !important;
    border-color: #6366f1 !important;
}

/* ─── Root Variables ─────────────────────────────────────────────────── */
:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --primary-light: #818cf8;
    --accent: #06b6d4;
    --accent-light: #67e8f9;
    --surface: #1e1b4b;
    --surface-light: #312e81;
    --bg-dark: #0a0a1a;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --success: #34d399;
    --warning: #fbbf24;
    --error: #f87171;
}

/* ─── Global Styles ──────────────────────────────────────────────────── */
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ─── Hero Banner ────────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #1a1145 0%, #2d1b69 50%, #1a1145 100%);
    border: 1px solid rgba(129, 140, 248, 0.3);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 30% 50%, rgba(99, 102, 241, 0.12) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 50%, rgba(6, 182, 212, 0.08) 0%, transparent 60%);
    animation: aurora 6s ease-in-out infinite alternate;
}
@keyframes aurora {
    0% { transform: translate(-5%, -5%) rotate(0deg); }
    100% { transform: translate(5%, 5%) rotate(3deg); }
}
.hero-banner h1 {
    font-size: 2.4em !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #818cf8, #67e8f9, #818cf8) !important;
    background-size: 200% auto !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    animation: shimmer 3s linear infinite !important;
    margin: 0 0 8px 0 !important;
    position: relative !important;
    letter-spacing: -0.02em !important;
}
@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.hero-banner .subtitle {
    color: #c7d2fe !important;
    font-size: 1.15em;
    margin: 4px 0;
    position: relative;
    font-weight: 500;
}
.hero-banner .powered-by {
    color: #64748b !important;
    font-size: 0.9em;
    margin: 8px 0 0 0;
    position: relative;
}

/* ─── Stats Cards ────────────────────────────────────────────────────── */
.stats-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.stat-card {
    flex: 1;
    min-width: 140px;
    background: linear-gradient(145deg, rgba(30, 27, 75, 0.8), rgba(45, 27, 105, 0.5));
    border: 1px solid rgba(129, 140, 248, 0.2);
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}
.stat-card:hover {
    transform: translateY(-3px);
    border-color: rgba(129, 140, 248, 0.4);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
}
.stat-card .stat-value {
    font-size: 2.2em;
    font-weight: 800;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    text-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
}
.stat-card .stat-label {
    font-size: 0.8em;
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

/* ─── Card Containers ────────────────────────────────────────────────── */
.card {
    background: rgba(30, 27, 75, 0.5);
    border: 1px solid rgba(129, 140, 248, 0.15);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
}

/* ─── Methodology tables in markdown ─────────────────────────────────── */
.gradio-container .markdown-text table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 16px 0 !important;
    background: rgba(15, 12, 41, 0.5) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
.gradio-container .markdown-text table th {
    background: rgba(99, 102, 241, 0.15) !important;
    padding: 10px 14px !important;
    text-align: left !important;
    font-weight: 700 !important;
    color: #c7d2fe !important;
    border-bottom: 2px solid rgba(99, 102, 241, 0.3) !important;
}
.gradio-container .markdown-text table td {
    padding: 10px 14px !important;
    border-bottom: 1px solid rgba(99, 102, 241, 0.1) !important;
    color: #cbd5e1 !important;
}

/* ─── Secondary buttons ──────────────────────────────────────────────── */
.gradio-container button.secondary,
.gradio-container button[variant="secondary"] {
    background: rgba(99, 102, 241, 0.1) !important;
    border: 1px solid rgba(129, 140, 248, 0.3) !important;
    color: #c7d2fe !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
.gradio-container button.secondary:hover,
.gradio-container button[variant="secondary"]:hover {
    background: rgba(99, 102, 241, 0.2) !important;
    border-color: rgba(129, 140, 248, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* ─── Primary button glow ────────────────────────────────────────────── */
.gradio-container button.primary {
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3) !important;
    transition: all 0.2s ease !important;
}
.gradio-container button.primary:hover {
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* ─── Dataframe / Table container ────────────────────────────────────── */
.gradio-container .table-wrap,
.gradio-container .dataframe {
    background: rgba(15, 12, 41, 0.6) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
    overflow: hidden !important;
}
.gradio-container .table-wrap tbody tr:nth-child(even) {
    background: rgba(99, 102, 241, 0.04) !important;
}

/* ─── Textarea / Input focus ring ────────────────────────────────────── */
.gradio-container textarea:focus,
.gradio-container input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    outline: none !important;
}

/* ─── Scrollbar styling ──────────────────────────────────────────────── */
.gradio-container ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
.gradio-container ::-webkit-scrollbar-track {
    background: rgba(15, 12, 41, 0.5);
    border-radius: 4px;
}
.gradio-container ::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.3);
    border-radius: 4px;
}
.gradio-container ::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.5);
}

/* ─── Gradio block spacing fixes ─────────────────────────────────────── */
.gradio-container .block {
    border: none !important;
    box-shadow: none !important;
}
"""


# ─── Helper Functions ──────────────────────────────────────────────────────────

def build_stats_html():
    """Build the stats cards HTML."""
    return f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value">{stats['total_videos']}</div>
            <div class="stat-label">Videos Indexed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['total_chunks']}</div>
            <div class="stat-label">Text Chunks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['embedding_dim']}</div>
            <div class="stat-label">Embedding Dims</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(qa_pairs)}</div>
            <div class="stat-label">QA Pairs</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{'✓' if stats['cache_exists'] else '✗'}</div>
            <div class="stat-label">Cache Status</div>
        </div>
    </div>
    """


def handle_rag_query(question: str):
    """Handle a user question through the RAG pipeline."""
    if not question or not question.strip():
        return (
            "⚠️ Please enter a question.",
            "No context retrieved.",
            "No sources."
        )

    result = engine.query(question.strip())

    # Format answer
    answer = result["answer"]

    # Format contexts
    ctx_parts = []
    for i, ctx in enumerate(result["contexts"], 1):
        ctx_parts.append(
            f"### 📎 Chunk {i} (Relevance: {ctx['score']:.2%})\n"
            f"**Source:** {ctx['video_title']}  \n"
            f"**Timestamp:** {ctx['timestamp']}  \n"
            f"**URL:** [{ctx['video_url']}]({ctx['video_url']})  \n\n"
            f"> {ctx['text'][:400]}{'...' if len(ctx['text']) > 400 else ''}\n\n"
            f"---"
        )
    contexts = "\n\n".join(ctx_parts) if ctx_parts else "No context retrieved."

    # Format sources
    sources = "\n".join([f"• {s}" for s in result["sources"]]) if result["sources"] else "No sources found."

    return answer, contexts, sources


def get_transcript_display(video_choice: str):
    """Get formatted transcript for the selected video."""
    for vid_id, meta in VIDEOS.items():
        if meta["title"] in video_choice:
            return get_transcript_for_display(transcripts, vid_id)
    return "Please select a video."


def get_video_choices():
    """Get list of video titles for dropdown."""
    return [f"{meta['title']}" for vid_id, meta in VIDEOS.items() if vid_id in transcripts]


# ─── Build Gradio App ─────────────────────────────────────────────────────────

with gr.Blocks(
    css=CUSTOM_CSS,
    title="Golden RAG Evaluation Studio",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        # Background
        body_background_fill="#0a0a1a",
        body_background_fill_dark="#0a0a1a",
        background_fill_primary="#0a0a1a",
        background_fill_primary_dark="#0a0a1a",
        background_fill_secondary="rgba(20, 18, 50, 0.85)",
        background_fill_secondary_dark="rgba(20, 18, 50, 0.85)",
        # Blocks
        block_background_fill="rgba(20, 18, 50, 0.85)",
        block_background_fill_dark="rgba(20, 18, 50, 0.85)",
        block_border_color="rgba(99, 102, 241, 0.15)",
        block_border_color_dark="rgba(99, 102, 241, 0.15)",
        block_label_text_color="#94a3b8",
        block_label_text_color_dark="#94a3b8",
        block_title_text_color="#e2e8f0",
        block_title_text_color_dark="#e2e8f0",
        # Panel
        panel_background_fill="rgba(15, 12, 41, 0.6)",
        panel_background_fill_dark="rgba(15, 12, 41, 0.6)",
        panel_border_color="rgba(99, 102, 241, 0.15)",
        panel_border_color_dark="rgba(99, 102, 241, 0.15)",
        # Inputs
        input_background_fill="rgba(15, 12, 41, 0.9)",
        input_background_fill_dark="rgba(15, 12, 41, 0.9)",
        input_border_color="rgba(99, 102, 241, 0.3)",
        input_border_color_dark="rgba(99, 102, 241, 0.3)",
        input_placeholder_color="#64748b",
        input_placeholder_color_dark="#64748b",
        # Buttons
        button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
        button_primary_background_fill_dark="linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #818cf8 0%, #6366f1 100%)",
        button_primary_background_fill_hover_dark="linear-gradient(135deg, #818cf8 0%, #6366f1 100%)",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
        # Table — with visible text
        table_border_color="rgba(99, 102, 241, 0.15)",
        table_border_color_dark="rgba(99, 102, 241, 0.15)",
        table_text_color="#e2e8f0",
        table_text_color_dark="#e2e8f0",
        table_even_background_fill="rgba(99, 102, 241, 0.04)",
        table_even_background_fill_dark="rgba(99, 102, 241, 0.04)",
        table_odd_background_fill="transparent",
        table_odd_background_fill_dark="transparent",
        table_row_focus="rgba(99, 102, 241, 0.1)",
        table_row_focus_dark="rgba(99, 102, 241, 0.1)",
        # Stat cards
        stat_background_fill="rgba(30, 27, 75, 0.6)",
        stat_background_fill_dark="rgba(30, 27, 75, 0.6)",
        # Links
        link_text_color="#818cf8",
        link_text_color_dark="#818cf8",
        link_text_color_hover="#a5b4fc",
        link_text_color_hover_dark="#a5b4fc",
        link_text_color_active="#6366f1",
        link_text_color_active_dark="#6366f1",
        link_text_color_visited="#818cf8",
        link_text_color_visited_dark="#818cf8",
        # Shadows
        shadow_spread="0 4px 16px rgba(0, 0, 0, 0.2)",
        shadow_spread_dark="0 4px 16px rgba(0, 0, 0, 0.2)",
        # Loader
        loader_color="#6366f1",
        loader_color_dark="#6366f1",
        # Slider
        slider_color="#6366f1",
        slider_color_dark="#6366f1",
        # Color accents
        color_accent="#6366f1",
        color_accent_soft="rgba(99, 102, 241, 0.15)",
    ),
) as app:

    # ─── Hero Banner ──────────────────────────────────────────────────────
    gr.HTML("""
        <div class="hero-banner">
            <h1>🧠 Golden RAG Evaluation Studio</h1>
            <p class="subtitle">Multi-Hop RAG Evaluation for Deep Learning Content</p>
            <p class="powered-by">
                Powered by 3Blue1Brown Neural Network Series • sentence-transformers • Cosine Similarity
            </p>
        </div>
    """)

    # ─── Stats Bar ────────────────────────────────────────────────────────
    gr.HTML(build_stats_html())

    # ─── Tabs ─────────────────────────────────────────────────────────────
    with gr.Tabs() as tabs:

        # ──────────────────────────────────────────────────────────────────
        # TAB 1: Dataset Viewer
        # ──────────────────────────────────────────────────────────────────
        with gr.Tab("📊 Dataset Viewer", id="dataset"):
            gr.Markdown("### 📋 RAG Evaluation QA Pairs")
            gr.Markdown(
                "These question-answer pairs are designed to test retrieval accuracy, "
                "multi-hop reasoning, and cross-source synthesis capabilities."
            )

            qa_data = get_qa_table_data()
            gr.Dataframe(
                value=qa_data,
                headers=["Question", "Answer", "Source", "Type"],
                datatype=["str", "str", "str", "str"],
                wrap=True,
                column_widths=["30%", "35%", "20%", "15%"],
                interactive=False,
            )

            # Methodology section
            gr.Markdown("---")
            gr.Markdown(get_methodology())

        # ──────────────────────────────────────────────────────────────────
        # TAB 2: Ask Questions (RAG Simulation)
        # ──────────────────────────────────────────────────────────────────
        with gr.Tab("🔍 Ask Questions", id="ask"):
            gr.Markdown("### 🔎 RAG-Powered Question Answering")
            gr.Markdown(
                "Ask any question about neural networks and deep learning. "
                "The system retrieves relevant transcript chunks and generates "
                "a grounded answer."
            )

            with gr.Row():
                with gr.Column(scale=4):
                    query_input = gr.Textbox(
                        placeholder="e.g. How does backpropagation compute gradients for weights?",
                        label="Your Question",
                        lines=2,
                        max_lines=4,
                    )
                with gr.Column(scale=1, min_width=120):
                    submit_btn = gr.Button(
                        "🔍 Search",
                        variant="primary",
                        size="lg",
                    )

            # Quick preset questions
            gr.Markdown("**💡 Try these questions:**")
            with gr.Row():
                preset_1 = gr.Button("What is a neural network?", size="sm", variant="secondary")
                preset_2 = gr.Button("How does gradient descent work?", size="sm", variant="secondary")
                preset_3 = gr.Button("What is backpropagation?", size="sm", variant="secondary")
                preset_4 = gr.Button("Why is ReLU better than sigmoid?", size="sm", variant="secondary")

            gr.Markdown("---")

            # Answer section with labeled areas
            gr.Markdown("#### 📝 Answer")
            answer_output = gr.Markdown(
                value="*Ask a question to see the RAG-generated answer.*",
            )

            gr.Markdown("#### 📎 Retrieved Context")
            context_output = gr.Markdown(
                value="*Context chunks will appear here.*",
            )

            gr.Markdown("#### 📍 Sources")
            sources_output = gr.Markdown(
                value="*Sources will appear here.*",
            )

            # Wire up events
            submit_btn.click(
                fn=handle_rag_query,
                inputs=[query_input],
                outputs=[answer_output, context_output, sources_output],
            )
            query_input.submit(
                fn=handle_rag_query,
                inputs=[query_input],
                outputs=[answer_output, context_output, sources_output],
            )

            # Preset buttons
            def set_and_query(q):
                a, c, s = handle_rag_query(q)
                return [q, a, c, s]

            for btn, q_text in [
                (preset_1, "What is a neural network?"),
                (preset_2, "How does gradient descent work?"),
                (preset_3, "What is backpropagation?"),
                (preset_4, "Why is ReLU better than sigmoid?"),
            ]:
                btn.click(
                    fn=set_and_query,
                    inputs=[gr.State(q_text)],
                    outputs=[query_input, answer_output, context_output, sources_output],
                )

        # ──────────────────────────────────────────────────────────────────
        # TAB 3: Transcript Explorer
        # ──────────────────────────────────────────────────────────────────
        with gr.Tab("📄 Transcript Explorer", id="transcripts"):
            gr.Markdown("### 📜 Video Transcript Browser")
            gr.Markdown("Select a video to explore its full transcript with timestamps.")

            video_choices = get_video_choices()
            video_dropdown = gr.Dropdown(
                choices=video_choices,
                label="🎬 Select Video",
                value=video_choices[0] if video_choices else None,
                interactive=True,
            )

            transcript_display = gr.Textbox(
                label="📄 Transcript",
                lines=25,
                max_lines=40,
                interactive=False,
            )

            # Load initial transcript
            if video_choices:
                transcript_display.value = get_transcript_display(video_choices[0])

            video_dropdown.change(
                fn=get_transcript_display,
                inputs=[video_dropdown],
                outputs=[transcript_display],
            )


# ─── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
