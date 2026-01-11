import streamlit as st
from PIL import Image
import numpy as np
from retrieval import ImageRetrievalSystem
import os
import time

st.set_page_config(page_title="Fashion Image Retrieval with Relevant Feedback", layout="wide")

@st.cache_resource
def get_retrieval_system():
    system = ImageRetrievalSystem(dataset_limit=1000, batch_size=32)
    system.initialize()
    return system

system = get_retrieval_system()

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'feedback_rel' not in st.session_state:
    st.session_state.feedback_rel = set()
if 'feedback_irr' not in st.session_state:
    st.session_state.feedback_irr = set()
if 'search_id' not in st.session_state:
    st.session_state.search_id = 0
if 'feedback_state' not in st.session_state:
    st.session_state.feedback_state = {}

with st.sidebar:
    st.title("Search Options")
    
    # 1. Text query
    st.subheader("Text Search")
    text_query = st.text_input("Enter product name (e.g., 'Shoes'):", "Apparel")
    if st.button("Search by Text", use_container_width=True):
        indices, scores = system.search_by_text(text_query, top_k=20)
        st.session_state.results = (indices, scores)
        st.session_state.feedback_rel.clear()
        st.session_state.feedback_irr.clear()
        st.session_state.feedback_state = {}
        st.session_state.search_id += 1
        st.rerun()

    st.divider()

    # 2. Upload image
    st.subheader("Image Search")
    uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Query Image", width=150)
        if st.button("Search by Image", use_container_width=True):
            indices, scores = system.search_by_uploaded_image(img, top_k=20)
            st.session_state.results = (indices, scores)
            st.session_state.feedback_rel.clear()
            st.session_state.feedback_irr.clear()
            st.session_state.feedback_state = {}
            st.session_state.search_id += 1
            st.rerun()

    st.divider()

    # 3. System stats
    st.subheader("System Stats")
    stats = system.get_session_stats()
    st.write(f"**Dataset Size:** {len(system.image_ids)}")
    if stats:
        st.write(f"**Iteration:** {stats['iteration']}")
        st.write(f"**Relevant:** {stats['n_relevant']}")
        st.write(f"**Irrelevant:** {stats['n_irrelevant']}")
        st.write(f"**Drift:** {stats['query_drift']:.4f}")

st.title("Fashion Product Retrieval System")

if st.session_state.results:
    indices, scores = st.session_state.results
    
    # Feedback controls
    col_fb1, col_fb2 = st.columns([4, 1])
    # with col_fb1:
    #     st.info(f"Selected: {len(st.session_state.feedback_rel)} Relevant (✓), {len(st.session_state.feedback_irr)} Irrelevant (✗)")
    with col_fb2:
        if st.button("🔄 Refine Results", type="primary", use_container_width=True):
            if not st.session_state.feedback_rel and not st.session_state.feedback_irr:
                st.warning("Please select at least one relevant or irrelevant image!")
            else:
                new_indices, new_scores = system.apply_relevance_feedback(
                    list(st.session_state.feedback_rel),
                    list(st.session_state.feedback_irr),
                    top_k=20
                )
                st.session_state.results = (new_indices, new_scores)
                st.session_state.feedback_rel.clear()
                st.session_state.feedback_irr.clear()
                st.session_state.feedback_state = {}
                st.session_state.search_id += 1
                st.rerun()
    
    def update_feedback(idx, choice):
        """Callback to update feedback state"""
        if choice == "Relevant ✓":
            st.session_state.feedback_rel.add(idx)
            st.session_state.feedback_irr.discard(idx)
        elif choice == "Irrelevant ✗":
            st.session_state.feedback_irr.add(idx)
            st.session_state.feedback_rel.discard(idx)
        else:  # "None"
            st.session_state.feedback_rel.discard(idx)
            st.session_state.feedback_irr.discard(idx)
        st.session_state.feedback_state[idx] = choice

    # Display results
    cols = st.columns(4)
    for i, (idx, score) in enumerate(zip(indices, scores)):
        with cols[i % 4]:
            try:
                img = system.get_image(idx)
                label = system.get_label(idx)
                
                st.image(img, use_container_width=True)
                st.caption(f"**{label}** | Score: {score:.3f}")

                default_idx = 0
                if idx in st.session_state.feedback_state:
                    choice = st.session_state.feedback_state[idx]
                    if choice == "Relevant ✓":
                        default_idx = 1
                    elif choice == "Irrelevant ✗":
                        default_idx = 2
                
                feedback_choice = st.radio(
                    "Feedback:",
                    options=["None", "Relevant ✓", "Irrelevant ✗"],
                    key=f"fb_{st.session_state.search_id}_{idx}_{i}",
                    index=default_idx,
                    horizontal=True,
                    label_visibility="collapsed",
                    on_change=update_feedback,
                    args=(idx, ),
                    kwargs={'choice': None}
                )
                
                if feedback_choice == "Relevant ✓":
                    st.session_state.feedback_rel.add(idx)
                    st.session_state.feedback_irr.discard(idx)
                elif feedback_choice == "Irrelevant ✗":
                    st.session_state.feedback_irr.add(idx)
                    st.session_state.feedback_rel.discard(idx)
                else:  # "None"
                    st.session_state.feedback_rel.discard(idx)
                    st.session_state.feedback_irr.discard(idx)
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error loading image {idx}: {e}")
else:
    st.info("👈 Please enter a query in the sidebar to begin searching.")
