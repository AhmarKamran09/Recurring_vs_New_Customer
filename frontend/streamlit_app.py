import io
import requests
from typing import List, Dict, Any
import streamlit as st
from PIL import Image
import base64

# Configuration
FASTAPI_URL = "http://localhost:8000"


def upload_files_to_api(files: List[bytes], filenames: List[str]) -> Dict[str, Any]:
    """Upload files to FastAPI backend"""
    try:
        files_data = []
        for file_bytes, filename in zip(files, filenames):
            # Determine MIME type based on file extension
            if filename.lower().endswith('.png'):
                mime_type = "image/png"
            elif filename.lower().endswith(('.jpg', '.jpeg')):
                mime_type = "image/jpeg"
            else:
                mime_type = "image/jpeg"  # default
            
            files_data.append(("files", (filename, file_bytes, mime_type)))
        
        response = requests.post(f"{FASTAPI_URL}/api/recognize-batch", files=files_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
        return None

def display_results(results: Dict[str, Any]):
    """Display recognition results"""
    if not results or "items" not in results:
        st.error("No results to display")
        return
    
    for item in results["items"]:
        with st.expander(f"📷 {item['filename']} - {item['num_faces']} face(s) detected"):
            if item["num_faces"] == 0:
                st.warning("No faces detected in this image")
                continue
            
            for i, result in enumerate(item["results"]):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if result["is_returning"]:
                        st.success(f"✅ Returning Customer")
                        st.metric("Similarity", f"{result['similarity']:.3f}")
                    else:
                        st.info(f"🆕 New Customer")
                        st.metric("Similarity", f"{result['similarity']:.3f}")
                        if result.get("saved_path"):
                            st.caption(f"Saved to: {result['saved_path']}")
                
                with col2:
                    st.write(f"**Face {i+1}**")
                    if result["is_returning"]:
                        st.write("This face matches an existing customer in the database.")
                    else:
                        st.write("This is a new face that has been added to the database.")

def main():
    st.set_page_config(
        page_title="Face Recognition System",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("👤 Face Recognition System")
    st.markdown("Upload images to detect and recognize faces using our AI-powered system.")
    
    # Sidebar for API status
    with st.sidebar:
        st.header("🔧 System Status")
        try:
            # Try to ping the API (we'll add a health endpoint later)
            response = requests.get(f"{FASTAPI_URL}/docs", timeout=5)
            st.success("✅ API Connected")
        except:
            st.error("❌ API Not Available")
            st.caption("Make sure FastAPI server is running on port 8000")
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Upload one or more images
        2. Click 'Process Images'
        3. View recognition results
        4. New faces are automatically added to the database
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Images")
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload one or more images containing faces"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) selected")
            
            # Display uploaded images
            st.subheader("📷 Preview")
            for i, uploaded_file in enumerate(uploaded_files):
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_column_width=True)
    
    with col2:
        st.header("🔍 Results")
        
        if uploaded_files:
            if st.button("🚀 Process Images", type="primary"):
                with st.spinner("Processing images..."):
                    # Convert uploaded files to bytes
                    files_data = []
                    filenames = []
                    
                    for uploaded_file in uploaded_files:
                        # Reset file pointer to beginning
                        uploaded_file.seek(0)
                        file_bytes = uploaded_file.read()
                        files_data.append(file_bytes)
                        filenames.append(uploaded_file.name)
                    # Send to API
                    results = upload_files_to_api(files_data, filenames)
                    
                    if results:
                        st.success("✅ Processing completed!")
                        display_results(results)
                    else:
                        st.error("❌ Processing failed")
        else:
            st.info("👆 Upload images to get started")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Face Recognition System powered by FastAPI + Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
