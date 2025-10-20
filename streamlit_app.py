import io
import os
import tempfile
import threading
from typing import List, Dict, Any, Optional
import streamlit as st
from PIL import Image
import numpy as np
import faiss
from datetime import datetime

# Set environment variables to help with OpenCV in cloud environments
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['OPENCV_IO_ENABLE_JASPER'] = '1'

# Handle OpenCV import for cloud environments
try:
    import cv2
    # Test basic OpenCV functionality
    test_img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imencode('.jpg', test_img)
except Exception as e:
    st.error(f"OpenCV initialization failed: {e}")
    st.error("This app requires OpenCV to work properly. Please check the deployment logs.")
    st.stop()

try:
    from deepface import DeepFace
except ImportError as e:
    st.error(f"DeepFace import failed: {e}")
    st.stop()

# Import our modules
try:
    from services import RecognitionService
    from models import RecognizeItem, RecognizePerImage, RecognizeBatchResponse
except ImportError as e:
    st.error(f"Local module import failed: {e}")
    st.stop()


def process_images_directly(files: List[bytes], filenames: List[str]) -> Dict[str, Any]:
    """Process images directly using the recognition service"""
    try:
        service = RecognitionService.instance()
        outputs: List[RecognizePerImage] = []
        
        for file_bytes, filename in zip(files, filenames):
            if not filename:
                outputs.append(RecognizePerImage(filename="", num_faces=0, results=[]))
                continue
            
            try:
                # Convert bytes to numpy array and decode
                np_arr = np.frombuffer(file_bytes, np.uint8)
                if len(np_arr) == 0:
                    outputs.append(RecognizePerImage(filename=filename, num_faces=0, results=[]))
                    continue
                    
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None:
                    outputs.append(RecognizePerImage(filename=filename, num_faces=0, results=[]))
                    continue
                
                # Save to temporary file for DeepFace processing
                suffix = os.path.splitext(filename)[1] or ".jpg"
                fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="upload_")
                os.close(fd)
                cv2.imwrite(tmp_path, img)
                
                # Process with recognition service
                results_raw = service.recognize_image_path(tmp_path)
                items = [RecognizeItem(**r) for r in results_raw]
                outputs.append(RecognizePerImage(filename=filename, num_faces=len(items), results=items))
                
            except Exception as e:
                st.error(f"Error processing {filename}: {e}")
                outputs.append(RecognizePerImage(filename=filename, num_faces=0, results=[]))
            finally:
                # Clean up temporary file
                try:
                    if 'tmp_path' in locals():
                        os.remove(tmp_path)
                except OSError:
                    pass
        
        return {"items": [item.dict() for item in outputs]}
        
    except Exception as e:
        st.error(f"Error processing images: {e}")
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
                        st.caption("Added to recognition database")
                
                with col2:
                    st.write(f"**Face {i+1}**")
                    if result["is_returning"]:
                        st.write("This face matches an existing customer in the database.")
                    else:
                        st.write("This is a new face that has been added to the recognition database.")

def main():
    st.set_page_config(
        page_title="Face Recognition System",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("👤 Face Recognition System")
    st.markdown("Upload images to detect and recognize faces using our AI-powered system.")
    
    # Sidebar for system status
    with st.sidebar:
        st.header("🔧 System Status")
        try:
            # Check if recognition service is available
            service = RecognitionService.instance()
            index_size = service.index.ntotal
            st.success("✅ Face Recognition Ready")
            st.metric("Database Size", f"{index_size} faces")
        except Exception as e:
            st.error("❌ System Error")
            st.caption(f"Error: {str(e)}")
        
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
                    # Process images directly
                    results = process_images_directly(files_data, filenames)
                    
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
        "Face Recognition System powered by Streamlit + DeepFace"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
