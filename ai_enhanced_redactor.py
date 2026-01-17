#!/usr/bin/env python3

"""
AI-Enhanced PDF Redaction Tool with DeepSeek Vision API

This script enhances the existing PDF redaction system by using DeepSeek's vision AI
to identify logos, branding elements, and other visual content that should be redacted.
It combines traditional pattern-based redaction with AI-powered visual analysis.
"""

import os
import re
import sys
import time
import logging
import pickle
import base64
import json
import requests
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import io
import glob

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
# Google Drive API imports
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# Import file tracking system
from file_tracking import FileTracker

# Import YOLO logo detector
try:
    from yolo_logo_detector import YOLOLogoDetector, create_yolo_detector
    YOLO_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"YOLO detector not available: {e}")
    YOLO_DETECTOR_AVAILABLE = False

# DeepSeek API Configuration
DEEPSEEK_API_KEY = "sk-a72814c0cb7c4d1480d19f0d2dc42a68"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Google Drive settings (inherited from existing system)
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/spreadsheets'
]
CLIENT_SECRET_FILE = 'client_secret_867948760205-cvbu5qdv6buq8863ola7vn3924b7hte0.apps.googleusercontent.com.json'
TOKEN_PICKLE_FILE = 'token.pickle'
DRIVE_INPUT_FOLDER_ID = '1oRwJHz2eeyDTm0hn6k6XlcII8o67BY88'
DRIVE_OUTPUT_FOLDER_ID = '1oRwJHz2eeyDTm0hn6k6XlcII8o67BY88'

# Google Sheets logging
SHEETS_SPREADSHEET_ID = '14mrL0LjwhuN0KxZ2AZ_aSrOcIImmyzC-Ys5hjQFT43Y'
SHEETS_RANGE = 'Sheet1!A:B'

# Branding configuration - Default to Precon Factory
DEFAULT_WATERMARK_TEXT = 'Precon Factory'
DEFAULT_FOOTER_IMAGE_URL = 'https://cfzuypbljirmibmxpabi.supabase.co/storage/v1/object/public/email-images/footer/footer%20precon%20factory.png'

# Fahad Javed branding configuration
FAHAD_WATERMARK_TEXT = 'Fahad Javed'
FAHAD_FOOTER_IMAGE_URL = 'https://cfzuypbljirmibmxpabi.supabase.co/storage/v1/object/public/email-images/fahad%20javed%20footer.png'

# GTA Lowrise branding configuration
GTA_LOWRISE_WATERMARK_TEXT = 'GTA Lowrise 416.399.4289'
GTA_LOWRISE_FOOTER_IMAGE_URL = 'https://cfzuypbljirmibmxpabi.supabase.co/storage/v1/object/public/email-images/footer/footer%20gta%20lowrise.png'

# Common watermark settings
WATERMARK_OPACITY = 0.30  # Light and subtle like a typical watermark
WATERMARK_ANGLE_DEGREES = 35

# Temporary processing folder
TEMP_PROCESSING_FOLDER = "./temp_processing/"

# Logo detection settings
LOGOS_FOLDER = "./logos/"
LOGO_MATCH_THRESHOLD = 0.7  # Template matching threshold
LOGO_MIN_SIZE = (20, 20)    # Minimum logo size to detect
LOGO_MAX_SIZE = (500, 500)  # Maximum logo size to detect

# Polling interval (seconds) for continuous mode
POLLING_INTERVAL_SECONDS = 5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pdf_redaction_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ai_enhanced_redactor")

# AI Vision Analysis Prompt
AI_ANALYSIS_PROMPT = """
You are an expert AI analyzing a real estate PDF page to identify ALL builder/developer branding and contact information that must be redacted before sharing with clients. Your job is to find EVERY piece of builder information while preserving property details.

### CRITICAL REDACTION TARGETS (find ALL of these):

1. **ALL LOGOS AND BRANDING**:
   - Builder/developer company logos (any size, any position)
   - Real estate brokerage logos and branding
   - Marketing company logos and watermarks
   - Small icons, emblems, or graphical elements with company names
   - QR codes linking to builder websites
   - Social media icons and handles

2. **ALL CONTACT INFORMATION**:
   - Phone numbers (any format: (555) 123-4567, 555-123-4567, etc.)
   - Email addresses (sales@builder.com, info@developer.ca, etc.)
   - Website URLs (www.builder.com, builder.ca, etc.)
   - Office addresses and contact details
   - Sales representative names and titles

3. **ALL BUILDER/DEVELOPER REFERENCES**:
   - Company names anywhere on the page
   - "Exclusive listing brokerage" information
   - Sales team information and agent details
   - Marketing slogans and taglines
   - Copyright notices with builder names

4. **HEADER/FOOTER BRANDING**:
   - Any branding in top 15% or bottom 15% of page
   - Company headers and footers
   - Legal disclaimers with builder names

### PRESERVE (DO NOT REDACT):
- Property prices, square footage, unit details
- Floor plans and building renderings
- Amenities, features, and specifications
- Occupancy dates and maintenance fees
- Property photos without branding

### DETECTION INSTRUCTIONS:
- Use your vision capabilities to identify visual logos and branding elements
- Look for text patterns that indicate builder/developer information
- Be thorough - find even small or subtle branding elements
- Return precise bounding boxes for each detected element

### RESPONSE FORMAT:
Return ONLY valid JSON in this exact format:

{
  "redaction_zones": [
    {
      "type": "logo|branding|contact|builder_info|header|footer",
      "description": "Specific description of what was found",
      "bbox": [x, y, width, height],
      "confidence": 0.95
    }
  ],
  "analysis_summary": "Brief summary of key redactions found"
}

Coordinates must be in pixels relative to the image dimensions. Be precise and thorough in your detection.
"""

class AIEnhancedRedactor:
    def __init__(self):
        self.deepseek_headers = {
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
            'Content-Type': 'application/json'
        }
        self.logo_templates = self._load_logo_templates()
        
        # Initialize YOLO detector if available
        self.yolo_detector = None
        if YOLO_DETECTOR_AVAILABLE:
            try:
                # Use higher confidence threshold to avoid false positives
                self.yolo_detector = create_yolo_detector(confidence_threshold=0.7)
                if self.yolo_detector:
                    logger.info("YOLO logo detector initialized successfully")
                else:
                    logger.warning("Failed to initialize YOLO detector")
            except Exception as e:
                logger.error(f"Error initializing YOLO detector: {e}")
                self.yolo_detector = None
        
    def image_to_base64(self, image_bytes: bytes) -> str:
        """Convert image bytes to base64 string for API transmission."""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def analyze_page_with_ai(self, page_image_bytes: bytes) -> Dict[str, Any]:
        """
        Send page image to DeepSeek API for AI-powered vision analysis.
        Falls back to enhanced local analysis if API fails.
        
        Args:
            page_image_bytes: PNG image bytes of the PDF page
            
        Returns:
            Dictionary containing redaction zones and analysis
        """
        try:
            # Convert image to base64
            image_b64 = self.image_to_base64(page_image_bytes)
            
            # Try different DeepSeek API formats
            api_formats = [
                # Format 1: Try with vision model and proper structure
                {
                    "model": "deepseek-vl-7b-chat",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": AI_ANALYSIS_PROMPT
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_b64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.1
                },
                # Format 2: Try with regular chat model and image description
                {
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{AI_ANALYSIS_PROMPT}\n\nPlease analyze this image for logos and branding elements. Return only valid JSON in the specified format."
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"}
                },
                # Format 3: Try with coder model
                {
                    "model": "deepseek-coder",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{AI_ANALYSIS_PROMPT}\n\nAnalyze the image and return JSON format as specified."
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"}
                }
            ]
            
            # Try each format
            for i, payload in enumerate(api_formats):
                logger.info(f"Trying DeepSeek API format {i+1}...")
                
                try:
                    response = requests.post(
                        DEEPSEEK_API_URL,
                        headers=self.deepseek_headers,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        content = result['choices'][0]['message']['content']
                        
                        # Extract JSON from response
                        try:
                            # Clean the response content
                            content = content.strip()
                            
                            # Try to find JSON in the response
                            json_start = content.find('{')
                            json_end = content.rfind('}') + 1
                            
                            if json_start >= 0 and json_end > json_start:
                                json_str = content[json_start:json_end]
                                analysis = json.loads(json_str)
                                
                                # Validate the analysis structure
                                if 'redaction_zones' in analysis and isinstance(analysis['redaction_zones'], list):
                                    logger.info(f"✅ DeepSeek API Analysis (format {i+1}): {analysis.get('analysis_summary', 'No summary')}")
                                    logger.info(f"Found {len(analysis['redaction_zones'])} redaction zones")
                                    return analysis
                                else:
                                    logger.warning(f"Invalid analysis structure from DeepSeek format {i+1}")
                                    continue
                            else:
                                logger.warning(f"No JSON found in DeepSeek response format {i+1}")
                                continue
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse DeepSeek response JSON format {i+1}: {e}")
                            continue
                    else:
                        logger.error(f"DeepSeek API error format {i+1}: {response.status_code} - {response.text}")
                        continue
                        
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request error format {i+1}: {e}")
                    continue
            
            # If all formats failed, use fallback
            logger.warning("All DeepSeek API formats failed, using enhanced computer vision fallback")
            return self._enhanced_computer_vision_analysis(page_image_bytes)
                
        except Exception as e:
            logger.error(f"Error in DeepSeek AI analysis: {str(e)}")
            return self._enhanced_computer_vision_analysis(page_image_bytes)
    
    def _enhanced_computer_vision_analysis(self, page_image_bytes: bytes) -> Dict[str, Any]:
        """
        Advanced computer vision analysis using OpenCV and image processing.
        Detects logos, text regions, and branding elements.
        """
        try:
            import cv2
            import numpy as np
            from PIL import Image
            import io
            
            # Load image
            image = Image.open(io.BytesIO(page_image_bytes))
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # Convert to OpenCV format
            if len(img_array.shape) == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array
                img_cv = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            detection_zones = []
            
            # CONSERVATIVE APPROACH: Only target specific areas where builder logos appear
            
            # 1. HEADER ZONE ONLY (top 10% - where builder logos typically appear)
            header_height = int(height * 0.10)
            header_region = gray[0:header_height, :]
            
            # Only check header for horizontal text banners (company names)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))  # Very horizontal
            header_morph = cv2.morphologyEx(header_region, cv2.MORPH_CLOSE, kernel)
            header_contours, _ = cv2.findContours(header_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in header_contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Only very horizontal banners (likely company names/logos)
                if aspect_ratio > 5 and h > 10 and h < 40 and w > 50:
                    detection_zones.append({
                        "type": "logo_banner",
                        "description": f"Header company banner: {w}x{h}",
                        "bbox": [x, y, w, h],
                        "confidence": 0.8
                    })
            
            # 2. FOOTER ZONE ONLY (bottom 8% - where contact info appears)
            footer_height = int(height * 0.08)
            footer_region = gray[height-footer_height:height, :]
            
            # Only check footer for horizontal text (contact info)
            footer_morph = cv2.morphologyEx(footer_region, cv2.MORPH_CLOSE, kernel)
            footer_contours, _ = cv2.findContours(footer_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in footer_contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Only horizontal contact info lines
                if aspect_ratio > 4 and h > 8 and h < 30 and w > 40:
                    detection_zones.append({
                        "type": "contact",
                        "description": f"Footer contact info: {w}x{h}",
                        "bbox": [x, y + height - footer_height, w, h],
                        "confidence": 0.7
                    })
            
            # Enhanced color-based logo detection (Multiple blue shades)
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            # Multiple blue ranges for different logo variations
            blue_ranges = [
                ([95, 60, 60], [140, 255, 255]),  # Standard blue
                ([100, 50, 50], [130, 255, 255]),  # Darker blue
                ([90, 40, 40], [150, 255, 255]),   # Lighter blue
                ([110, 80, 80], [125, 255, 255])   # Bright blue
            ]
            
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in blue_ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Restrict to header, footer, and corner areas where logos appear
            top_mask = combined_mask[0:int(height*0.3), :]
            bottom_mask = combined_mask[int(height*0.7):, :]
            
            # Process top area - ONLY VERY SMALL LOGOS
            contours, _ = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                area = w*h
                # EXTREMELY CONSERVATIVE: Only redact very small elements (likely logos)
                if area < 500 or area > 20000:  # Only small logos, not home images
                    continue
                aspect = w / h if h>0 else 0
                # Only small, logo-like elements
                if 0.3 < aspect < 4.0 and h < 80 and w < 150:
                    detection_zones.append({
                        "type": "logo_color",
                        "description": f"Small blue logo region (top): {w}x{h}",
                        "bbox": [x, y, w, h],
                        "confidence": 0.9
                    })
            
            # Process bottom area - ONLY VERY SMALL LOGOS
            contours, _ = cv2.findContours(bottom_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                area = w*h
                # EXTREMELY CONSERVATIVE: Only redact very small elements (likely logos)
                if area < 500 or area > 20000:  # Only small logos, not home images
                    continue
                aspect = w / h if h>0 else 0
                # Only small, logo-like elements
                if 0.3 < aspect < 4.0 and h < 80 and w < 150:
                    detection_zones.append({
                        "type": "logo_color",
                        "description": f"Small blue logo region (bottom): {w}x{h}",
                        "bbox": [x, y + int(height*0.7), w, h],
                        "confidence": 0.9
                    })
            
            # Enhanced edge detection for logo shapes
            edges = cv2.Canny(gray, 30, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in header area only
            header_edges = edges[0:int(height*0.2), :]
            contours, _ = cv2.findContours(header_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                area = w*h
                # EXTREMELY CONSERVATIVE: Only very small logo-shaped elements
                if area < 1000 or area > 15000:  # Only small logos, not home images
                    continue
                aspect = w / h if h>0 else 0
                # Only small, logo-like elements in header area
                if 0.4 < aspect < 3.0 and w < 120 and h < 80:
                    detection_zones.append({
                        "type": "logo_shape",
                        "description": f"Small logo-shaped region: {w}x{h}",
                        "bbox": [x, y, w, h],
                        "confidence": 0.7
                    })

            # 3. CORNER ZONES ONLY (very small corners - just for tiny logos)
            corner_size = min(width, height) // 12  # Even smaller corners
            corner_zones = [
                {"desc": "Top-left logo corner", "bbox": [0, 0, corner_size, corner_size]},
                {"desc": "Top-right logo corner", "bbox": [width-corner_size, 0, corner_size, corner_size]}
            ]
            
            for corner in corner_zones:
                x, y, w, h = corner["bbox"]
                corner_region = gray[y:y+h, x:x+w]
                
                # Only detect if there's significant logo-like content
                corner_edges = cv2.Canny(corner_region, 50, 150)
                edge_density = np.sum(corner_edges > 0) / (w * h)
                
                if edge_density > 0.08:  # Even higher threshold - only very clear small logos
                    detection_zones.append({
                        "type": "logo",
                        "description": f"Small corner logo: {corner['desc']}",
                        "bbox": corner["bbox"],
                        "confidence": 0.6
                    })
            
            logger.info(f"Computer Vision: Detected {len(detection_zones)} potential logo/branding zones")
            return {
                "redaction_zones": detection_zones,
                "analysis_summary": f"Computer vision detected {len(detection_zones)} logo/branding zones using edge detection and morphological analysis"
            }
            
        except Exception as e:
            logger.error(f"Computer vision analysis failed: {e}")
            return self._enhanced_computer_vision_analysis(page_image_bytes)
    
    def _fallback_vision_analysis(self, page_image_bytes: bytes) -> Dict[str, Any]:
        """
        Enhanced fallback vision analysis using image processing techniques.
        
        Args:
            page_image_bytes: PNG image bytes of the PDF page
            
        Returns:
            Dictionary containing redaction zones and analysis
        """
        try:
            from PIL import Image
            import io
            
            # Load the image for analysis
            image = Image.open(io.BytesIO(page_image_bytes))
            width, height = image.size
            
            # PRECISION logo detection zones - targeting ONLY small logo areas, not content areas
            enhanced_zones = []
            
            # Small header strip (top 8% only) - just for logos, not content
            header_height = int(height * 0.08)
            enhanced_zones.append({
                "type": "logo",
                "description": "Narrow header strip - logo area only",
                "bbox": [0, 0, width, header_height],
                "confidence": 0.90
            })
            
            # Small footer strip (bottom 8% only) - just for contact/logos
            footer_y = int(height * 0.92)
            enhanced_zones.append({
                "type": "branding",
                "description": "Narrow footer strip - logo area only",
                "bbox": [0, footer_y, width, height - footer_y],
                "confidence": 0.85
            })
            
            # Small corner areas ONLY (much smaller than before)
            corner_size = min(width, height) // 8  # Very small corners
            enhanced_zones.extend([
                {
                    "type": "logo",
                    "description": "Small top-left corner - logo only",
                    "bbox": [0, 0, int(corner_size), int(corner_size)],
                    "confidence": 0.95
                },
                {
                    "type": "logo", 
                    "description": "Small top-right corner - logo only",
                    "bbox": [width - int(corner_size), 0, int(corner_size), int(corner_size)],
                    "confidence": 0.95
                }
            ])
            
            # DISABLED: Remove all the overly aggressive zones that were redacting content
            
            logger.info("Enhanced Vision Analysis: Applied comprehensive logo detection zones")
            return {
                "redaction_zones": enhanced_zones,
                "analysis_summary": "Enhanced vision-based detection applied to common logo placement areas"
            }
                
        except Exception as e:
            logger.error(f"Error in fallback vision analysis: {str(e)}")
            return {"redaction_zones": [], "analysis_summary": f"Error: {str(e)}"}
    
    def extract_page_as_image(self, page: fitz.Page, dpi: int = 150) -> bytes:
        """
        Extract a PDF page as a high-quality PNG image for AI analysis.
        
        Args:
            page: PyMuPDF page object
            dpi: Resolution for image extraction
            
        Returns:
            PNG image as bytes
        """
        # Create a transformation matrix for the desired DPI
        mat = fitz.Matrix(dpi/72, dpi/72)
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PNG bytes
        img_data = pix.tobytes("png")
        pix = None  # Free memory
        
        return img_data
    
    def apply_ai_redactions(self, page: fitz.Page, redaction_zones: List[Dict], scale_factor: float = 1.0) -> int:
        """
        Apply redactions based on AI-detected zones.
        
        Args:
            page: PyMuPDF page object
            redaction_zones: List of redaction zones from AI analysis
            scale_factor: Scale factor to adjust coordinates from analysis image to page
            
        Returns:
            Number of redactions applied
        """
        redactions_applied = 0
        
        for zone in redaction_zones:
            try:
                bbox = zone.get('bbox', [])
                if len(bbox) != 4:
                    logger.warning(f"Invalid bbox format: {bbox}")
                    continue
                
                # Scale coordinates if needed
                x, y, w, h = bbox
                x *= scale_factor
                y *= scale_factor
                w *= scale_factor
                h *= scale_factor
                
                # Create redaction rectangle
                rect = fitz.Rect(x, y, x + w, y + h)
                
                # Apply redaction using annotation (white fill)
                redact_annot = page.add_redact_annot(rect)
                redact_annot.set_colors(fill=(1, 1, 1))
                redact_annot.update()
                redactions_applied += 1
                
                logger.info(f"Applied AI redaction: {zone.get('type', 'unknown')} - {zone.get('description', 'no description')}")
                
            except Exception as e:
                logger.error(f"Error applying redaction zone: {e}")
                continue
        
        return redactions_applied
    
    def _load_logo_templates(self) -> List[Dict[str, Any]]:
        """
        Load all logo templates from the logos folder for template matching.
        
        Returns:
            List of dictionaries containing logo templates and metadata
        """
        logo_templates = []
        
        if not os.path.exists(LOGOS_FOLDER):
            logger.warning(f"Logos folder not found: {LOGOS_FOLDER}")
            return logo_templates
        
        # Supported image formats
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.webp', '*.gif']
        
        for extension in image_extensions:
            pattern = os.path.join(LOGOS_FOLDER, extension)
            for logo_path in glob.glob(pattern):
                try:
                    # Load image
                    if logo_path.lower().endswith('.svg'):
                        # Convert SVG to PNG for template matching
                        img = self._convert_svg_to_png(logo_path)
                    else:
                        img = cv2.imread(logo_path, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        continue
                    
                    # Resize if too large
                    height, width = img.shape[:2]
                    if width > LOGO_MAX_SIZE[0] or height > LOGO_MAX_SIZE[1]:
                        scale = min(LOGO_MAX_SIZE[0] / width, LOGO_MAX_SIZE[1] / height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        img = cv2.resize(img, (new_width, new_height))
                    
                    # Skip if too small
                    if width < LOGO_MIN_SIZE[0] or height < LOGO_MIN_SIZE[1]:
                        continue
                    
                    # Extract filename without extension for identification
                    logo_name = os.path.splitext(os.path.basename(logo_path))[0]
                    
                    logo_templates.append({
                        'name': logo_name,
                        'path': logo_path,
                        'template': img,
                        'width': img.shape[1],
                        'height': img.shape[0]
                    })
                    
                    logger.info(f"Loaded logo template: {logo_name} ({img.shape[1]}x{img.shape[0]})")
                    
                except Exception as e:
                    logger.error(f"Error loading logo {logo_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(logo_templates)} logo templates for matching")
        return logo_templates
    
    def _convert_svg_to_png(self, svg_path: str) -> Optional[np.ndarray]:
        """
        Convert SVG to PNG format for template matching.
        
        Args:
            svg_path: Path to SVG file
            
        Returns:
            OpenCV image array or None if conversion fails
        """
        try:
            from cairosvg import svg2png
            import io
            
            # Convert SVG to PNG bytes
            png_bytes = svg2png(url=svg_path)
            
            # Convert to OpenCV format
            img_array = np.frombuffer(png_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            return img
            
        except ImportError:
            logger.warning("cairosvg not available, skipping SVG conversion")
            return None
        except Exception as e:
            logger.error(f"Error converting SVG {svg_path}: {e}")
            return None
    
    def detect_logos_in_page(self, page_image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Detect logos in a PDF page using template matching.
        
        Args:
            page_image_bytes: PNG image bytes of the PDF page
            
        Returns:
            List of detected logo regions with bounding boxes
        """
        if not self.logo_templates:
            return []
        
        try:
            # Convert page image to OpenCV format
            img_array = np.frombuffer(page_image_bytes, np.uint8)
            page_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if page_img is None:
                return []
            
            detected_logos = []
            
            for logo_template in self.logo_templates:
                template = logo_template['template']
                template_name = logo_template['name']
                
                # Perform template matching
                result = cv2.matchTemplate(page_img, template, cv2.TM_CCOEFF_NORMED)
                
                # Find locations where template matches
                locations = np.where(result >= LOGO_MATCH_THRESHOLD)
                
                for pt in zip(*locations[::-1]):  # Switch x and y coordinates
                    x, y = pt
                    w, h = template.shape[1], template.shape[0]
                    
                    # Check if this detection overlaps with existing ones
                    new_rect = (x, y, x + w, y + h)
                    overlaps = False
                    
                    for existing in detected_logos:
                        existing_rect = existing['bbox']
                        if self._rectangles_overlap(new_rect, existing_rect):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        detected_logos.append({
                            'type': 'logo_template',
                            'description': f"Detected logo: {template_name}",
                            'bbox': [x, y, w, h],
                            'confidence': float(result[y, x]),
                            'logo_name': template_name
                        })
            
            # Sort by confidence (highest first)
            detected_logos.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"Detected {len(detected_logos)} logos using template matching")
            return detected_logos
            
        except Exception as e:
            logger.error(f"Error in logo detection: {e}")
            return []
    
    def _rectangles_overlap(self, rect1: Tuple[int, int, int, int], rect2: List[int]) -> bool:
        """
        Check if two rectangles overlap.
        
        Args:
            rect1: (x1, y1, x2, y2) format
            rect2: [x, y, w, h] format
            
        Returns:
            True if rectangles overlap
        """
        x1, y1, x2, y2 = rect1
        x3, y3, w3, h3 = rect2
        x4, y4 = x3 + w3, y3 + h3
        
        return not (x2 <= x3 or x4 <= x1 or y2 <= y3 or y4 <= y1)
    
    def apply_logo_redactions(self, page: fitz.Page, logo_detections: List[Dict], scale_factor: float = 1.0) -> int:
        """
        Apply redactions for detected logos.
        
        Args:
            page: PyMuPDF page object
            logo_detections: List of logo detections from template matching
            scale_factor: Scale factor to adjust coordinates from analysis image to page
            
        Returns:
            Number of logo redactions applied
        """
        redactions_applied = 0
        
        for detection in logo_detections:
            try:
                bbox = detection.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                # Scale coordinates if needed
                x, y, w, h = bbox
                x *= scale_factor
                y *= scale_factor
                w *= scale_factor
                h *= scale_factor
                
                # Create redaction rectangle
                rect = fitz.Rect(x, y, x + w, y + h)
                
                # Apply redaction using annotation (white fill)
                redact_annot = page.add_redact_annot(rect)
                redact_annot.set_colors(fill=(1, 1, 1))
                redact_annot.update()
                redactions_applied += 1
                
                logo_name = detection.get('logo_name', 'unknown')
                confidence = detection.get('confidence', 0.0)
                logger.info(f"Applied logo redaction: {logo_name} (confidence: {confidence:.2f})")
                
            except Exception as e:
                logger.error(f"Error applying logo redaction: {e}")
                continue
        
        return redactions_applied
    
    def detect_logos_with_yolo(self, page_image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Detect logos using the trained YOLO model - PRIMARY detection method.
        
        Args:
            page_image_bytes: PNG image bytes of the PDF page
            
        Returns:
            List of detected logo regions with bounding boxes
        """
        if not self.yolo_detector:
            logger.warning("YOLO detector not available - this is the PRIMARY detection method")
            return []
        
        try:
            # Use the trained YOLO model with more permissive settings
            detections = self.yolo_detector.detect_logos_in_page(
                page_image_bytes,
                min_size=(15, 15),  # Smaller minimum to catch tiny logos
                max_size=(300, 300)  # Larger maximum to catch bigger logos
            )
            
            # Log detailed detection information
            class_counts = {}
            for detection in detections:
                class_name = detection.get('class', 'unknown')
                confidence = detection.get('confidence', 0.0)
                bbox = detection.get('bbox', [])
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                logger.info(f"YOLO detected {class_name} at {bbox} (confidence: {confidence:.3f})")
            
            class_summary = ", ".join([f"{count} {cls}" for cls, count in class_counts.items()])
            logger.info(f"YOLO detected {len(detections)} total elements: {class_summary}")
            return detections
            
        except Exception as e:
            logger.error(f"YOLO logo detection failed: {e}")
            return []
    
    def apply_yolo_redactions(self, page: fitz.Page, yolo_detections: List[Dict], scale_factor: float = 1.0) -> int:
        """
        Apply redactions for YOLO-detected logos.
        
        Args:
            page: PyMuPDF page object
            yolo_detections: List of YOLO logo detections
            scale_factor: Scale factor to adjust coordinates from analysis image to page
            
        Returns:
            Number of YOLO redactions applied
        """
        redactions_applied = 0
        
        for detection in yolo_detections:
            try:
                bbox = detection.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                # Scale coordinates if needed
                x, y, w, h = bbox
                x *= scale_factor
                y *= scale_factor
                w *= scale_factor
                h *= scale_factor
                
                # Create redaction rectangle
                rect = fitz.Rect(x, y, x + w, y + h)
                
                # Apply redaction using annotation (white fill)
                redact_annot = page.add_redact_annot(rect)
                redact_annot.set_colors(fill=(1, 1, 1))
                redact_annot.update()
                redactions_applied += 1
                
                confidence = detection.get('confidence', 0.0)
                logger.info(f"Applied YOLO redaction: confidence {confidence:.2f}")
                
            except Exception as e:
                logger.error(f"Error applying YOLO redaction: {e}")
                continue
        
        return redactions_applied

# Existing pattern-based redaction patterns (from original system)
REDACTION_PATTERNS = {
    "phone_numbers": [
        r"\bT\.\s*\d{3}\.\d{3}\.\d{4}\b",
        r"\bTel(?:ephone)?[:.]*\s*(?:\+?\d{1,3}[\s.-]*)?(?:\(?\d{3}\)?[\s.-]*)?\d{3}[\s.-]*\d{4}\b",
        r"(?:\+?\d{1,3}[\s.-]*)?(?:\(?\d{3}\)?[\s.-]*)?\d{3}[\s.-]*\d{4}(?:\s*(?:ext|x|extension)\.?\s*\d+)?\b",
        r"(?:Cell|Mobile|Office|Fax|Direct|Phone|Tel|Sales)[:.]\s*(?:\+?\d{1,3}[\s.-]*)?\d{3}[\s.-]*\d{3}[\s.-]*\d{4}",
        r"\b\d{3}[\s.-]\d{3}[\s.-]\d{4}\b",  # Basic 123-456-7890 format
        r"\b\(\d{3}\)\s*\d{3}[\s.-]\d{4}\b",  # (123) 456-7890 format
        r"\b1[\s.-]?\d{3}[\s.-]\d{3}[\s.-]\d{4}\b",  # North American format with 1
        r"(?:Call|Contact|Phone)[:\s]+(?:\+?\d{1,3}[\s.-]*)?\d{3}[\s.-]*\d{3}[\s.-]*\d{4}",
        r"\d{3}\.\d{3}\.\d{4}",  # 123.456.7890
        r"\+1[\s.-]?\d{3}[\s.-]?\d{3}[\s.-]?\d{4}",  # International format
    ],
    "email_addresses": [
        r"\bE\.\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        r"\bEmail[:.]*\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        r"(?:Contact|Info|Support|Sales)[:.]\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"
    ],
    "websites": [
        r"\bW\.\s*(?:www\.)?[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        r"\bWebsite[:.]*\s*(?:https?://)?(?:www\.)?[A-Za-z0-9\-]+\.[A-Za-z0-9\-]+\b",
        r"(?:https?://)?(?:www\.)?[A-Za-z0-9\-]+\.[A-Za-z0-9\-]+(?:/[^\s]*)?\b",
        r"(?:facebook|twitter|instagram|linkedin)\.com/[A-Za-z0-9._%+-]+"
    ],
    "broker_info": [
        r"EXCLUSIVE\s+LISTING\s+BROKERAGE[^.]*?BROKERAGE",
        r"Brokers?\s+Protected",
        r"(?:Listing|Selling|Exclusive)\s+(?:Agent|Broker|Representative|REALTOR)[^.]*?\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        r"(?:Sales|Leasing)\s+(?:Team|Representative|Agent|Associate)[^.]*?\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        r"(?:Mr\.|Mrs\.|Ms\.|Dr\.|Sales\s+Rep|Agent)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(?:Sales|Rep|Agent|Associate)",
        r"(?:Contact|Call|Speak\s+with|Meet\s+with)[^.]*?\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        r"(?:Your\s+Sales|Your\s+Rep|Your\s+Agent)[^.]*?\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        r"(?:Sales\s+Team|Sales\s+Staff)[^.]*?\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        # Builder/developer specific sales roles
        r"(?:Builder\s+Representative|Developer\s+Rep|Project\s+Rep)[^.]*?\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        r"(?:Site\s+Sales|Model\s+Home\s+Sales|Presentation\s+Center\s+Sales)[^.]*?\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        # Contact information with names
        r"(?:Direct|Cell|Mobile|Office)[^.]*?\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        r"(?:Extension|Ext)[^.]*?\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
    ],
    "brand_names": [
        # Fieldgate variations
        r"\bFieldgate(?:\s+Homes)?\b",
        r"\bFieldgatehomes\b", 
        r"FIELDGATE\s+HOMES",
        r"FIELDGATEHOMES\.COM",  # Specific for the website
        r"fieldgatehomes\.com",  # Case variations
        r"FIELDGATEHOMES",  # Without .com
        r"fieldgatehomes",  # Lowercase
        r"OVER\s+\d+\s+YEARS\s+OF\s+EXCELLENCE",
        r"\bFieldgate\b",
        r"(?i)fieldgate",  # Case insensitive
        r"(?i)field\s*gate",  # With potential spacing
        r"EST\.\s*1957",  # Fieldgate establishment year
        
        # Major Canadian builders
        r"\bTridel\b",
        r"\bBuzzBuzzHome\b",
        r"\bDaniels\b",
        r"\bConcord\s+Pacific\b",
        r"\bGreenpark\s+Group\b",
        r"\bTimes\s+Group\b",
        r"\bMattamy\s+Homes\b",
        r"\bGreat\s+Gulf\b",
        r"\bPinnacle\s+International\b",
        r"\bMenkes\b",
        r"\bCresford\b",
        r"\bLanterra\b",
        r"\bPlazacorp\b",
        r"\bRioCan\b",
        r"\bBrookfield\b",
        
        # Generic builder patterns
        r"\b[A-Z][a-z]+\s+(?:Homes|Housing|Developments|Properties|Group|Corp|Corporation)\b",
        r"\b[A-Z][a-z]+\s+(?:Real\s+Estate|Construction|Building)\b",
    ],
    
    # Sales office and presentation center (COMPREHENSIVE)
    "sales_office": [
        r"Sales\s+(?:Office|Centre|Center|Gallery|Pavilion)[^.]*",
        r"Presentation\s+(?:Centre|Center|Office)[^.]*",
        r"Marketing\s+Suite[^.]*",
        r"Model\s+(?:Home|Suite|Unit)[^.]*",
        r"Information\s+(?:Centre|Center)[^.]*",
        r"Welcome\s+(?:Centre|Center)[^.]*",
        r"Discovery\s+(?:Centre|Center)[^.]*",
        r"(?:Visit|Contact|Call)\s+(?:our|the)\s+(?:sales|presentation|marketing)\s+(?:centre|center|office|gallery)[^.]*",
        r"(?:Hours|Open)[^.]*?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)",
        r"(?:Sales|Presentation|Marketing)\s+(?:Centre|Center|Office|Gallery)[^.]*?\d{3}[\s.-]*\d{3}[\s.-]*\d{4}",
    ],
    
    # Builder addresses and locations
    "builder_addresses": [
        r"\d+\s+[A-Z][a-zA-Z\s,.'-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Place|Pl|Court|Ct)\b[^.]*(?:Suite|Unit|Floor)\s*\d+",
        r"(?:Located|Address|Visit us)[^.]*\d+\s+[A-Z][a-zA-Z\s,.'-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)",
        r"(?:Head\s+Office|Corporate\s+Office|Main\s+Office)[^.]*\d+\s+[A-Z][a-zA-Z\s,.'-]+",
        r"\d+\s+[A-Z][a-zA-Z\s,.'-]+,\s*[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\s+[A-Z0-9]{3}\s*[A-Z0-9]{3}",  # Canadian postal code
        r"Suite\s+\d+[^.]*(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)",
        r"Unit\s+\d+[^.]*(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)",
        r"Floor\s+\d+[^.]*(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)",
        r"\d+\s+[A-Z][a-zA-Z\s,.'-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way)\b",  # Basic street addresses
    ],
    
    # Contact information headers and sections
    "contact_sections": [
        r"(?:Contact|Reach|Call|Email|Visit)\s+(?:Us|Information|Details)[^.]*",
        r"For\s+(?:More\s+)?Information[^.]*",
        r"To\s+(?:Learn\s+More|Register|Book|Schedule)[^.]*",
        r"Get\s+in\s+Touch[^.]*",
        r"Connect\s+with\s+Us[^.]*",
        r"Questions\?[^.]*",
        r"Need\s+Help\?[^.]*",
        r"Contact\s+(?:Information|Details|Us)",
        r"Reach\s+Out[^.]*",
        r"Speak\s+with[^.]*",
    ]
}

# Protected terms that should never be redacted
PROTECTED_TERMS = [
    r"\$\s*[\d,.]+(?:\s*[KMB])?(?:\s*(?:CAD|USD))?",
    r"\b(?:from|starting|prices?\s+from)\s+\$[\d,.]+(?:\s*[KMB])?\b",
    r"\b\d+(?:\.\d+)?%\s*(?:deposit|down\s+payment)\b",
    r"\b(?:HST|GST|tax)\s+(?:included|excluded|applicable)\b",
    r"\d+(?:\.\d+)?\s*(?:sq\.?\s*ft\.?|square\s+feet|m²)",
    r"\b\d+(?:\.\d+)?\s*(?:BR|BDRM|bedroom|bed)s?\b",
    r"\b\d+(?:\.\d+)?\s*(?:bath(?:room)?|WC)s?\b",
    r"\b(?:maintenance|condo)\s+fees?\s*[:=]\s*\$[\d,.]+(?:/[\w\s]+)?\b",
    r"\b(?:floor|level|storey)s?\s+\d+(?:\s*-\s*\d+)?\b",
    r"\b\d+\s*(?:floor|level|storey)s?\b",
    r"\b(?:parking|locker|storage)\s+(?:included|available|optional)\b",
    r"\b(?:occupancy|closing|completion)\s+(?:date|expected)\s*[:=]\s*[^\n.]*\d{4}\b",
    r"\b(?:incentives?|promotions?|special\s+offers?)\s*[:=][^\n.]*\b",
    r"\b(?:Assignment|Resale|Pre-construction|New\s+Construction)\b",
    # Additional property-related terms to protect
    r"\b(?:unit|suite|apartment|condo|townhouse|house)\s*(?:type|size|area)\b",
    r"\b(?:amenities|features|inclusions|upgrades)\b",
    r"\b(?:balcony|terrace|patio|deck)\b",
    r"\b(?:view|facing|orientation)\b",
    r"\b(?:kitchen|living|dining|bedroom|bathroom|laundry)\b",
    r"\b(?:closet|storage|pantry|den|office)\b",
    r"\b(?:hardwood|tile|carpet|granite|marble|stainless)\b",
    r"\b(?:appliances|fixtures|finishes|materials)\b",
    r"\b(?:building|development|project|community)\b",
    r"\b(?:location|address|neighborhood|area)\b"
]

def should_protect_text(text: str) -> bool:
    """Check if text contains any protected terms that should not be redacted."""
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in PROTECTED_TERMS)

# Pre-compile regex patterns for better performance
COMPILED_PATTERNS = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in REDACTION_PATTERNS.items()
}

def _get_branding_config(project_folder_path: str) -> Tuple[str, str]:
    """
    Determine watermark text and footer URL based on project folder.
    
    Args:
        project_folder_path: Path to the project folder (format: "BrandingName/ProjectName")
        
    Returns:
        Tuple of (watermark_text, footer_image_url)
    """
    # Convert to lowercase for case-insensitive comparison
    folder_path_lower = project_folder_path.lower()
    
    # Check if this is a GTA Lowrise project
    if 'gta' in folder_path_lower and 'lowrise' in folder_path_lower:
        logger.info("Using GTA Lowrise branding for project folder")
        return GTA_LOWRISE_WATERMARK_TEXT, GTA_LOWRISE_FOOTER_IMAGE_URL
    
    # Check if this is a Fahad Javed project
    if 'fahad' in folder_path_lower and 'javed' in folder_path_lower:
        logger.info("Using Fahad Javed branding for project folder")
        return FAHAD_WATERMARK_TEXT, FAHAD_FOOTER_IMAGE_URL
    
    # Default to Precon Factory branding
    logger.info("Using Precon Factory branding for project folder")
    return DEFAULT_WATERMARK_TEXT, DEFAULT_FOOTER_IMAGE_URL

def _download_image_bytes(url: str) -> Optional[bytes]:
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None

def _make_watermark_png(text: str, page_width: int, page_height: int, opacity: float, angle_deg: float) -> bytes:
    canvas_w, canvas_h = int(page_width), int(page_height)
    img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    base = min(canvas_w, canvas_h)
    font_size = max(36, int(base * 0.15))  # Moderate size - not too large, not too small
    
    # Try multiple font paths for better compatibility across systems
    font_paths = [
        'Arial.ttf',
        'Helvetica.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',  # Linux
        '/System/Library/Fonts/Helvetica.ttc',  # macOS
        'C:\\Windows\\Fonts\\arial.ttf',  # Windows
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',  # Linux alternative
    ]
    
    font = None
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
    except Exception:
            continue
    
    # If no TrueType font found, use default and scale it up
    using_default_font = False
    if font is None:
            font = ImageFont.load_default()
        using_default_font = True
        
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    # If using default font, scale up the dimensions moderately
    if using_default_font:
        scale_factor = 8  # Moderate scaling - 50% smaller than before (was 15x)
        text_w = text_w * scale_factor
        text_h = text_h * scale_factor
    
    # Increase padding to account for rotation
    padding = 40
    text_img = Image.new('RGBA', (text_w + padding, text_h + padding), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_img)
    alpha = int(255 * max(0.0, min(1.0, opacity)))
    
    if using_default_font:
        # Draw text small, then scale up
        small_img = Image.new('RGBA', (text_bbox[2] - text_bbox[0] + 10, text_bbox[3] - text_bbox[1] + 10), (0, 0, 0, 0))
        small_draw = ImageDraw.Draw(small_img)
        small_draw.text((5, 5), text, font=font, fill=(0, 0, 0, alpha))
        # Scale up with high-quality resampling
        text_img = small_img.resize((text_w + padding, text_h + padding), Image.Resampling.LANCZOS)
    else:
    text_draw.text((padding // 2, padding // 2), text, font=font, fill=(0, 0, 0, alpha))
    
    rotated = text_img.rotate(angle_deg, expand=True)
    rx, ry = rotated.size
    
    # Ensure the watermark fits within the page bounds
    # Add extra margin to prevent cutoff
    margin = 50
    px = max(margin, (canvas_w - rx) // 2)
    py = max(margin, (canvas_h - ry) // 2)
    
    # Make sure watermark doesn't extend beyond page bounds
    if px + rx > canvas_w - margin:
        px = canvas_w - rx - margin
    if py + ry > canvas_h - margin:
        py = canvas_h - ry - margin
    
    # Ensure coordinates are not negative
    px = max(0, px)
    py = max(0, py)
    
    img.alpha_composite(rotated, (px, py))
    out = io.BytesIO()
    img.save(out, format='PNG')
    return out.getvalue()

def _insert_footer_image(page: fitz.Page, footer_png: Optional[bytes]) -> None:
    if not footer_png:
        return
    try:
        page_w = page.rect.width
        page_h = page.rect.height
        with Image.open(io.BytesIO(footer_png)) as im:
            img_w, img_h = im.size
        target_w = page_w
        scale = target_w / float(img_w)
        target_h = img_h * scale
        x0 = 0.0
        y1 = page_h
        y0 = y1 - target_h
        rect = fitz.Rect(x0, y0, x0 + target_w, y1)
        page.insert_image(rect, stream=footer_png, keep_proportion=True, overlay=True)
    except Exception as e:
        logger.warning(f"Failed to insert footer image: {e}")

def ai_enhanced_redact_pdf(input_path: str, output_path: str, project_folder_path: str = "") -> Tuple[int, Dict[int, int]]:
    """
    Enhanced PDF redaction using both AI vision analysis and traditional pattern matching.
    
    Args:
        input_path: Path to input PDF file
        output_path: Path where redacted PDF will be saved
        project_folder_path: Path to project folder to determine branding
        
    Returns:
        Tuple containing:
            - Total number of redactions applied
            - Dictionary mapping page numbers to number of redactions on that page
    """
    # Determine branding based on project folder
    watermark_text, footer_image_url = _get_branding_config(project_folder_path)
    footer_png = _download_image_bytes(footer_image_url)
    
    doc = fitz.open(input_path)
    total_redactions = 0
    redactions_by_page = {}
    ai_redactor = AIEnhancedRedactor()
    
    # NO REMOVAL - Only watermark and footer will be ADDED (no content removed)
    logger.info("✨ PDF PROCESSING MODE: Adding watermark and footer only (NO content removal)")
    
    try:
        for page_num, page in enumerate(doc):
            page_redactions = 0
            page_height = page.rect.height
            page_width = page.rect.width
            
            logger.info(f"Processing page {page_num + 1} with AI-enhanced redaction...")
            
            # Step 1: Extract page as image for analysis
            try:
                page_image_bytes = ai_redactor.extract_page_as_image(page, dpi=150)
                
                # Calculate scale factor (image DPI vs page points)
                image_dpi = 150
                points_per_inch = 72
                scale_factor = points_per_inch / image_dpi
                
            except Exception as e:
                logger.error(f"Failed to extract page image for page {page_num + 1}: {e}")
                continue
            
            # PAUSED: Step 2: YOLO logo detection (PRIMARY METHOD - most accurate)
            # REDACTION PAUSED - Only headers and footers will be processed
            logger.info(f"⏸️  YOLO logo detection paused for page {page_num + 1}")
            # try:
            #     yolo_detections = ai_redactor.detect_logos_with_yolo(page_image_bytes)
            #     if yolo_detections:
            #         yolo_redactions = ai_redactor.apply_yolo_redactions(
            #             page, yolo_detections, scale_factor
            #         )
            #         page_redactions += yolo_redactions
            #         logger.info(f"✅ Applied {yolo_redactions} YOLO logo redactions on page {page_num + 1}")
            #     else:
            #         logger.info(f"No YOLO detections on page {page_num + 1}")
            # except Exception as e:
            #     logger.error(f"YOLO logo detection failed for page {page_num + 1}: {e}")
            
            # PAUSED: Step 3: AI Vision Analysis (SECONDARY METHOD - for complex cases)
            # REDACTION PAUSED - Only headers and footers will be processed
            logger.info(f"⏸️  AI vision analysis paused for page {page_num + 1}")
            # try:
            #     ai_analysis = ai_redactor.analyze_page_with_ai(page_image_bytes)
            #     
            #     if ai_analysis.get('redaction_zones'):
            #         # Apply AI-detected redactions with better filtering
            #         ai_zones = ai_analysis['redaction_zones']
            #         
            #         # Filter for relevant types and reasonable sizes
            #         filtered_zones = []
            #         for zone in ai_zones:
            #             zone_type = zone.get('type', '')
            #             bbox = zone.get('bbox', [])
            #             
            #             # Accept all relevant types
            #             if zone_type in ['logo', 'branding', 'contact', 'builder_info', 'header', 'footer']:
            #                 if len(bbox) == 4:
            #                     w, h = bbox[2], bbox[3]
            #                     # Allow reasonable sizes (not too small, not too large)
            #                     if 10 <= w <= 400 and 10 <= h <= 400:
            #                         filtered_zones.append(zone)
            #         
            #         if filtered_zones:
            #             ai_redactions = ai_redactor.apply_ai_redactions(
            #                 page, filtered_zones, scale_factor
            #             )
            #             page_redactions += ai_redactions
            #             logger.info(f"✅ Applied {ai_redactions} AI vision redactions on page {page_num + 1}")
            #         else:
            #             logger.info(f"No suitable AI zones found on page {page_num + 1}")
            #     
            # except Exception as e:
            #     logger.error(f"AI vision analysis failed for page {page_num + 1}: {e}")
            
            # PAUSED: Step 4: Template matching logo detection (FALLBACK METHOD)
            # REDACTION PAUSED - Only headers and footers will be processed
            logger.info(f"⏸️  Template matching logo detection paused for page {page_num + 1}")
            # try:
            #     logo_detections = ai_redactor.detect_logos_in_page(page_image_bytes)
            #     if logo_detections:
            #         logo_redactions = ai_redactor.apply_logo_redactions(
            #             page, logo_detections, scale_factor
            #         )
            #         page_redactions += logo_redactions
            #         logger.info(f"✅ Applied {logo_redactions} template matching redactions on page {page_num + 1}")
            # except Exception as e:
            #     logger.error(f"Template matching logo detection failed for page {page_num + 1}: {e}")
            
            # Step 6: Traditional pattern-based redaction for text (DISABLED - NO REMOVAL)
            # PAUSED: Header and footer removal - We only ADD watermark/footer, not remove anything
            # header_region = fitz.Rect(0, 0, page_width, page_height * 0.03)  # Very small header strip
            # footer_region = fitz.Rect(0, page_height * 0.97, page_width, page_height)  # Very small footer strip
            # 
            # blocks = page.get_text("dict")["blocks"]
            # 
            # for block in blocks:
            #     if "lines" not in block:
            #         continue
            #     
            #     block_rect = fitz.Rect(block["bbox"])
            #     
            #     # Always redact headers and footers with proper annotations (DISABLED)
            #     if header_region.intersects(block_rect) or footer_region.intersects(block_rect):
            #         redact_annot = page.add_redact_annot(block_rect)
            #         redact_annot.set_colors(fill=(1, 1, 1))  # White fill
            #         redact_annot.update()
            #         page_redactions += 1
            #         continue
                
                # PAUSED: Content redaction based on patterns
                # REDACTION PAUSED - Only headers and footers will be processed
                # Combine all text in the block
                # block_text = " ".join(
                #     span["text"].strip()
                #     for line in block["lines"]
                #     for span in line["spans"]
                # )
                # 
                # if not block_text or should_protect_text(block_text):
                #     continue
                # 
                # # Check for sensitive patterns
                # should_redact = False
                # for patterns in COMPILED_PATTERNS.values():
                #     if any(pattern.search(block_text) for pattern in patterns):
                #         should_redact = True
                #         break
                # 
                # if should_redact:
                #     # Use proper redaction annotation instead of just drawing rectangles
                #     redact_annot = page.add_redact_annot(block_rect)
                #     redact_annot.set_colors(fill=(1, 1, 1))  # White fill
                #     redact_annot.update()
                #     page_redactions += 1
            
            # Apply all redactions for this page
            if page_redactions > 0:
                page.apply_redactions()
                total_redactions += page_redactions
                redactions_by_page[page_num + 1] = page_redactions
            
            # Add watermark and footer
            try:
                wm_png = _make_watermark_png(
                    text=watermark_text,
                    page_width=int(page_width),
                    page_height=int(page_height),
                    opacity=WATERMARK_OPACITY,
                    angle_deg=WATERMARK_ANGLE_DEGREES,
                )
                page.insert_image(page.rect, stream=wm_png, overlay=True, keep_proportion=False)
            except Exception as e:
                logger.warning(f"Failed to insert watermark: {e}")
            
            _insert_footer_image(page, footer_png)
        
        # Save the redacted document
        doc.save(output_path)
        
    finally:
        doc.close()
    
    return total_redactions, redactions_by_page

# Include all the existing Google Drive and file processing functions
def _get_google_credentials() -> Credentials:
    creds = None
    if os.path.exists(TOKEN_PICKLE_FILE):
        with open(TOKEN_PICKLE_FILE, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.error(f"Error refreshing token: {e}. Need to re-authenticate.")
                creds = None
        if not creds:
            if not os.path.exists(CLIENT_SECRET_FILE):
                logger.error(f"Client secret file not found: {CLIENT_SECRET_FILE}")
                logger.error("Please download your client secret from Google Cloud Console and place it here.")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=8765)
        with open(TOKEN_PICKLE_FILE, 'wb') as token:
            pickle.dump(creds, token)
    return creds

def get_drive_service():
    """Authenticates with Google Drive API and returns the service object."""
    creds = _get_google_credentials()
    return build('drive', 'v3', credentials=creds)

def get_sheets_service():
    creds = _get_google_credentials()
    return build('sheets', 'v4', credentials=creds)

def process_pdf_enhanced(input_path: str, output_path: str, project_folder_path: str = "") -> None:
    """Process a single PDF file with AI-enhanced redaction."""
    start_time = time.time()
    
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # Create output directory if it doesn't exist (only if there's a directory path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting AI-enhanced processing of {input_path}")
        total_redactions, redactions_by_page = ai_enhanced_redact_pdf(input_path, output_path, project_folder_path)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Successfully processed {input_path} in {processing_time:.2f} seconds")
        logger.info(f"Total redactions applied: {total_redactions}")
        
        if redactions_by_page:
            logger.info("Redactions by page:")
            for page_num, count in sorted(redactions_by_page.items()):
                logger.info(f"  Page {page_num}: {count} redactions")
        else:
            logger.info("No redactions were necessary in this document")
            
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the AI-enhanced redaction system
    test_input = "test_input.pdf"
    test_output = "test_output.pdf"
    
    if os.path.exists(test_input):
        # Test with default Precon Factory branding
        process_pdf_enhanced(test_input, test_output, "")
        logger.info("AI-enhanced redaction test completed!")
    else:
        logger.info("AI-Enhanced PDF Redaction System Ready!")
        logger.info("Place a PDF file named 'test_input.pdf' in this directory to test the system.")
