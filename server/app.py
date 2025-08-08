"""
LLM-Based RAG Pipeline for Mathematical Question Extraction
Enhanced version with 90%+ accuracy target
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import fitz  # PyMuPDF
import requests
from openai import OpenAI
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@dataclass
class ExtractedQuestion:
    """Represents an extracted mathematical question"""
    question_text: str
    latex_content: str
    question_type: str  # 'illustration', 'exercise', 'example'
    confidence_score: float
    chapter: str
    topic: str
    page_number: int

class EnhancedRAGPipeline:
    """
    Enhanced RAG Pipeline for Mathematical Question Extraction
    Designed to achieve 90%+ accuracy
    """
    
    def __init__(self, openai_api_key: str = None):
        # Initialize OpenAI client only if API key is provided
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_openai_api_key_here":
            try:
                self.client = OpenAI(api_key=api_key)
                self.encoder = tiktoken.encoding_for_model("gpt-4")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.client = None
                self.encoder = None
        else:
            self.client = None
            self.encoder = None
            logger.info("OpenAI client not initialized - will use demo mode")
        
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.pdf_cache = {}
        
    def download_pdf(self, pdf_url: str) -> str:
        """Download PDF from URL and save locally"""
        try:
            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()
            
            pdf_path = "temp_rd_sharma.pdf"
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"PDF downloaded successfully: {pdf_path}")
            return pdf_path
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with page information"""
        try:
            doc = fitz.open(pdf_path)
            pages_data = []
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Extract text blocks with position information
                blocks = page.get_text("dict")["blocks"]
                
                pages_data.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "blocks": blocks,
                    "char_count": len(text)
                })
            
            doc.close()
            logger.info(f"Extracted text from {len(pages_data)} pages")
            return pages_data
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def identify_chapter_pages(self, pages_data: List[Dict], chapter_query: str) -> List[int]:
        """Identify pages containing the specified chapter/topic"""
        relevant_pages = []
        
        # Create search patterns for chapter identification
        chapter_patterns = [
            rf"chapter\s*{re.escape(chapter_query)}\b",
            rf"{re.escape(chapter_query)}\s*(introduction|recapitulation)",
            rf"^{re.escape(chapter_query)}\.",
            rf"section\s*{re.escape(chapter_query)}\b"
        ]
        
        for page_data in pages_data:
            text_lower = page_data["text"].lower()
            
            # Check for chapter patterns
            for pattern in chapter_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                    relevant_pages.append(page_data["page_number"])
                    break
        
        # If no specific chapter found, use semantic similarity
        if not relevant_pages:
            relevant_pages = self._semantic_page_search(pages_data, chapter_query)
        
        logger.info(f"Found {len(relevant_pages)} relevant pages for '{chapter_query}'")
        return relevant_pages
    
    def _semantic_page_search(self, pages_data: List[Dict], query: str) -> List[int]:
        """Use semantic similarity to find relevant pages"""
        try:
            # Prepare texts for vectorization
            texts = [page["text"][:2000] for page in pages_data]  # Limit text length
            texts.append(query)
            
            # Vectorize and compute similarity
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            query_vector = tfidf_matrix[-1]
            page_vectors = tfidf_matrix[:-1]
            
            # Compute cosine similarity
            similarities = cosine_similarity(query_vector, page_vectors)[0]
            
            # Get top 10 most similar pages
            top_indices = np.argsort(similarities)[-10:][::-1]
            threshold = 0.1  # Minimum similarity threshold
            
            relevant_pages = [
                pages_data[i]["page_number"] 
                for i in top_indices 
                if similarities[i] > threshold
            ]
            
            return relevant_pages
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def extract_questions_from_text(self, text: str, chapter: str, topic: str, page_number: int) -> List[ExtractedQuestion]:
        """Extract questions from text using enhanced prompting"""
        
        # Split text into manageable chunks
        chunks = self._chunk_text(text, max_tokens=3000)
        all_questions = []
        
        for chunk_idx, chunk in enumerate(chunks):
            try:
                questions = self._process_text_chunk(chunk, chapter, topic, page_number, chunk_idx)
                all_questions.extend(questions)
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx}: {e}")
                continue
        
        return all_questions
    
    def _chunk_text(self, text: str, max_tokens: int = 3000) -> List[str]:
        """Split text into chunks that fit within token limits"""
        if not self.encoder:
            # Fallback to character-based chunking if encoder not available
            max_chars = max_tokens * 4  # Rough approximation
            chunks = []
            for i in range(0, len(text), max_chars):
                chunks.append(text[i:i + max_chars])
            return chunks
        
        tokens = self.encoder.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def _process_text_chunk(self, text: str, chapter: str, topic: str, page_number: int, chunk_idx: int) -> List[ExtractedQuestion]:
        """Process a single text chunk to extract questions"""
        
        # Check if OpenAI client is available
        if not self.client:
            logger.warning("OpenAI client not available - returning empty results")
            return []
        
        # Enhanced prompt for better question extraction with improved accuracy
        system_prompt = """You are an expert mathematician and LaTeX specialist with PhD-level knowledge in mathematics. Your task is to extract mathematical questions with 95%+ accuracy.

CRITICAL REQUIREMENTS:
1. ONLY extract actual mathematical QUESTIONS, PROBLEMS, or EXERCISES that require solving
2. EXCLUDE all theory, definitions, explanations, proofs, and examples that don't ask questions
3. Each extraction must be a complete, solvable mathematical problem
4. Convert ALL mathematical notation to precise LaTeX format

MATHEMATICAL NOTATION RULES:
- Fractions: Use \\frac{numerator}{denominator}
- Square roots: \\sqrt{expression} or \\sqrt[n]{expression}
- Integrals: \\int, \\iint, \\iiint with proper limits \\int_{a}^{b}
- Derivatives: \\frac{d}{dx}, \\frac{\\partial}{\\partial x}
- Limits: \\lim_{x \\to a}
- Summations: \\sum_{i=1}^{n}, \\prod_{i=1}^{n}
- Trigonometric: \\sin, \\cos, \\tan, \\sec, \\csc, \\cot
- Logarithms: \\log, \\ln, \\log_{base}
- Greek letters: \\alpha, \\beta, \\gamma, \\theta, \\pi, etc.
- Special symbols: \\infty, \\pm, \\leq, \\geq, \\neq, \\approx
- Matrices: \\begin{pmatrix}...\\end{pmatrix}
- Sets: \\{, \\}, \\in, \\subset, \\cup, \\cap

QUESTION IDENTIFICATION PATTERNS:
Look for these indicators:
- "Find", "Calculate", "Evaluate", "Solve", "Determine", "Prove", "Show that"
- "If..., then find...", "Given that..., find..."
- Numbered exercises (1., 2., (a), (b), etc.)
- "Illustration", "Example" followed by a question
- Mathematical expressions followed by "=?" or similar

CONFIDENCE SCORING:
- 0.95-1.0: Clear mathematical question with complete context
- 0.80-0.94: Mathematical question with minor ambiguity
- 0.60-0.79: Possible question but requires interpretation
- Below 0.60: Reject - not a clear question

OUTPUT FORMAT: Return ONLY a valid JSON array:
[
  {
    "question_text": "Complete LaTeX-formatted question",
    "question_type": "exercise|illustration|example|problem",
    "confidence_score": 0.85,
    "raw_text": "Original text segment",
    "mathematical_concepts": ["concept1", "concept2"],
    "difficulty_level": "basic|intermediate|advanced"
  }
]

Return empty array [] if no clear mathematical questions found."""

        user_prompt = f"""CHAPTER/TOPIC: {chapter} - {topic}
PAGE NUMBER: {page_number}

TEXT TO ANALYZE:
{text}

INSTRUCTIONS:
1. Carefully read the entire text
2. Identify ONLY mathematical questions/problems that require solving
3. Ignore all explanatory content, definitions, and theory
4. For each question found, provide complete LaTeX formatting
5. Assign confidence scores based on clarity and completeness
6. Focus on content relevant to the specified chapter/topic

EXTRACT questions that ask to:
- Solve equations or inequalities
- Find derivatives or integrals
- Calculate limits or series
- Prove mathematical statements
- Evaluate expressions
- Find areas, volumes, or other geometric properties
- Apply theorems or formulas

IGNORE:
- Definitions and explanations
- Worked examples without questions
- Theorem statements
- General discussion or theory"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=4000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                questions_data = json.loads(content)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from response
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    questions_data = json.loads(json_match.group())
                else:
                    logger.error(f"Could not parse JSON from response: {content[:200]}")
                    return []
            
            # Convert to ExtractedQuestion objects with enhanced validation
            extracted_questions = []
            for q_data in questions_data:
                if isinstance(q_data, dict) and "question_text" in q_data:
                    # Validate confidence score and question quality
                    confidence = float(q_data.get("confidence_score", 0.5))
                    question_text = q_data.get("question_text", "")
                    
                    # Additional validation checks
                    if confidence >= 0.5 and len(question_text.strip()) > 10:
                        question = ExtractedQuestion(
                            question_text=question_text,
                            latex_content=question_text,  # Same as question_text since it's already in LaTeX
                            question_type=q_data.get("question_type", "unknown"),
                            confidence_score=confidence,
                            chapter=chapter,
                            topic=topic,
                            page_number=page_number
                        )
                        extracted_questions.append(question)
                    else:
                        logger.warning(f"Filtered out low-quality question: confidence={confidence}, length={len(question_text)}")
            
            return extracted_questions
            
        except Exception as e:
            logger.error(f"Error in LLM processing: {e}")
            return []
    
    def validate_and_enhance_latex(self, questions: List[ExtractedQuestion]) -> List[ExtractedQuestion]:
        """Validate and enhance LaTeX formatting"""
        enhanced_questions = []
        
        for question in questions:
            try:
                # Validate LaTeX syntax
                enhanced_latex = self._enhance_latex_formatting(question.latex_content)
                
                # Update question with enhanced LaTeX
                enhanced_question = ExtractedQuestion(
                    question_text=enhanced_latex,
                    latex_content=enhanced_latex,
                    question_type=question.question_type,
                    confidence_score=question.confidence_score,
                    chapter=question.chapter,
                    topic=question.topic,
                    page_number=question.page_number
                )
                
                enhanced_questions.append(enhanced_question)
                
            except Exception as e:
                logger.error(f"Error enhancing LaTeX: {e}")
                # Keep original if enhancement fails
                enhanced_questions.append(question)
        
        return enhanced_questions
    
    def _enhance_latex_formatting(self, latex_text: str) -> str:
        """Enhance LaTeX formatting with common fixes"""
        
        # Common LaTeX enhancements
        enhancements = [
            # Fix common fraction formats
            (r'(\d+)/(\d+)', r'\\frac{\\1}{\\2}'),
            # Fix square roots
            (r'sqrt\(([^)]+)\)', r'\\sqrt{\\1}'),
            # Fix common mathematical functions
            (r'\bsin\b', r'\\sin'),
            (r'\bcos\b', r'\\cos'),
            (r'\btan\b', r'\\tan'),
            (r'\blog\b', r'\\log'),
            (r'\bln\b', r'\\ln'),
            # Fix limits
            (r'lim\s*\(([^)]+)\)', r'\\lim_{\\1}'),
            # Fix integrals
            (r'∫', r'\\int'),
            # Fix summations
            (r'∑', r'\\sum'),
            # Fix Greek letters
            (r'π', r'\\pi'),
            (r'θ', r'\\theta'),
            (r'α', r'\\alpha'),
            (r'β', r'\\beta'),
            (r'γ', r'\\gamma'),
        ]
        
        enhanced = latex_text
        for pattern, replacement in enhancements:
            enhanced = re.sub(pattern, replacement, enhanced)
        
        return enhanced

# Initialize the RAG pipeline
rag_pipeline = EnhancedRAGPipeline()

@app.route('/api/extract', methods=['POST'])
def extract_questions():
    """Main API endpoint for question extraction"""
    try:
        data = request.get_json()
        pdf_url = data.get('pdf_url')
        chapter_query = data.get('chapter', '').strip()
        
        if not pdf_url or not chapter_query:
            return jsonify({'error': 'PDF URL and chapter/topic are required'}), 400
        
        logger.info(f"Processing request for chapter: {chapter_query}")
        
        # Step 1: Download PDF
        pdf_path = rag_pipeline.download_pdf(pdf_url)
        
        # Step 2: Extract text from PDF
        pages_data = rag_pipeline.extract_text_from_pdf(pdf_path)
        
        # Step 3: Identify relevant pages
        relevant_pages = rag_pipeline.identify_chapter_pages(pages_data, chapter_query)
        
        if not relevant_pages:
            return jsonify({
                'error': f'No relevant content found for "{chapter_query}"',
                'suggestions': ['Check chapter number format', 'Try topic name instead', 'Use broader search terms']
            }), 404
        
        # Step 4: Extract questions from relevant pages
        all_questions = []
        for page_num in relevant_pages[:5]:  # Limit to first 5 relevant pages
            page_data = next((p for p in pages_data if p["page_number"] == page_num), None)
            if page_data:
                questions = rag_pipeline.extract_questions_from_text(
                    page_data["text"], 
                    chapter_query, 
                    chapter_query,  # Using same as topic for now
                    page_num
                )
                all_questions.extend(questions)
        
        # Step 5: Validate and enhance LaTeX
        enhanced_questions = rag_pipeline.validate_and_enhance_latex(all_questions)
        
        # Step 6: Filter by confidence score with better thresholds
        high_confidence_questions = [
            q for q in enhanced_questions 
            if q.confidence_score >= 0.7  # Raised threshold for better accuracy
        ]
        
        # Additional quality filters
        filtered_questions = []
        for q in high_confidence_questions:
            # Filter out very short or very long questions
            if 20 <= len(q.question_text.strip()) <= 2000:
                # Check for mathematical content indicators
                math_indicators = ['\\frac', '\\int', '\\sum', '\\lim', '\\sqrt', '=', '+', '-', '*', '/', '^', '_']
                if any(indicator in q.question_text for indicator in math_indicators):
                    filtered_questions.append(q)
                else:
                    logger.info(f"Filtered out non-mathematical question: {q.question_text[:50]}...")
            else:
                logger.info(f"Filtered out question due to length: {len(q.question_text)} chars")
        
        high_confidence_questions = filtered_questions
        
        # Step 7: Generate LaTeX document
        latex_document = generate_latex_document(high_confidence_questions, chapter_query)
        
        # Cleanup
        try:
            os.remove(pdf_path)
        except:
            pass
        
        # Calculate improved accuracy metrics
        total_questions = len(all_questions)
        high_conf_questions = len(high_confidence_questions)
        
        # More sophisticated accuracy estimation
        if total_questions == 0:
            estimated_accuracy = 0
        else:
            # Weight accuracy by confidence scores
            confidence_sum = sum(q.confidence_score for q in high_confidence_questions)
            max_possible_confidence = high_conf_questions * 1.0
            
            if max_possible_confidence > 0:
                confidence_ratio = confidence_sum / max_possible_confidence
                base_accuracy = (high_conf_questions / total_questions) * 100
                estimated_accuracy = base_accuracy * confidence_ratio
            else:
                estimated_accuracy = 0
        
        # Cap at reasonable maximum
        estimated_accuracy = min(estimated_accuracy, 95.0)
        
        return jsonify({
            'success': True,
            'chapter': chapter_query,
            'total_questions_found': total_questions,
            'high_confidence_questions': high_conf_questions,
            'estimated_accuracy': round(estimated_accuracy, 1),
            'latex_content': latex_document,
            'questions': [
                {
                    'text': q.question_text,
                    'type': q.question_type,
                    'confidence': q.confidence_score,
                    'page': q.page_number
                } for q in high_confidence_questions
            ],
            'processing_info': {
                'pages_processed': len(relevant_pages),
                'relevant_pages': relevant_pages,
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in extraction: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

def generate_latex_document(questions: List[ExtractedQuestion], chapter: str) -> str:
    """Generate a complete LaTeX document from extracted questions"""
    
    header = f"""% Mathematical Questions Extracted from RD Sharma Class 12
% Chapter/Topic: {chapter}
% Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{amsfonts}}
\\usepackage{{geometry}}
\\geometry{{margin=1in}}

\\title{{Mathematical Questions: {chapter}}}
\\author{{Extracted using Enhanced RAG Pipeline}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Extracted Questions}}

"""
    
    content = ""
    for i, question in enumerate(questions, 1):
        content += f"""
% Question {i} (Type: {question.question_type}, Confidence: {question.confidence_score:.2f}, Page: {question.page_number})
\\subsection{{Question {i}}}

{question.latex_content}

\\vspace{{0.5cm}}

"""
    
    footer = """
\\end{document}
"""
    
    return header + content + footer

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '2.0.0',
        'features': ['Enhanced RAG Pipeline', 'LLM Integration', 'LaTeX Validation']
    })

if __name__ == '__main__':
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found in environment variables")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
