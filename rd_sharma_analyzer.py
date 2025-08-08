"""
RD Sharma Class 12th Volume 2 MCQ Analyzer
Deep analysis and extraction of every single problem for RAG pipeline training
"""

import pdfplumber
import re
import json
import pandas as pd
from collections import defaultdict
import numpy as np
from datetime import datetime

class RDSharmaAnalyzer:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.questions = []
        self.chapters = {}
        self.topics = defaultdict(list)
        self.difficulty_levels = defaultdict(list)
        self.question_types = defaultdict(list)
        
    def extract_text_from_pdf(self):
        """Extract all text from PDF with page information"""
        print("🔍 Starting PDF text extraction...")
        all_text = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"📄 Total pages to process: {total_pages}")
            
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"Processing page {page_num}/{total_pages}...", end="\r")
                text = page.extract_text()
                if text:
                    all_text.append({
                        'page': page_num,
                        'text': text,
                        'bbox': page.bbox
                    })
        
        print(f"\n✅ Successfully extracted text from {len(all_text)} pages")
        return all_text
    
    def identify_question_patterns(self, text):
        """Identify different question patterns in RD Sharma"""
        patterns = {
            'mcq_numbered': r'\b(\d+)\.\s*(.+?)(?=\n\s*(?:\([A-D]\)|[A-D]\.))',
            'mcq_with_options': r'(\([A-D]\)|[A-D]\.)(.+?)(?=\n|$)',
            'chapter_headers': r'CHAPTER\s+(\d+)\s*[-:]\s*(.+)',
            'exercise_headers': r'EXERCISE\s+(\d+\.\d+)',
            'topic_headers': r'(?:TOPIC|SECTION)\s*[-:]\s*(.+)',
            'question_numbers': r'^(\d+)\.\s+',
            'formulas': r'\$(.+?)\$|\\(.+?)\\',
            'answer_key': r'(?:ANSWER|ANS):\s*([A-D])',
            'solution_start': r'(?:SOLUTION|SOL):\s*',
            'hint_start': r'(?:HINT):\s*'
        }
        
        matches = {}
        for pattern_name, pattern in patterns.items():
            matches[pattern_name] = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
        
        return matches
    
    def extract_questions_deep_analysis(self, page_data):
        """Deep analysis to extract every single question with context"""
        questions_found = []
        
        for page_info in page_data:
            page_num = page_info['page']
            text = page_info['text']
            
            # Split text into lines for detailed analysis
            lines = text.split('\n')
            current_question = None
            current_options = []
            in_question = False
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Check for question number pattern
                question_match = re.match(r'^(\d+)\.\s+(.+)', line)
                if question_match:
                    # Save previous question if exists
                    if current_question:
                        questions_found.append(self.format_question(
                            current_question, current_options, page_num
                        ))
                    
                    # Start new question
                    current_question = {
                        'number': question_match.group(1),
                        'text': question_match.group(2),
                        'page': page_num,
                        'full_text': line
                    }
                    current_options = []
                    in_question = True
                    continue
                
                # Check for options (A), (B), (C), (D) or A. B. C. D.
                option_match = re.match(r'^\(([A-D])\)\s*(.+)|^([A-D])\.\s*(.+)', line)
                if option_match and in_question:
                    if option_match.group(1):  # (A) format
                        option_letter = option_match.group(1)
                        option_text = option_match.group(2)
                    else:  # A. format
                        option_letter = option_match.group(3)
                        option_text = option_match.group(4)
                    
                    current_options.append({
                        'letter': option_letter,
                        'text': option_text
                    })
                    continue
                
                # If we're in a question and line doesn't match option pattern,
                # it might be continuation of question text
                if in_question and current_question and not option_match:
                    # Check if this looks like continuation of question
                    if not re.match(r'^\d+\.', line) and not re.match(r'^CHAPTER|^EXERCISE|^ANSWER', line):
                        current_question['text'] += ' ' + line
                        current_question['full_text'] += ' ' + line
            
            # Don't forget the last question
            if current_question:
                questions_found.append(self.format_question(
                    current_question, current_options, page_num
                ))
        
        return questions_found
    
    def format_question(self, question_data, options, page_num):
        """Format question with all metadata"""
        return {
            'id': f"RDS_{page_num}_{question_data['number']}",
            'number': question_data['number'],
            'question_text': question_data['text'],
            'full_question': question_data['full_text'],
            'options': options,
            'page': page_num,
            'chapter': self.identify_chapter(question_data['text']),
            'topic': self.identify_topic(question_data['text']),
            'difficulty': self.estimate_difficulty(question_data['text'], options),
            'question_type': self.classify_question_type(question_data['text']),
            'has_formula': self.contains_mathematical_notation(question_data['text']),
            'word_count': len(question_data['text'].split()),
            'extracted_at': datetime.now().isoformat()
        }
    
    def identify_chapter(self, question_text):
        """Identify chapter based on question content"""
        chapter_keywords = {
            'Calculus': ['derivative', 'integral', 'limit', 'continuity', 'differentiation'],
            'Algebra': ['equation', 'polynomial', 'matrix', 'determinant'],
            'Trigonometry': ['sin', 'cos', 'tan', 'trigonometric', 'angle'],
            'Geometry': ['circle', 'triangle', 'coordinate', 'distance', 'area'],
            'Probability': ['probability', 'random', 'event', 'sample space'],
            'Statistics': ['mean', 'median', 'mode', 'standard deviation', 'variance'],
            'Vector': ['vector', 'dot product', 'cross product', 'magnitude'],
            '3D Geometry': ['plane', '3d', 'three dimensional', 'direction cosines']
        }
        
        question_lower = question_text.lower()
        for chapter, keywords in chapter_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return chapter
        
        return 'General'
    
    def identify_topic(self, question_text):
        """Identify specific topic within chapter"""
        # This can be expanded with more sophisticated NLP
        question_lower = question_text.lower()
        
        if 'derivative' in question_lower:
            return 'Derivatives'
        elif 'integral' in question_lower:
            return 'Integration'
        elif 'limit' in question_lower:
            return 'Limits'
        elif 'matrix' in question_lower:
            return 'Matrices'
        elif 'probability' in question_lower:
            return 'Probability'
        else:
            return 'Mixed'
    
    def estimate_difficulty(self, question_text, options):
        """Estimate difficulty level based on various factors"""
        difficulty_score = 0
        
        # Length factor
        if len(question_text.split()) > 30:
            difficulty_score += 1
        
        # Mathematical complexity
        complex_terms = ['integral', 'derivative', 'matrix', 'determinant', 'probability']
        difficulty_score += sum(1 for term in complex_terms if term in question_text.lower())
        
        # Number of options with formulas
        formula_options = sum(1 for opt in options if self.contains_mathematical_notation(opt.get('text', '')))
        difficulty_score += formula_options * 0.5
        
        if difficulty_score <= 1:
            return 'Easy'
        elif difficulty_score <= 3:
            return 'Medium'
        else:
            return 'Hard'
    
    def classify_question_type(self, question_text):
        """Classify the type of mathematical question"""
        question_lower = question_text.lower()
        
        if any(word in question_lower for word in ['find', 'calculate', 'compute']):
            return 'Calculation'
        elif any(word in question_lower for word in ['prove', 'show', 'verify']):
            return 'Proof'
        elif any(word in question_lower for word in ['which', 'what', 'identify']):
            return 'Conceptual'
        elif 'graph' in question_lower or 'plot' in question_lower:
            return 'Graphical'
        else:
            return 'Application'
    
    def contains_mathematical_notation(self, text):
        """Check if text contains mathematical formulas or notation"""
        math_patterns = [
            r'\$.*?\$',  # LaTeX math
            r'\\[a-zA-Z]+',  # LaTeX commands
            r'[∫∑∏√π∞±≤≥≠]',  # Math symbols
            r'\^[0-9]',  # Superscripts
            r'_[0-9]',  # Subscripts
            r'[a-zA-Z]\([x-z]\)',  # Functions like f(x)
        ]
        
        return any(re.search(pattern, text) for pattern in math_patterns)
    
    def generate_statistics(self):
        """Generate comprehensive statistics about extracted questions"""
        if not self.questions:
            return {}
        
        stats = {
            'total_questions': len(self.questions),
            'questions_per_chapter': {},
            'questions_per_difficulty': {},
            'questions_per_type': {},
            'questions_with_formulas': 0,
            'average_word_count': 0,
            'page_distribution': {}
        }
        
        for q in self.questions:
            # Chapter distribution
            chapter = q['chapter']
            stats['questions_per_chapter'][chapter] = stats['questions_per_chapter'].get(chapter, 0) + 1
            
            # Difficulty distribution
            difficulty = q['difficulty']
            stats['questions_per_difficulty'][difficulty] = stats['questions_per_difficulty'].get(difficulty, 0) + 1
            
            # Type distribution
            q_type = q['question_type']
            stats['questions_per_type'][q_type] = stats['questions_per_type'].get(q_type, 0) + 1
            
            # Formula count
            if q['has_formula']:
                stats['questions_with_formulas'] += 1
            
            # Page distribution
            page = q['page']
            stats['page_distribution'][page] = stats['page_distribution'].get(page, 0) + 1
        
        # Calculate averages
        stats['average_word_count'] = np.mean([q['word_count'] for q in self.questions])
        
        return stats
    
    def save_to_json(self, output_file):
        """Save all extracted data to JSON"""
        output_data = {
            'metadata': {
                'source_pdf': self.pdf_path,
                'extraction_date': datetime.now().isoformat(),
                'total_questions': len(self.questions)
            },
            'statistics': self.generate_statistics(),
            'questions': self.questions
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Data saved to {output_file}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting RD Sharma Deep Analysis Pipeline...")
        print("="*60)
        
        # Step 1: Extract text
        page_data = self.extract_text_from_pdf()
        
        # Step 2: Deep question analysis
        print("\n🔍 Performing deep question analysis...")
        self.questions = self.extract_questions_deep_analysis(page_data)
        print(f"✅ Extracted {len(self.questions)} questions")
        
        # Step 3: Generate statistics
        print("\n📊 Generating comprehensive statistics...")
        stats = self.generate_statistics()
        
        # Display key statistics
        print(f"\n📈 ANALYSIS RESULTS:")
        print(f"   Total Questions: {stats['total_questions']}")
        print(f"   Questions with Formulas: {stats['questions_with_formulas']}")
        print(f"   Average Word Count: {stats['average_word_count']:.1f}")
        print(f"\n📚 Chapter Distribution:")
        for chapter, count in sorted(stats['questions_per_chapter'].items()):
            print(f"   {chapter}: {count} questions")
        print(f"\n🎯 Difficulty Distribution:")
        for difficulty, count in stats['questions_per_difficulty'].items():
            print(f"   {difficulty}: {count} questions")
        
        return self.questions, stats

# Main execution
if __name__ == "__main__":
    pdf_path = r"C:\Users\HP\Downloads\RD-SHARMA CLASS 12TH VOLUME 2 MCQS (R.D.SHARMA) (Z-Library).pdf"
    
    analyzer = RDSharmaAnalyzer(pdf_path)
    questions, statistics = analyzer.run_complete_analysis()
    
    # Save results
    output_file = "rd_sharma_questions_complete.json"
    analyzer.save_to_json(output_file)
    
    print(f"\n🎉 Analysis complete! {len(questions)} questions extracted and analyzed.")
    print(f"📁 Results saved to: {output_file}")
