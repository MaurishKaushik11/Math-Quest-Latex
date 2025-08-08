"""
Refined RD Sharma MCQ Extractor - Better pattern matching for actual MCQ questions
"""

import pdfplumber
import re
import json
from collections import defaultdict
import numpy as np
from datetime import datetime

class RefinedRDSharmaExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.mcq_questions = []
        
    def extract_text_from_pdf(self):
        """Extract all text from PDF with page information"""
        print("🔍 Starting refined PDF text extraction...")
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
    
    def find_mcq_questions(self, page_data):
        """Find actual MCQ questions with proper structure"""
        mcq_questions = []
        
        for page_info in page_data:
            page_num = page_info['page']
            text = page_info['text']
            
            # Look for MCQ patterns - questions followed by options (A), (B), (C), (D)
            # Pattern: Question number followed by question text, then options
            
            # Split text into blocks
            blocks = self.split_into_question_blocks(text)
            
            for block in blocks:
                mcq = self.parse_mcq_block(block, page_num)
                if mcq:
                    mcq_questions.append(mcq)
        
        return mcq_questions
    
    def split_into_question_blocks(self, text):
        """Split text into potential question blocks"""
        # Look for numbered patterns that could be questions
        lines = text.split('\n')
        blocks = []
        current_block = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a question number pattern
            if re.match(r'^\d+\.\s', line):
                # Save previous block
                if current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
                current_block.append(line)
            elif current_block:
                current_block.append(line)
        
        # Don't forget last block
        if current_block:
            blocks.append('\n'.join(current_block))
            
        return blocks
    
    def parse_mcq_block(self, block, page_num):
        """Parse a text block to extract MCQ structure"""
        lines = block.split('\n')
        
        # First line should contain the question
        if not lines:
            return None
            
        first_line = lines[0].strip()
        question_match = re.match(r'^(\d+)\.\s+(.+)', first_line)
        if not question_match:
            return None
            
        question_num = question_match.group(1)
        question_text = question_match.group(2)
        
        # Look for options in subsequent lines
        options = []
        remaining_question_text = []
        found_options = False
        
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
                
            # Check for option patterns: (A), (B), (C), (D) or A., B., C., D.
            option_match = re.match(r'^\(([A-D])\)\s*(.+)|^([A-D])\.\s*(.+)', line)
            
            if option_match:
                found_options = True
                if option_match.group(1):  # (A) format
                    letter = option_match.group(1)
                    text = option_match.group(2)
                else:  # A. format
                    letter = option_match.group(3)
                    text = option_match.group(4)
                
                options.append({
                    'letter': letter,
                    'text': text.strip()
                })
            elif not found_options:
                # This might be continuation of question text
                remaining_question_text.append(line)
        
        # Combine question text
        if remaining_question_text:
            question_text += ' ' + ' '.join(remaining_question_text)
        
        # Only return if we found at least 2 options (some questions might have incomplete options)
        if len(options) >= 2:
            return self.format_mcq_question(question_num, question_text, options, page_num, block)
        
        return None
    
    def format_mcq_question(self, question_num, question_text, options, page_num, full_block):
        """Format the MCQ question with metadata"""
        return {
            'id': f"RDS_MCQ_{page_num}_{question_num}",
            'number': question_num,
            'question_text': question_text.strip(),
            'options': options,
            'page': page_num,
            'chapter': self.identify_chapter(question_text),
            'topic': self.identify_topic(question_text),
            'difficulty': self.estimate_difficulty(question_text, options),
            'question_type': self.classify_question_type(question_text),
            'has_formula': self.contains_mathematical_notation(question_text + ' ' + ' '.join([opt['text'] for opt in options])),
            'word_count': len(question_text.split()),
            'num_options': len(options),
            'full_block': full_block,
            'extracted_at': datetime.now().isoformat()
        }
    
    def identify_chapter(self, question_text):
        """Identify chapter based on question content"""
        chapter_keywords = {
            'Integration': ['integral', 'integrate', '∫', 'dx', 'definite integral'],
            'Differentiation': ['derivative', 'differentiate', 'dy/dx', "f'(x)", 'tangent'],
            'Limits': ['limit', 'lim', 'tends to', 'approaches'],
            'Continuity': ['continuous', 'continuity', 'discontinuous'],
            'Matrix': ['matrix', 'determinant', 'transpose', 'inverse matrix'],
            'Vector': ['vector', 'dot product', 'cross product', 'magnitude', 'unit vector'],
            'Trigonometry': ['sin', 'cos', 'tan', 'sec', 'cosec', 'cot', 'trigonometric'],
            'Probability': ['probability', 'event', 'sample space', 'random', 'odds'],
            'Statistics': ['mean', 'median', 'mode', 'variance', 'standard deviation'],
            '3D Geometry': ['plane', 'line in space', 'direction cosines', 'direction ratios'],
            'Coordinate Geometry': ['circle', 'parabola', 'ellipse', 'hyperbola', 'coordinate'],
            'Function': ['function', 'domain', 'range', 'onto', 'one-one'],
            'Relations': ['relation', 'equivalence', 'reflexive', 'symmetric', 'transitive']
        }
        
        question_lower = question_text.lower()
        for chapter, keywords in chapter_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return chapter
        
        return 'General Mathematics'
    
    def identify_topic(self, question_text):
        """Identify specific topic within chapter"""
        question_lower = question_text.lower()
        
        topic_map = {
            'derivative': 'Differentiation',
            'integral': 'Integration', 
            'limit': 'Limits',
            'matrix': 'Matrices',
            'determinant': 'Determinants',
            'probability': 'Probability Theory',
            'vector': 'Vector Algebra',
            'trigonometric': 'Trigonometric Functions',
            'function': 'Functions',
            'geometry': 'Coordinate Geometry'
        }
        
        for keyword, topic in topic_map.items():
            if keyword in question_lower:
                return topic
                
        return 'Mixed Topics'
    
    def estimate_difficulty(self, question_text, options):
        """Estimate difficulty based on content complexity"""
        difficulty_score = 0
        
        # Length factor
        if len(question_text.split()) > 25:
            difficulty_score += 1
        
        # Complex mathematical terms
        complex_terms = [
            'derivative', 'integral', 'matrix', 'determinant', 'eigenvalue', 
            'probability', 'statistics', 'limit', 'continuity', 'differential'
        ]
        difficulty_score += sum(1 for term in complex_terms if term in question_text.lower())
        
        # Formula complexity in options
        formula_count = sum(1 for opt in options if self.contains_mathematical_notation(opt['text']))
        difficulty_score += formula_count * 0.5
        
        # Option length complexity
        avg_option_length = np.mean([len(opt['text'].split()) for opt in options])
        if avg_option_length > 10:
            difficulty_score += 1
        
        if difficulty_score <= 1:
            return 'Easy'
        elif difficulty_score <= 3:
            return 'Medium'
        else:
            return 'Hard'
    
    def classify_question_type(self, question_text):
        """Classify the mathematical question type"""
        question_lower = question_text.lower()
        
        if any(word in question_lower for word in ['find', 'calculate', 'compute', 'evaluate']):
            return 'Calculation'
        elif any(word in question_lower for word in ['prove', 'show that', 'verify', 'demonstrate']):
            return 'Proof'
        elif any(word in question_lower for word in ['which', 'what is', 'identify', 'select']):
            return 'Conceptual'
        elif any(word in question_lower for word in ['graph', 'plot', 'sketch', 'draw']):
            return 'Graphical'
        elif any(word in question_lower for word in ['solve', 'determine', 'obtain']):
            return 'Problem Solving'
        else:
            return 'Application'
    
    def contains_mathematical_notation(self, text):
        """Check for mathematical formulas and notation"""
        math_patterns = [
            r'\$.*?\$',  # LaTeX math mode
            r'\\[a-zA-Z]+',  # LaTeX commands
            r'[∫∑∏√π∞±≤≥≠→∈∉∪∩⊂⊃]',  # Math symbols
            r'\^[0-9\{\}]',  # Superscripts
            r'_[0-9\{\}]',  # Subscripts
            r'[a-zA-Z]\([a-z]\)',  # Functions like f(x)
            r'[0-9]+/[0-9]+',  # Fractions
            r'log|ln|sin|cos|tan|exp',  # Math functions
            r'dx|dy|d[a-z]',  # Differentials
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in math_patterns)
    
    def generate_statistics(self):
        """Generate comprehensive statistics"""
        if not self.mcq_questions:
            return {}
        
        stats = {
            'total_mcq_questions': len(self.mcq_questions),
            'questions_per_chapter': {},
            'questions_per_difficulty': {},
            'questions_per_type': {},
            'questions_with_formulas': 0,
            'average_word_count': 0,
            'average_options_per_question': 0,
            'page_distribution': {}
        }
        
        for q in self.mcq_questions:
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
        stats['average_word_count'] = np.mean([q['word_count'] for q in self.mcq_questions])
        stats['average_options_per_question'] = np.mean([q['num_options'] for q in self.mcq_questions])
        
        return stats
    
    def save_refined_data(self, output_file):
        """Save refined MCQ data"""
        output_data = {
            'metadata': {
                'source_pdf': self.pdf_path,
                'extraction_date': datetime.now().isoformat(),
                'total_mcq_questions': len(self.mcq_questions),
                'extraction_method': 'refined_pattern_matching'
            },
            'statistics': self.generate_statistics(),
            'mcq_questions': self.mcq_questions
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Refined MCQ data saved to {output_file}")
    
    def run_refined_extraction(self):
        """Run the refined MCQ extraction pipeline"""
        print("🚀 Starting Refined RD Sharma MCQ Extraction...")
        print("=" * 60)
        
        # Extract text
        page_data = self.extract_text_from_pdf()
        
        # Find MCQ questions
        print("\n🔍 Finding properly formatted MCQ questions...")
        self.mcq_questions = self.find_mcq_questions(page_data)
        print(f"✅ Found {len(self.mcq_questions)} properly formatted MCQ questions")
        
        # Generate statistics
        stats = self.generate_statistics()
        
        # Display results
        print(f"\n📈 REFINED EXTRACTION RESULTS:")
        print(f"   Total MCQ Questions: {stats['total_mcq_questions']}")
        print(f"   Average Options per Question: {stats['average_options_per_question']:.1f}")
        print(f"   Questions with Math Formulas: {stats['questions_with_formulas']}")
        print(f"   Average Word Count: {stats['average_word_count']:.1f}")
        
        if stats['questions_per_chapter']:
            print(f"\n📚 Chapter Distribution:")
            for chapter, count in sorted(stats['questions_per_chapter'].items(), key=lambda x: x[1], reverse=True):
                print(f"   {chapter}: {count} questions")
        
        if stats['questions_per_difficulty']:
            print(f"\n🎯 Difficulty Distribution:")
            for difficulty, count in stats['questions_per_difficulty'].items():
                print(f"   {difficulty}: {count} questions")
        
        return self.mcq_questions, stats

# Main execution
if __name__ == "__main__":
    pdf_path = r"C:\Users\HP\Downloads\RD-SHARMA CLASS 12TH VOLUME 2 MCQS (R.D.SHARMA) (Z-Library).pdf"
    
    extractor = RefinedRDSharmaExtractor(pdf_path)
    mcq_questions, stats = extractor.run_refined_extraction()
    
    # Save results
    output_file = "rd_sharma_mcq_refined.json"
    extractor.save_refined_data(output_file)
    
    # Show some sample questions
    if mcq_questions:
        print(f"\n🎯 SAMPLE MCQ QUESTIONS:")
        print("=" * 50)
        for i, q in enumerate(mcq_questions[:5]):
            print(f"\nQuestion {q['number']} (Page {q['page']}):")
            print(f"Text: {q['question_text'][:100]}...")
            print("Options:")
            for opt in q['options']:
                print(f"  {opt['letter']}. {opt['text'][:60]}...")
            print(f"Chapter: {q['chapter']}, Difficulty: {q['difficulty']}")
    
    print(f"\n🎉 Refined extraction complete! Ready for RAG training.")
