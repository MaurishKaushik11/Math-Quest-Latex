"""
Advanced MCQ Extractor for RD Sharma - Multiple pattern strategies
"""

import pdfplumber
import re
import json
from collections import defaultdict
import numpy as np
from datetime import datetime

class AdvancedMCQExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.all_questions = []
        self.page_samples = {}
        
    def sample_pages_for_analysis(self, sample_size=20):
        """Sample random pages to understand the structure"""
        print("🔍 Sampling pages to analyze PDF structure...")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            # Sample pages from different sections
            sample_pages = [
                50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
                550, 600, 650, 700, 750
            ]
            
            for page_num in sample_pages:
                if page_num <= total_pages:
                    page = pdf.pages[page_num - 1]  # 0-indexed
                    text = page.extract_text()
                    if text:
                        self.page_samples[page_num] = text
                        print(f"✅ Sampled page {page_num}")
        
        return self.page_samples
    
    def analyze_page_structure(self):
        """Analyze the structure of sampled pages"""
        print("\n📊 Analyzing page structures...")
        
        patterns_found = {}
        
        for page_num, text in self.page_samples.items():
            print(f"\n--- PAGE {page_num} ANALYSIS ---")
            print(f"First 200 chars: {text[:200]}...")
            
            # Look for various patterns
            patterns = {
                'numbered_questions': len(re.findall(r'^\s*\d+\.\s+', text, re.MULTILINE)),
                'options_parentheses': len(re.findall(r'\([A-D]\)', text)),
                'options_periods': len(re.findall(r'\b[A-D]\.\s', text)),
                'multiple_choice': len(re.findall(r'[A-D]\)\s*[A-Z]', text)),
                'mathematical_notation': len(re.findall(r'[∫∑∏√π∞±≤≥≠]|dx|dy|sin|cos|tan', text)),
                'question_keywords': len(re.findall(r'\b(find|calculate|solve|determine|prove|show)\b', text, re.IGNORECASE))
            }
            
            patterns_found[page_num] = patterns
            
            for pattern, count in patterns.items():
                if count > 0:
                    print(f"  {pattern}: {count}")
        
        return patterns_found
    
    def extract_all_text_with_coordinates(self):
        """Extract text with spatial coordinates for better parsing"""
        print("\n🔍 Extracting all text with spatial information...")
        
        all_content = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num in range(1, min(total_pages + 1, 100)):  # First 100 pages for now
                print(f"Processing page {page_num}/{min(total_pages, 100)}...", end="\r")
                
                page = pdf.pages[page_num - 1]
                
                # Extract with different methods
                basic_text = page.extract_text()
                
                # Try to extract text with more structure
                try:
                    words = page.extract_words()
                    chars = page.chars
                    
                    all_content.append({
                        'page': page_num,
                        'text': basic_text,
                        'words': words[:50],  # First 50 words for analysis
                        'char_count': len(chars) if chars else 0,
                        'bbox': page.bbox
                    })
                except Exception as e:
                    all_content.append({
                        'page': page_num,
                        'text': basic_text,
                        'words': [],
                        'char_count': 0,
                        'bbox': page.bbox
                    })
        
        print(f"\n✅ Extracted content from {len(all_content)} pages")
        return all_content
    
    def find_mcq_patterns_advanced(self, content_data):
        """Advanced pattern matching for MCQs"""
        print("\n🎯 Advanced pattern matching for MCQs...")
        
        questions_found = []
        
        for page_data in content_data:
            page_num = page_data['page']
            text = page_data['text']
            
            if not text:
                continue
            
            # Multiple strategies for finding questions
            questions_found.extend(self.strategy_1_basic_numbered(text, page_num))
            questions_found.extend(self.strategy_2_option_blocks(text, page_num))
            questions_found.extend(self.strategy_3_mathematical_context(text, page_num))
        
        # Remove duplicates and filter
        unique_questions = self.deduplicate_questions(questions_found)
        
        print(f"✅ Found {len(unique_questions)} unique MCQ candidates")
        return unique_questions
    
    def strategy_1_basic_numbered(self, text, page_num):
        """Strategy 1: Look for numbered questions followed by options"""
        questions = []
        
        # Split into lines
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for question pattern
            question_match = re.match(r'^(\d+)\.\s+(.+)', line)
            if question_match:
                question_num = question_match.group(1)
                question_text = question_match.group(2)
                
                # Look ahead for options
                options = []
                j = i + 1
                
                # Continue reading question text and options
                while j < len(lines) and j < i + 10:  # Look up to 10 lines ahead
                    next_line = lines[j].strip()
                    
                    if not next_line:
                        j += 1
                        continue
                    
                    # Check if this is an option
                    option_match = re.match(r'^\(?([A-D])\)?\s*(.+)', next_line)
                    if option_match:
                        options.append({
                            'letter': option_match.group(1),
                            'text': option_match.group(2)
                        })
                        j += 1
                    elif not options and not re.match(r'^\d+\.', next_line):
                        # Might be continuation of question
                        question_text += ' ' + next_line
                        j += 1
                    else:
                        break
                
                if len(options) >= 2:
                    questions.append({
                        'strategy': 'basic_numbered',
                        'page': page_num,
                        'number': question_num,
                        'question_text': question_text,
                        'options': options
                    })
                
                i = j
            else:
                i += 1
        
        return questions
    
    def strategy_2_option_blocks(self, text, page_num):
        """Strategy 2: Look for blocks of options and work backwards to find question"""
        questions = []
        
        # Find groups of consecutive options
        lines = text.split('\n')
        option_groups = []
        current_group = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            option_match = re.match(r'^\(?([A-D])\)?\s*(.+)', line)
            
            if option_match:
                current_group.append({
                    'line_num': i,
                    'letter': option_match.group(1),
                    'text': option_match.group(2)
                })
            else:
                if len(current_group) >= 2:
                    option_groups.append(current_group)
                current_group = []
        
        # Last group
        if len(current_group) >= 2:
            option_groups.append(current_group)
        
        # For each option group, look backwards for question
        for group in option_groups:
            start_line = group[0]['line_num']
            question_text = ""
            question_num = None
            
            # Look backwards up to 5 lines
            for i in range(max(0, start_line - 5), start_line):
                line = lines[i].strip()
                if line:
                    # Check if this looks like a question
                    question_match = re.match(r'^(\d+)\.\s+(.+)', line)
                    if question_match:
                        question_num = question_match.group(1)
                        question_text = question_match.group(2)
                        break
                    else:
                        question_text = line + ' ' + question_text
            
            if question_text and len(question_text.split()) > 3:
                questions.append({
                    'strategy': 'option_blocks',
                    'page': page_num,
                    'number': question_num or f"P{page_num}_Q{len(questions)+1}",
                    'question_text': question_text.strip(),
                    'options': [{'letter': opt['letter'], 'text': opt['text']} for opt in group]
                })
        
        return questions
    
    def strategy_3_mathematical_context(self, text, page_num):
        """Strategy 3: Focus on mathematical content"""
        questions = []
        
        # Look for mathematical expressions followed by options
        math_patterns = [
            r'∫.*?dx', r'∑.*?=', r'lim.*?→', r'f\(.*?\)', 
            r'sin|cos|tan', r'matrix', r'determinant', r'probability'
        ]
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # Check if line contains mathematical content
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in math_patterns):
                # Look for options nearby
                options = []
                question_text = line.strip()
                
                # Check following lines for options
                for j in range(i + 1, min(i + 6, len(lines))):
                    next_line = lines[j].strip()
                    option_match = re.match(r'^\(?([A-D])\)?\s*(.+)', next_line)
                    if option_match:
                        options.append({
                            'letter': option_match.group(1),
                            'text': option_match.group(2)
                        })
                
                if len(options) >= 2:
                    questions.append({
                        'strategy': 'mathematical_context',
                        'page': page_num,
                        'number': f"P{page_num}_M{len(questions)+1}",
                        'question_text': question_text,
                        'options': options
                    })
        
        return questions
    
    def deduplicate_questions(self, questions):
        """Remove duplicate questions"""
        unique_questions = []
        seen_texts = set()
        
        for q in questions:
            # Create a signature for the question
            signature = (q['question_text'][:50], len(q['options']))
            
            if signature not in seen_texts:
                seen_texts.add(signature)
                unique_questions.append(q)
        
        return unique_questions
    
    def enhance_question_metadata(self, questions):
        """Add metadata to questions"""
        enhanced = []
        
        for q in questions:
            enhanced_q = {
                'id': f"RDS_ADV_{q['page']}_{q['number']}",
                'strategy_used': q['strategy'],
                'page': q['page'],
                'number': q['number'],
                'question_text': q['question_text'],
                'options': q['options'],
                'chapter': self.identify_chapter(q['question_text']),
                'topic': self.identify_topic(q['question_text']),
                'difficulty': self.estimate_difficulty(q['question_text'], q['options']),
                'question_type': self.classify_question_type(q['question_text']),
                'has_formula': self.contains_mathematical_notation(q['question_text']),
                'word_count': len(q['question_text'].split()),
                'num_options': len(q['options']),
                'extracted_at': datetime.now().isoformat()
            }
            enhanced.append(enhanced_q)
        
        return enhanced
    
    def identify_chapter(self, question_text):
        """Identify chapter based on content"""
        chapter_keywords = {
            'Integration': ['integral', '∫', 'integrate', 'dx', 'definite'],
            'Differentiation': ['derivative', "f'", 'dy/dx', 'tangent', 'differentiate'],
            'Limits': ['limit', 'lim', 'approaches', 'tends to'],
            'Trigonometry': ['sin', 'cos', 'tan', 'sec', 'cosec', 'cot'],
            'Probability': ['probability', 'event', 'random', 'sample space'],
            'Vector': ['vector', 'dot product', 'cross product'],
            'Matrix': ['matrix', 'determinant', 'transpose'],
            'Statistics': ['mean', 'median', 'variance', 'standard deviation'],
            'Geometry': ['circle', 'triangle', 'coordinate', 'distance'],
            'Function': ['function', 'domain', 'range', 'onto', 'bijective']
        }
        
        text_lower = question_text.lower()
        for chapter, keywords in chapter_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return chapter
        
        return 'General'
    
    def identify_topic(self, question_text):
        """Identify specific topic"""
        return 'Mixed'  # Simplified for now
    
    def estimate_difficulty(self, question_text, options):
        """Estimate difficulty"""
        score = 0
        
        if len(question_text.split()) > 20:
            score += 1
        
        complex_terms = ['integral', 'derivative', 'limit', 'matrix', 'probability']
        score += sum(1 for term in complex_terms if term in question_text.lower())
        
        if score <= 1:
            return 'Easy'
        elif score <= 2:
            return 'Medium'
        else:
            return 'Hard'
    
    def classify_question_type(self, question_text):
        """Classify question type"""
        text_lower = question_text.lower()
        
        if any(word in text_lower for word in ['find', 'calculate', 'compute']):
            return 'Calculation'
        elif any(word in text_lower for word in ['which', 'what', 'identify']):
            return 'Conceptual'
        else:
            return 'Application'
    
    def contains_mathematical_notation(self, text):
        """Check for mathematical notation"""
        math_patterns = [
            r'[∫∑∏√π∞±≤≥≠]', r'sin|cos|tan', r'dx|dy', 
            r'f\(', r'\^', r'_', r'matrix', r'lim'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in math_patterns)
    
    def run_advanced_extraction(self):
        """Run the complete advanced extraction"""
        print("🚀 Starting Advanced MCQ Extraction Pipeline...")
        print("=" * 60)
        
        # Step 1: Sample pages to understand structure
        self.sample_pages_for_analysis()
        
        # Step 2: Analyze structure
        structure_patterns = self.analyze_page_structure()
        
        # Step 3: Extract all content
        content_data = self.extract_all_text_with_coordinates()
        
        # Step 4: Find MCQ patterns
        raw_questions = self.find_mcq_patterns_advanced(content_data)
        
        # Step 5: Enhance with metadata
        self.all_questions = self.enhance_question_metadata(raw_questions)
        
        # Step 6: Generate statistics
        stats = self.generate_statistics()
        
        return self.all_questions, stats
    
    def generate_statistics(self):
        """Generate statistics"""
        if not self.all_questions:
            return {}
        
        stats = {
            'total_questions': len(self.all_questions),
            'chapters': {},
            'difficulties': {},
            'types': {},
            'strategies_used': {},
            'pages_covered': len(set(q['page'] for q in self.all_questions))
        }
        
        for q in self.all_questions:
            stats['chapters'][q['chapter']] = stats['chapters'].get(q['chapter'], 0) + 1
            stats['difficulties'][q['difficulty']] = stats['difficulties'].get(q['difficulty'], 0) + 1
            stats['types'][q['question_type']] = stats['types'].get(q['question_type'], 0) + 1
            stats['strategies_used'][q['strategy_used']] = stats['strategies_used'].get(q['strategy_used'], 0) + 1
        
        return stats
    
    def save_results(self, filename):
        """Save extraction results"""
        output_data = {
            'metadata': {
                'source_pdf': self.pdf_path,
                'extraction_date': datetime.now().isoformat(),
                'total_questions': len(self.all_questions),
                'method': 'advanced_multi_strategy'
            },
            'statistics': self.generate_statistics(),
            'questions': self.all_questions,
            'page_samples': self.page_samples
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to {filename}")

# Main execution
if __name__ == "__main__":
    pdf_path = r"C:\Users\HP\Downloads\RD-SHARMA CLASS 12TH VOLUME 2 MCQS (R.D.SHARMA) (Z-Library).pdf"
    
    extractor = AdvancedMCQExtractor(pdf_path)
    questions, statistics = extractor.run_advanced_extraction()
    
    # Save results
    extractor.save_results("rd_sharma_advanced_extraction.json")
    
    # Display results
    print(f"\n📊 ADVANCED EXTRACTION SUMMARY:")
    print("=" * 50)
    print(f"Total Questions Found: {statistics.get('total_questions', 0)}")
    print(f"Pages Covered: {statistics.get('pages_covered', 0)}")
    print(f"Strategies Used: {list(statistics.get('strategies_used', {}).keys())}")
    
    if questions:
        print(f"\n🎯 SAMPLE QUESTIONS:")
        for i, q in enumerate(questions[:3]):
            print(f"\nQ{i+1} (Page {q['page']}, Strategy: {q['strategy_used']}):")
            print(f"Text: {q['question_text'][:80]}...")
            print(f"Options: {len(q['options'])} choices")
            print(f"Chapter: {q['chapter']}")
    
    print(f"\n🎉 Advanced extraction complete!")
