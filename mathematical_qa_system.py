"""
Complete Mathematical Question Answering System
Uses the trained RAG models to answer RD Sharma-style MCQs with high accuracy
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import re
from datetime import datetime
import os

class MathematicalQASystem:
    def __init__(self, models_dir="rag_models"):
        self.models_dir = models_dir
        self.models = {}
        self.vectorizer = None
        self.scaler = None
        self.is_loaded = False
        self.question_history = []
        
    def load_trained_models(self):
        """Load all trained models and components"""
        print("🤖 Loading trained mathematical models...")
        
        try:
            # Load models
            model_files = {
                'RandomForest': 'randomforest_model.pkl',
                'GradientBoosting': 'gradientboosting_model.pkl', 
                'NeuralNetwork': 'neuralnetwork_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                filepath = os.path.join(self.models_dir, filename)
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    print(f"✅ Loaded {model_name} model")
            
            # Load vectorizer
            vectorizer_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                print("✅ Loaded TF-IDF vectorizer")
            
            # Load metadata
            metadata_path = os.path.join(self.models_dir, 'training_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ Loaded training metadata (Best model: {self.metadata.get('best_model', 'Unknown')})")
            
            self.is_loaded = len(self.models) > 0 and self.vectorizer is not None
            
            if self.is_loaded:
                print(f"🎯 System ready! Loaded {len(self.models)} models")
            else:
                print("❌ Failed to load required models")
                
            return self.is_loaded
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def clean_mathematical_text(self, text):
        """Clean and normalize mathematical text (same as training)"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize mathematical symbols
        replacements = {
            'dx': ' dx ',
            'dy': ' dy ',
            'sin': ' sin ',
            'cos': ' cos ',
            'tan': ' tan ',
            'log': ' log ',
            'ln': ' ln ',
            'lim': ' lim ',
            '∫': ' integral ',
            '∑': ' sum ',
            '∏': ' product ',
            '√': ' sqrt ',
            'π': ' pi ',
            '∞': ' infinity ',
            '±': ' plus_minus ',
            '≤': ' less_equal ',
            '≥': ' greater_equal ',
            '≠': ' not_equal ',
            '→': ' approaches ',
            '∈': ' belongs_to ',
            '∪': ' union ',
            '∩': ' intersection ',
        }
        
        for symbol, replacement in replacements.items():
            text = text.replace(symbol, replacement)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, question_text, options):
        """Extract features for a new question (same as training)"""
        features = {}
        
        # Basic text features
        features['question_length'] = len(question_text)
        features['question_word_count'] = len(question_text.split())
        features['avg_option_length'] = np.mean([len(opt['text']) for opt in options])
        features['num_options'] = len(options)
        
        # Mathematical content features
        math_keywords = [
            'integral', 'derivative', 'limit', 'sum', 'product', 'sin', 'cos', 'tan',
            'matrix', 'determinant', 'probability', 'function', 'equation', 'solve'
        ]
        
        features['math_keyword_count'] = sum(1 for keyword in math_keywords 
                                           if keyword in question_text.lower())
        
        # Question type indicators
        calc_words = ['find', 'calculate', 'compute', 'evaluate', 'determine']
        concept_words = ['which', 'what', 'identify', 'select', 'choose']
        proof_words = ['prove', 'show', 'verify', 'demonstrate']
        
        features['is_calculation'] = int(any(word in question_text.lower() for word in calc_words))
        features['is_conceptual'] = int(any(word in question_text.lower() for word in concept_words))
        features['is_proof'] = int(any(word in question_text.lower() for word in proof_words))
        
        # Option similarity features
        option_texts = [opt['text'] for opt in options]
        if len(option_texts) > 1:
            similarities = []
            for i in range(len(option_texts)):
                for j in range(i+1, len(option_texts)):
                    sim = self.text_similarity(option_texts[i], option_texts[j])
                    similarities.append(sim)
            features['avg_option_similarity'] = np.mean(similarities) if similarities else 0
            features['max_option_similarity'] = np.max(similarities) if similarities else 0
        else:
            features['avg_option_similarity'] = 0
            features['max_option_similarity'] = 0
        
        # Chapter-specific features (simplified)
        chapters = ['Integration', 'Differentiation', 'Limits', 'Trigonometry', 
                   'Probability', 'Vector', 'Matrix', 'Statistics', 'Geometry', 'General']
        
        detected_chapter = self.detect_chapter(question_text)
        for ch in chapters:
            features[f'chapter_{ch.lower()}'] = int(detected_chapter == ch)
        
        # Complexity indicators
        features['has_fractions'] = int(bool(re.search(r'\d+/\d+', question_text)))
        features['has_powers'] = int(bool(re.search(r'\^\d+', question_text)))
        features['has_parentheses'] = int('(' in question_text and ')' in question_text)
        features['has_variables'] = int(bool(re.search(r'\b[a-z]\b', question_text)))
        
        return features
    
    def text_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1
        if not words1 or not words2:
            return 0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0
    
    def detect_chapter(self, question_text):
        """Detect the mathematical chapter/topic"""
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
    
    def answer_question(self, question_text, options):
        """Answer a mathematical MCQ question"""
        if not self.is_loaded:
            return {
                'error': 'Models not loaded. Please run load_trained_models() first.',
                'success': False
            }
        
        print(f"🔍 Analyzing question: {question_text[:60]}...")
        
        try:
            # Clean and process the question
            cleaned_question = self.clean_mathematical_text(question_text)
            
            # Process options
            processed_options = []
            for i, option in enumerate(options):
                if isinstance(option, str):
                    # If options are just strings, assign letters
                    processed_options.append({
                        'letter': chr(65 + i),  # A, B, C, D
                        'text': self.clean_mathematical_text(option)
                    })
                else:
                    # If options are dictionaries
                    processed_options.append({
                        'letter': option.get('letter', chr(65 + i)),
                        'text': self.clean_mathematical_text(option.get('text', str(option)))
                    })
            
            # Extract features
            features = self.extract_features(cleaned_question, processed_options)
            
            # Prepare prediction data
            predictions = {}
            confidences = {}
            
            # For each option, get prediction
            for opt in processed_options:
                # Create combined text for TF-IDF
                combined_text = f"{cleaned_question} {opt['text']}"
                
                # Create feature vector
                feature_vector = list(features.values()) + [
                    len(opt['text']),
                    len(opt['text'].split()),
                    int('formula' in opt['text'].lower() or any(symbol in opt['text'] 
                        for symbol in ['∫', '∑', 'dx', 'dy', 'sin', 'cos']))
                ]
                
                # Get TF-IDF features
                X_tfidf = self.vectorizer.transform([combined_text])
                
                # Combine features (same as training)
                X_combined = np.hstack([
                    X_tfidf.toarray(),
                    np.array(feature_vector).reshape(1, -1)
                ])
                
                # Get predictions from all models
                model_preds = []
                model_probs = []
                
                for model_name, model in self.models.items():
                    pred_proba = model.predict_proba(X_combined)[0]
                    prob_correct = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
                    
                    model_preds.append(model.predict(X_combined)[0])
                    model_probs.append(prob_correct)
                
                # Ensemble prediction (average probabilities)
                avg_prob = np.mean(model_probs)
                predictions[opt['letter']] = avg_prob
                confidences[opt['letter']] = avg_prob
            
            # Find the best answer
            best_answer = max(predictions.keys(), key=lambda k: predictions[k])
            best_confidence = predictions[best_answer]
            
            # Create detailed analysis
            analysis = {
                'question': question_text,
                'cleaned_question': cleaned_question,
                'detected_chapter': self.detect_chapter(question_text),
                'question_type': self.classify_question_type(question_text),
                'predictions': predictions,
                'best_answer': best_answer,
                'confidence': float(best_confidence),
                'confidence_percentage': f"{best_confidence * 100:.1f}%",
                'all_options_analysis': []
            }
            
            # Add detailed option analysis
            for opt in processed_options:
                opt_analysis = {
                    'letter': opt['letter'],
                    'text': opt['text'],
                    'probability': float(predictions[opt['letter']]),
                    'is_predicted_correct': opt['letter'] == best_answer
                }
                analysis['all_options_analysis'].append(opt_analysis)
            
            # Sort options by probability
            analysis['all_options_analysis'].sort(key=lambda x: x['probability'], reverse=True)
            
            # Add to history
            self.question_history.append({
                'timestamp': datetime.now().isoformat(),
                'question': question_text,
                'answer': best_answer,
                'confidence': best_confidence
            })
            
            print(f"✅ Answer: {best_answer} (Confidence: {best_confidence:.1%})")
            
            return {
                'success': True,
                'answer': best_answer,
                'confidence': best_confidence,
                'analysis': analysis
            }
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def classify_question_type(self, question_text):
        """Classify the type of mathematical question"""
        text_lower = question_text.lower()
        
        if any(word in text_lower for word in ['find', 'calculate', 'compute', 'evaluate']):
            return 'Calculation'
        elif any(word in text_lower for word in ['prove', 'show that', 'verify']):
            return 'Proof'
        elif any(word in text_lower for word in ['which', 'what is', 'identify']):
            return 'Conceptual'
        elif any(word in text_lower for word in ['graph', 'plot', 'sketch']):
            return 'Graphical'
        else:
            return 'Application'
    
    def batch_answer_questions(self, questions_list):
        """Answer multiple questions at once"""
        if not self.is_loaded:
            print("❌ Models not loaded")
            return []
        
        results = []
        print(f"📝 Processing {len(questions_list)} questions...")
        
        for i, q in enumerate(questions_list):
            print(f"\nQuestion {i+1}/{len(questions_list)}:")
            result = self.answer_question(q['question'], q['options'])
            results.append(result)
        
        return results
    
    def get_performance_summary(self):
        """Get performance summary of the system"""
        if not self.question_history:
            return "No questions processed yet."
        
        total_questions = len(self.question_history)
        avg_confidence = np.mean([q['confidence'] for q in self.question_history])
        
        confidence_distribution = {
            'High (>80%)': sum(1 for q in self.question_history if q['confidence'] > 0.8),
            'Medium (50-80%)': sum(1 for q in self.question_history if 0.5 <= q['confidence'] <= 0.8),
            'Low (<50%)': sum(1 for q in self.question_history if q['confidence'] < 0.5)
        }
        
        return {
            'total_questions_answered': total_questions,
            'average_confidence': f"{avg_confidence:.1%}",
            'confidence_distribution': confidence_distribution,
            'most_recent_questions': self.question_history[-5:] if len(self.question_history) > 5 else self.question_history
        }
    
    def interactive_mode(self):
        """Run interactive question-answering mode"""
        print("\n🎓 MATHEMATICAL QA SYSTEM - INTERACTIVE MODE")
        print("=" * 50)
        print("Enter your mathematical questions and I'll provide answers!")
        print("Type 'quit' to exit, 'summary' for performance summary")
        print("=" * 50)
        
        while True:
            try:
                print("\n📝 Enter your question:")
                question = input().strip()
                
                if question.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                if question.lower() == 'summary':
                    summary = self.get_performance_summary()
                    print("\n📊 PERFORMANCE SUMMARY:")
                    print(json.dumps(summary, indent=2))
                    continue
                
                if not question:
                    print("Please enter a question.")
                    continue
                
                print("\n📋 Enter options (one per line, press Enter twice when done):")
                options = []
                while True:
                    option = input().strip()
                    if not option:
                        break
                    options.append(option)
                
                if len(options) < 2:
                    print("Please provide at least 2 options.")
                    continue
                
                # Answer the question
                result = self.answer_question(question, options)
                
                if result['success']:
                    print(f"\n🎯 ANSWER: {result['answer']}")
                    print(f"📊 Confidence: {result['confidence']:.1%}")
                    print(f"📚 Chapter: {result['analysis']['detected_chapter']}")
                    print(f"🔍 Question Type: {result['analysis']['question_type']}")
                    
                    print(f"\n📋 All Options Analysis:")
                    for opt in result['analysis']['all_options_analysis']:
                        marker = "✅" if opt['is_predicted_correct'] else "  "
                        print(f"{marker} {opt['letter']}: {opt['probability']:.1%} confidence")
                else:
                    print(f"❌ Error: {result['error']}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

# Main execution and demonstration
if __name__ == "__main__":
    # Initialize the system
    qa_system = MathematicalQASystem()
    
    # Load trained models
    if qa_system.load_trained_models():
        print("\n🎉 Mathematical QA System is ready!")
        
        # Demo question
        demo_question = {
            'question': "Find the value of the integral ∫ sin(x) dx from 0 to π",
            'options': [
                "A. 0",
                "B. 2", 
                "C. π",
                "D. -2"
            ]
        }
        
        print("\n🔍 DEMO: Answering a sample question...")
        result = qa_system.answer_question(demo_question['question'], demo_question['options'])
        
        if result['success']:
            print(f"\n🎯 DEMO RESULT:")
            print(f"Question: {demo_question['question']}")
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.1%}")
            
        # Start interactive mode
        response = input("\n🎮 Would you like to try interactive mode? (y/n): ")
        if response.lower().startswith('y'):
            qa_system.interactive_mode()
        else:
            print("✅ System ready for API usage!")
    else:
        print("❌ Failed to load models. Please check the model files.")
