import json

# Load the extracted questions
with open('rd_sharma_questions_complete.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("📊 RD SHARMA EXTRACTION ANALYSIS")
print("=" * 50)

questions = data['questions']
print(f"Total Questions: {len(questions)}")

# Find questions with good options
good_questions = [q for q in questions if q['options'] and len(q['options']) >= 3]
print(f"Questions with 3+ options: {len(good_questions)}")

# Show sample good questions
print("\n🎯 SAMPLE WELL-FORMATTED QUESTIONS:")
print("-" * 40)

count = 0
for q in questions:
    if q['options'] and len(q['options']) >= 3 and count < 5:
        print(f"\nQuestion {q['number']} (Page {q['page']}):")
        print(f"Text: {q['question_text'][:150]}...")
        print("Options:")
        for opt in q['options']:
            print(f"  {opt['letter']}. {opt['text']}")
        print(f"Chapter: {q['chapter']}, Type: {q['question_type']}")
        count += 1

# Find questions with formulas
formula_questions = [q for q in questions if q['has_formula']]
print(f"\n🔬 Questions with mathematical formulas: {len(formula_questions)}")

# Show distribution by page
pages_with_questions = {}
for q in questions:
    pages_with_questions[q['page']] = pages_with_questions.get(q['page'], 0) + 1

print(f"\n📄 Page distribution (top 10 pages with most questions):")
sorted_pages = sorted(pages_with_questions.items(), key=lambda x: x[1], reverse=True)
for page, count in sorted_pages[:10]:
    print(f"  Page {page}: {count} questions")

# Check question quality
proper_questions = []
for q in questions:
    # Check if it looks like a proper question
    text = q['question_text'].lower()
    if (any(word in text for word in ['find', 'what', 'which', 'calculate', 'prove', 'show', 'determine', 'if', 'when'])
        and len(q['question_text'].split()) > 5
        and not q['question_text'].isupper()):  # Not all caps (likely headers)
        proper_questions.append(q)

print(f"\n✅ Proper mathematical questions: {len(proper_questions)}")

# Show some proper questions
print("\n🔍 SAMPLE PROPER MATHEMATICAL QUESTIONS:")
print("-" * 45)
for i, q in enumerate(proper_questions[:10]):
    print(f"{i+1}. {q['question_text'][:120]}...")
    if q['options']:
        print(f"   Options: {len(q['options'])} choices")
    print()
