import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
import os

class CHATBOT:
    def __init__(self, data_file='data.json'):
        self.data_file = data_file
        self.load_data()
        self.setup_vectorizer()
        
    def load_data(self):
        """Load the JSON data file"""
        with open(self.data_file, 'r') as file:
            self.data = json.load(file)
            
        # Create easy lookup dictionaries
        self.stories = {item['id']: item['story'] for item in self.data['data']}
        self.qa_pairs = {}
        for item in self.data['data']:
            self.qa_pairs[item['id']] = {
                q['turn_id']: {
                    'question': q['input_text'],
                    'answer': next(a['input_text'] for a in item['answers'] 
                                 if a['turn_id'] == q['turn_id'])
                }
                for q in item['questions']
            }
    
    def setup_vectorizer(self):
        """Initialize and fit the TF-IDF vectorizer"""
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2)
        )
        # Create story vectors
        stories_text = list(self.stories.values())
        self.story_vectors = self.vectorizer.fit_transform(stories_text)
        self.story_ids = list(self.stories.keys())
    
    def find_most_relevant_story(self, query, threshold=0.1):
        """Find the most relevant story based on the query"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.story_vectors).flatten()
        
        best_match_idx = similarities.argmax()
        if similarities[best_match_idx] < threshold:
            return None
        
        return self.story_ids[best_match_idx]
    
    def find_best_matching_question(self, story_id, user_question):
        """Find the most similar existing question in the story"""
        if story_id not in self.qa_pairs:
            return None
        
        story_questions = [qa['question'] 
                         for qa in self.qa_pairs[story_id].values()]
        
        if not story_questions:
            return None
            
        # Vectorize questions
        question_vectors = self.vectorizer.transform(story_questions)
        query_vector = self.vectorizer.transform([user_question])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, question_vectors).flatten()
        best_match_idx = similarities.argmax()
        
        if similarities[best_match_idx] < 0.3:  # Threshold for question matching
            return None
            
        # Get the turn_id for the best matching question
        turn_id = list(self.qa_pairs[story_id].keys())[best_match_idx]
        return turn_id
    
    def get_answer(self, story_id, turn_id):
        """Get the answer for a specific question"""
        return self.qa_pairs[story_id][turn_id]['answer']
    
    def save_data(self):
        """Save changes back to the JSON file"""
        # Create backup
        backup_file = f"data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_file, 'w') as f:
            json.dump(self.data, f, indent=2)
            
        # Save updated data
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2)
            
        # Reload data and rebuild vectors
        self.load_data()
        self.setup_vectorizer()
    
    def add_qa_pair(self, story_id, question, answer):
        """Add a new question-answer pair to a specific story"""
        if story_id not in self.stories:
            return False
            
        # Find the story in the original data structure
        for item in self.data['data']:
            if item['id'] == story_id:
                # Get next available turn_id
                max_turn_id = max([q['turn_id'] for q in item['questions']]) if item['questions'] else 0
                new_turn_id = max_turn_id + 1
                
                # Add new question
                item['questions'].append({
                    'input_text': question,
                    'turn_id': new_turn_id
                })
                
                # Add new answer
                item['answers'].append({
                    'input_text': answer,
                    'turn_id': new_turn_id,
                    'span_text': answer,
                    'span_start': 0,
                    'span_end': len(answer)
                })
                
                self.save_data()
                return True
        return False
    
    def edit_qa_pair(self, story_id, turn_id, new_question=None, new_answer=None):
        """Edit an existing question-answer pair"""
        if story_id not in self.stories:
            return False
            
        for item in self.data['data']:
            if item['id'] == story_id:
                modified = False
                
                if new_question:
                    for q in item['questions']:
                        if q['turn_id'] == turn_id:
                            q['input_text'] = new_question
                            modified = True
                
                if new_answer:
                    for a in item['answers']:
                        if a['turn_id'] == turn_id:
                            a['input_text'] = new_answer
                            a['span_text'] = new_answer
                            a['span_end'] = len(new_answer)
                            modified = True
                
                if modified:
                    self.save_data()
                    return True
        return False

def create_chatbot_interface():
    """Create an interactive command-line interface for the chatbot"""
    chatbot = CHATBOT('data.json')
    
    def print_help():
        print("\nAvailable commands:")
        print("1. Ask a question (just type your question)")
        print("2. 'add' - Add a new question-answer pair")
        print("3. 'edit' - Edit an existing question-answer pair")
        print("4. 'help' - Show this help message")
        print("5. 'exit' - Exit the chatbot")
    
    print("Welcome to CHATBOT! Type 'help' for available commands.")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
                
            elif user_input.lower() == 'help':
                print_help()
                
            elif user_input.lower() == 'add':
                # Get story context first
                context_q = input("Enter a question or context to find the relevant story: ")
                story_id = chatbot.find_most_relevant_story(context_q)
                
                if not story_id:
                    print("Could not find a relevant story. Please try again.")
                    continue
                    
                print("\nFound relevant story:")
                print(chatbot.stories[story_id][:200] + "...")
                
                if input("\nIs this the correct story? (yes/no): ").lower() != 'yes':
                    continue
                
                question = input("Enter the new question: ")
                answer = input("Enter the answer: ")
                
                if chatbot.add_qa_pair(story_id, question, answer):
                    print("Successfully added new Q&A pair!")
                else:
                    print("Failed to add Q&A pair.")
                    
            elif user_input.lower() == 'edit':
                # Get story context first
                context_q = input("Enter a question or context to find the relevant story: ")
                story_id = chatbot.find_most_relevant_story(context_q)
                
                if not story_id:
                    print("Could not find a relevant story. Please try again.")
                    continue
                
                print("\nExisting Q&A pairs for this story:")
                for turn_id, qa in chatbot.qa_pairs[story_id].items():
                    print(f"\nTurn {turn_id}:")
                    print(f"Q: {qa['question']}")
                    print(f"A: {qa['answer']}")
                
                turn_id = int(input("\nEnter the turn_id to edit: "))
                new_question = input("Enter new question (or press Enter to skip): ")
                new_answer = input("Enter new answer (or press Enter to skip): ")
                
                if chatbot.edit_qa_pair(
                    story_id, 
                    turn_id,
                    new_question if new_question else None,
                    new_answer if new_answer else None
                ):
                    print("Successfully updated Q&A pair!")
                else:
                    print("Failed to update Q&A pair.")
                    
            else:  # Treat as a question
                story_id = chatbot.find_most_relevant_story(user_input)
                if not story_id:
                    print("I couldn't find a relevant story to answer your question.")
                    continue
                    
                turn_id = chatbot.find_best_matching_question(story_id, user_input)
                if not turn_id:
                    print("I found a relevant story but couldn't match your question to any existing questions.")
                    print("\nRelevant story excerpt:")
                    print(chatbot.stories[story_id][:200] + "...")
                    continue
                    
                answer = chatbot.get_answer(story_id, turn_id)
                print(f"\nBot: {answer}")
                
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again or type 'help' for available commands.")

if __name__ == "__main__":
    create_chatbot_interface()