# kg_builder.py
import os
import json
import re
import time
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
import networkx as nx
from collections import defaultdict
import pandas as pd
from google.api_core import retry

load_dotenv()

# Optional OpenAI import for alternative medical LLM
try:
    import openai
    _has_openai = True
except Exception:
    openai = None
    _has_openai = False

def call_with_retry(func, max_retries=5, initial_wait=5):
    """
    Call a function with exponential backoff retry logic for rate limiting.
    Handles 429 (quota exceeded) errors gracefully.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit error
            if '429' in error_msg or 'quota' in error_msg.lower():
                if attempt < max_retries - 1:
                    # Extract wait time from error if available
                    wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
                    
                    # Try to parse wait time from error message
                    if 'retry in' in error_msg:
                        import re as regex
                        match = regex.search(r'retry in (\d+(?:\.\d+)?)', error_msg)
                        if match:
                            wait_time = float(match.group(1)) + 2
                    
                    print(f"⚠️ Rate limited. Waiting {wait_time:.1f}s before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Max retries exceeded. Skipping this request.")
                    return None
            else:
                # For non-rate-limit errors, raise immediately
                raise
    
    return None

class MedicalKnowledgeGraphBuilder:
    """
    Comprehensive Knowledge Graph Builder for Medical First Aid ChatBot
    Based on Red Cross Society Manual
    """
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=self.api_key)
        
        # Initialize Gemini models (using latest available models)
        self.extraction_model = genai.GenerativeModel('gemini-2.5-pro')
        self.relationship_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # OpenAI fallback: if OPENAI_API_KEY present, prefer OpenAI for extraction
        self.openai_key = os.getenv('OPENAI_API_KEY')
        # For now, default to Gemini; set USE_OPENAI=1 env var to enable OpenAI
        if os.getenv('USE_OPENAI') == '1' and self.openai_key and _has_openai:
            openai.api_key = self.openai_key
            self.use_openai = True
            print("✅ Using OpenAI (gpt-4o-mini) for medical extraction")
        else:
            self.use_openai = False
            print("✅ Using Gemini (gemini-2.5-pro/flash) for medical extraction")

        # Knowledge Graph structure
        self.graph = nx.MultiDiGraph()
        self.entities = defaultdict(dict)
        self.relationships = []
        
        # Medical ontology structure
        self.ontology = {
            'conditions': [],
            'symptoms': [],
            'treatments': [],
            'procedures': [],
            'anatomical_structures': [],
            'medications': [],
            'equipment': [],
            'emergency_types': []
        }

    def _call_llm(self, prompt: str, model_type: str = 'extraction', expect: str = 'object'):
        """
        Call the preferred LLM (OpenAI if configured, otherwise Gemini via genai).
        Returns parsed JSON (object or array) or raises on parse error.
        """
        # Use OpenAI if available and configured
        if self.use_openai:
            try:
                # Send prompt as a single user message (modern OpenAI API >= 1.0.0)
                client = openai.OpenAI(api_key=self.openai_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1200
                )
                response_text = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️ OpenAI error: {e}. Falling back to Gemini...")
                # Fallback to Gemini on OpenAI error
                if model_type == 'relationships':
                    response = self.relationship_model.generate_content(prompt)
                else:
                    response = self.extraction_model.generate_content(prompt)
                response_text = response.text.strip()
        else:
            # Choose Gemini model
            if model_type == 'relationships':
                response = self.relationship_model.generate_content(prompt)
            else:
                response = self.extraction_model.generate_content(prompt)
            response_text = response.text.strip()

        # Clean markdown code fences
        response_text = re.sub(r'```json\n?', '', response_text)
        response_text = re.sub(r'```\n?', '', response_text)

        # Extract JSON
        if expect == 'object':
            start = response_text.find('{')
            end = response_text.rfind('}')
        else:
            start = response_text.find('[')
            end = response_text.rfind(']')

        if start == -1 or end == -1:
            raise ValueError('No JSON found in LLM response')

        json_str = response_text[start:end+1]
        return json.loads(json_str)
        
        # Knowledge Graph structure
        self.graph = nx.MultiDiGraph()
        self.entities = defaultdict(dict)
        self.relationships = []
        
        # Medical ontology structure
        self.ontology = {
            'conditions': [],
            'symptoms': [],
            'treatments': [],
            'procedures': [],
            'anatomical_structures': [],
            'medications': [],
            'equipment': [],
            'emergency_types': []
        }
        
    def extract_entities_with_gemini(self, text_chunk: str) -> Dict[str, List[str]]:
        """
        Extract medical entities using Gemini API with retry logic
        """
        prompt = f"""
        You are a medical knowledge extraction expert. Analyze the following text from the Red Cross First Aid Manual 
        and extract ALL relevant medical entities in the following categories:
        
        1. **Medical Conditions/Emergencies**: (e.g., heart attack, stroke, burns, fractures, poisoning)
        2. **Symptoms**: (e.g., chest pain, difficulty breathing, bleeding, unconsciousness)
        3. **Treatments/Interventions**: (e.g., CPR, bandaging, cooling burns, recovery position)
        4. **Anatomical Structures**: (e.g., heart, lungs, skull, spine, skin layers)
        5. **Medications**: (e.g., paracetamol, aspirin, ORS, insulin)
        6. **Medical Equipment**: (e.g., stretcher, bandages, thermometer, splints)
        7. **Procedures**: (e.g., resuscitation, triage, wound cleaning, immobilization)
        8. **Emergency Types**: (e.g., drowning, choking, snake bite, heat stroke)
        
        Text to analyze:
        {text_chunk}
        
        Return ONLY a valid JSON object with these exact keys: conditions, symptoms, treatments, 
        anatomical_structures, medications, equipment, procedures, emergency_types
        
        Each value should be a list of strings. Extract as many relevant entities as possible.
        Be comprehensive and precise.
        """
        
        try:
            result = call_with_retry(lambda: self._call_llm(prompt, model_type='extraction', expect='object'))
            return result if result else {
                'conditions': [], 'symptoms': [], 'treatments': [],
                'anatomical_structures': [], 'medications': [], 'equipment': [],
                'procedures': [], 'emergency_types': []
            }
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return {
                'conditions': [], 'symptoms': [], 'treatments': [],
                'anatomical_structures': [], 'medications': [], 'equipment': [],
                'procedures': [], 'emergency_types': []
            }
    
    def extract_relationships_with_gemini(self, text_chunk: str, entities: Dict[str, List[str]]) -> List[Dict]:
        """
        Extract relationships between entities using Gemini with retry logic
        """
        prompt = f"""
        You are a medical knowledge graph expert. Given the following text and extracted entities, 
        identify ALL meaningful relationships between them.
        
        Text:
        {text_chunk}
        
        Entities:
        {json.dumps(entities, indent=2)}
        
        Extract relationships in the following format:
        - Condition -> HAS_SYMPTOM -> Symptom
        - Condition -> TREATED_BY -> Treatment
        - Condition -> AFFECTS -> Anatomical Structure
        - Treatment -> REQUIRES -> Equipment
        - Treatment -> USES -> Medication
        - Symptom -> INDICATES -> Condition
        - Procedure -> PART_OF -> Treatment
        - Emergency -> REQUIRES -> Procedure
        - Condition -> CAUSES -> Symptom
        - Treatment -> PREVENTS -> Condition
        
        Return a JSON array of relationships with this structure:
        [
            {{
                "source": "entity1",
                "relationship": "RELATIONSHIP_TYPE",
                "target": "entity2",
                "context": "brief explanation from text"
            }}
        ]
        
        Be thorough and extract ALL relevant relationships mentioned in the text.
        """
        
        try:
            result = call_with_retry(lambda: self._call_llm(prompt, model_type='relationships', expect='array'))
            return result if result else []
        except Exception as e:
            print(f"Error extracting relationships: {e}")
            return []
    
    def extract_step_by_step_procedures(self, text_chunk: str) -> List[Dict]:
        """
        Extract step-by-step medical procedures using Gemini with retry logic
        """
        prompt = f"""
        Extract ALL step-by-step medical procedures from this text.
        For each procedure, identify:
        1. Procedure name
        2. When to use it (condition/emergency)
        3. Ordered steps
        4. Required equipment
        5. Safety warnings
        6. When to seek medical help
        
        Text:
        {text_chunk}
        
        Return as JSON array:
        [
            {{
                "procedure_name": "name",
                "condition": "when to use",
                "steps": ["step 1", "step 2", ...],
                "equipment": ["item1", "item2"],
                "warnings": ["warning1", ...],
                "seek_help_criteria": ["criterion1", ...]
            }}
        ]
        """
        
        try:
            result = call_with_retry(lambda: self._call_llm(prompt, model_type='extraction', expect='array'))
            return result if result else []
        except Exception as e:
            print(f"Error extracting procedures: {e}")
            return []
    
    def build_knowledge_graph_from_manual(self, manual_text: str, chunk_size: int = 3000):
        """
        Build comprehensive knowledge graph from the entire manual
        """
        print("Building Knowledge Graph from Red Cross Manual...")
        
        # Split manual into chunks
        chunks = self._split_text(manual_text, chunk_size)
        print(f"Processing {len(chunks)} text chunks...")
        
        all_procedures = []
        
        for idx, chunk in enumerate(chunks):
            print(f"Processing chunk {idx + 1}/{len(chunks)}...")
            
            # Extract entities
            entities = self.extract_entities_with_gemini(chunk)
            
            # Add entities to ontology
            for category, entity_list in entities.items():
                if category in self.ontology:
                    self.ontology[category].extend(entity_list)
            
            # Extract relationships
            relationships = self.extract_relationships_with_gemini(chunk, entities)
            self.relationships.extend(relationships)
            
            # Extract procedures
            procedures = self.extract_step_by_step_procedures(chunk)
            all_procedures.extend(procedures)
            
            # Add to graph
            self._add_entities_to_graph(entities)
            self._add_relationships_to_graph(relationships)
        
        # Remove duplicates from ontology
        for category in self.ontology:
            self.ontology[category] = list(set(self.ontology[category]))
        
        # Add procedures as special nodes
        self._add_procedures_to_graph(all_procedures)
        
        print(f"\nKnowledge Graph Statistics:")
        print(f"Total Nodes: {self.graph.number_of_nodes()}")
        print(f"Total Edges: {self.graph.number_of_edges()}")
        print(f"Conditions: {len(self.ontology['conditions'])}")
        print(f"Symptoms: {len(self.ontology['symptoms'])}")
        print(f"Treatments: {len(self.ontology['treatments'])}")
        print(f"Procedures: {len(all_procedures)}")
        
        return all_procedures
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into manageable chunks
        """
        # Split by sections (chapters)
        sections = re.split(r'\n(?=[A-Z]\.\d+\s+[A-Z])', text)
        
        chunks = []
        current_chunk = ""
        
        for section in sections:
            if len(current_chunk) + len(section) < chunk_size:
                current_chunk += section
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = section
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _add_entities_to_graph(self, entities: Dict[str, List[str]]):
        """
        Add entities as nodes to the graph
        """
        for category, entity_list in entities.items():
            for entity in entity_list:
                entity = entity.strip()
                if entity and len(entity) > 2:
                    self.graph.add_node(
                        entity,
                        type=category,
                        category=category
                    )
    
    def _add_relationships_to_graph(self, relationships: List[Dict]):
        """
        Add relationships as edges to the graph
        """
        for rel in relationships:
            source = rel.get('source', '').strip()
            target = rel.get('target', '').strip()
            rel_type = rel.get('relationship', 'RELATED_TO')
            context = rel.get('context', '')
            
            if source and target and len(source) > 2 and len(target) > 2:
                self.graph.add_edge(
                    source,
                    target,
                    relationship=rel_type,
                    context=context
                )
    
    def _add_procedures_to_graph(self, procedures: List[Dict]):
        """
        Add procedures as special nodes with detailed information
        """
        for proc in procedures:
            proc_name = proc.get('procedure_name', '').strip()
            if not proc_name:
                continue
            
            self.graph.add_node(
                proc_name,
                type='procedure',
                category='procedures',
                steps=proc.get('steps', []),
                equipment=proc.get('equipment', []),
                warnings=proc.get('warnings', []),
                seek_help=proc.get('seek_help_criteria', [])
            )
            
            # Link procedure to condition
            condition = proc.get('condition', '').strip()
            if condition:
                self.graph.add_edge(
                    condition,
                    proc_name,
                    relationship='REQUIRES_PROCEDURE'
                )
    
    def export_to_neo4j(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Export knowledge graph to Neo4j database
        """
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        with driver.session() as session:
            # Create nodes
            for node, data in self.graph.nodes(data=True):
                category = data.get('category', 'unknown')
                node_type = data.get('type', 'entity')
                
                cypher = f"""
                MERGE (n:{category} {{name: $name}})
                SET n.type = $type
                """
                
                # Add procedure-specific properties
                if node_type == 'procedure':
                    cypher += """
                    SET n.steps = $steps,
                        n.equipment = $equipment,
                        n.warnings = $warnings,
                        n.seek_help = $seek_help
                    """
                    session.run(cypher, 
                               name=node, 
                               type=node_type,
                               steps=data.get('steps', []),
                               equipment=data.get('equipment', []),
                               warnings=data.get('warnings', []),
                               seek_help=data.get('seek_help', []))
                else:
                    session.run(cypher, name=node, type=node_type)
            
            # Create relationships
            for source, target, data in self.graph.edges(data=True):
                rel_type = data.get('relationship', 'RELATED_TO')
                context = data.get('context', '')
                
                cypher = f"""
                MATCH (a {{name: $source}})
                MATCH (b {{name: $target}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r.context = $context
                """
                session.run(cypher, source=source, target=target, context=context)
        
        driver.close()
        print("Knowledge Graph exported to Neo4j successfully!")
    
    def export_to_json(self, output_file: str = 'medical_knowledge_graph.json'):
        """
        Export knowledge graph to JSON format
        """
        kg_data = {
            'ontology': self.ontology,
            'nodes': [
                {
                    'id': node,
                    'attributes': data
                }
                for node, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    'relationship': data.get('relationship', 'RELATED_TO'),
                    'context': data.get('context', '')
                }
                for source, target, data in self.graph.edges(data=True)
            ],
            'statistics': {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'node_types': {
                    category: len(entities)
                    for category, entities in self.ontology.items()
                }
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(kg_data, f, indent=2, ensure_ascii=False)
        
        print(f"Knowledge Graph exported to {output_file}")
        
        return kg_data
    
    def create_summary_report(self) -> str:
        """
        Create a comprehensive summary report of the knowledge graph
        """
        prompt = f"""
        Create a comprehensive summary report of this medical knowledge graph:
        
        Statistics:
        - Total Entities: {self.graph.number_of_nodes()}
        - Total Relationships: {self.graph.number_of_edges()}
        - Conditions: {len(self.ontology['conditions'])}
        - Symptoms: {len(self.ontology['symptoms'])}
        - Treatments: {len(self.ontology['treatments'])}
        - Procedures: {len(self.ontology['procedures'])}
        
        Key Entities:
        {json.dumps(self.ontology, indent=2)[:2000]}
        
        Provide:
        1. Overview of the knowledge graph coverage
        2. Main medical domains covered
        3. Most important relationships
        4. Potential use cases for this knowledge graph
        5. Quality assessment
        """
        
        try:
            result = call_with_retry(lambda: self.extraction_model.generate_content(prompt).text)
            return result if result else "Summary generation skipped due to quota limits."
        except Exception as e:
            print(f"⚠️ Could not generate summary: {e}")
            return "Summary generation skipped due to quota limits."


# chatbot.py
class MedicalChatBot:
    """
    Medical ChatBot using the Knowledge Graph
    """
    
    def __init__(self, kg_data: Dict):
        self.api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=self.api_key)
        
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        self.kg_data = kg_data
        self.chat_history = []
        
        # Create knowledge context
        self.knowledge_context = self._create_knowledge_context()
    
    def _create_knowledge_context(self) -> str:
        """
        Create a comprehensive knowledge context from KG
        """
        context = f"""
        You are a medical first aid assistant with access to comprehensive knowledge from the 
        Indian Red Cross Society First Aid Manual.
        
        Your knowledge base includes:
        - {len(self.kg_data['ontology']['conditions'])} medical conditions and emergencies
        - {len(self.kg_data['ontology']['symptoms'])} symptoms
        - {len(self.kg_data['ontology']['treatments'])} treatments
        - {len(self.kg_data['ontology']['procedures'])} medical procedures
        
        Key Medical Domains:
        {json.dumps(self.kg_data['ontology'], indent=2)[:3000]}
        
        Important: 
        - Always prioritize safety
        - Recommend seeking professional medical help when necessary
        - Provide step-by-step guidance for emergencies
        - Consider the Indian context and resource availability
        """
        
        return context
    
    def query(self, user_question: str) -> str:
        """
        Process user query using KG-enhanced prompting
        """
        # Find relevant entities
        relevant_info = self._find_relevant_knowledge(user_question)
        
        prompt = f"""
        {self.knowledge_context}
        
        Relevant Knowledge:
        {relevant_info}
        
        User Question: {user_question}
        
        Chat History:
        {self._format_chat_history()}
        
        Provide a clear, accurate, and actionable response based on the Red Cross First Aid Manual.
        Include:
        1. Immediate actions to take
        2. Step-by-step procedures if applicable
        3. Warning signs
        4. When to seek medical help
        
        Be empathetic and clear. If you're unsure, say so and recommend professional help.
        """
        
        response = self.model.generate_content(prompt)
        answer = response.text
        
        # Update chat history
        self.chat_history.append({
            'question': user_question,
            'answer': answer
        })
        
        return answer
    
    def _find_relevant_knowledge(self, query: str) -> str:
        """
        Find relevant nodes and relationships from KG
        """
        query_lower = query.lower()
        relevant_nodes = []
        relevant_edges = []
        
        # Find matching nodes
        for node in self.kg_data['nodes']:
            node_name = node['id'].lower()
            if any(word in node_name for word in query_lower.split()):
                relevant_nodes.append(node)
        
        # Find matching edges
        for edge in self.kg_data['edges']:
            source = edge['source'].lower()
            target = edge['target'].lower()
            if any(word in source or word in target for word in query_lower.split()):
                relevant_edges.append(edge)
        
        relevant_info = {
            'matched_entities': relevant_nodes[:10],
            'matched_relationships': relevant_edges[:10]
        }
        
        return json.dumps(relevant_info, indent=2)
    
    def _format_chat_history(self) -> str:
        """
        Format recent chat history
        """
        if not self.chat_history:
            return "No previous conversation."
        
        recent = self.chat_history[-3:]  # Last 3 exchanges
        formatted = []
        for exchange in recent:
            formatted.append(f"Q: {exchange['question']}\nA: {exchange['answer'][:200]}...")
        
        return "\n\n".join(formatted)


# main.py
def main():
    """
    Main execution function
    """
    print("=== Medical Knowledge Graph Builder ===\n")
    
    # Show which LLM will be used
    use_openai_mode = os.getenv('USE_OPENAI') == '1'
    if use_openai_mode:
        print("🔧 LLM Configuration: OpenAI (gpt-4o-mini) - Premium mode")
    else:
        print("🔧 LLM Configuration: Gemini (gemini-2.5-pro/flash) - Free tier mode")
    print()
    
    # Read the manual from PDF
    manual_file = 'RedCrossSocietyManual.pdf'
    
    # Extract text from PDF
    print(f"📖 Extracting text from {manual_file}...")
    import pdfplumber
    
    manual_text = ""
    try:
        with pdfplumber.open(manual_file) as pdf:
            for page in pdf.pages:
                manual_text += page.extract_text() or ""
                manual_text += "\n"
        print(f"✅ Extracted {len(manual_text)} characters from PDF\n")
    except FileNotFoundError:
        print(f"❌ Error: {manual_file} not found!")
        print("Make sure the PDF is in the same directory as this script.")
        return
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return
    
    # Initialize KG Builder
    kg_builder = MedicalKnowledgeGraphBuilder()
    
    # Build Knowledge Graph
    print("\n1. Building Knowledge Graph...")
    procedures = kg_builder.build_knowledge_graph_from_manual(manual_text)
    
    # Export to JSON
    print("\n2. Exporting to JSON...")
    kg_data = kg_builder.export_to_json('medical_kg.json')
    
    # Export to Neo4j (optional)
    if os.getenv('NEO4J_URI'):
        print("\n3. Exporting to Neo4j...")
        kg_builder.export_to_neo4j(
            os.getenv('NEO4J_URI'),
            os.getenv('NEO4J_USER'),
            os.getenv('NEO4J_PASSWORD')
        )
    
    # Generate summary report
    print("\n4. Generating Summary Report...")
    report = kg_builder.create_summary_report()
    print("\n" + report)
    
    # Initialize ChatBot
    print("\n5. Initializing ChatBot...")
    chatbot = MedicalChatBot(kg_data)
    
    # Interactive session
    print("\n=== Medical ChatBot Ready ===")
    print("Ask questions about first aid, emergencies, or type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Stay safe! Goodbye.")
            break
        
        if not user_input:
            continue
        
        print("\nChatBot: ", end="")
        response = chatbot.query(user_input)
        print(response)
        print("\n" + "-"*80 + "\n")


if __name__ == "__main__":
    main()