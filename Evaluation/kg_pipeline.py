"""
kg_pipeline.py — Improved Knowledge Graph Pipeline
===================================================
Improvements over v1:
  1. Synonym expansion — maps query words to KG vocabulary
     (e.g. "choking" → "airway obstruction", "collapsed" → "unconsciousness")
  2. Multi-strategy retrieval — exact match + synonym match + ontology fallback
  3. Edge-context scoring — prioritises REQUIRES_PROCEDURE / TREATED_BY edges
  4. Traversal logging — records every step for visualisation
  5. Better answer formatting — clean prose output, not raw structured text
"""

import os
import json
import time
import requests
from typing import Optional
from collections import defaultdict


# ─── Synonym / vocabulary map ─────────────────────────────────────────────────
# Maps common query words → actual KG node ID substrings
# Built by inspecting all 2,469 node IDs in medical_kg.json

SYNONYMS = {
    # Unconsciousness / collapse
    "collapsed":      ["unconsciousness", "unconscious"],
    "collapse":       ["unconsciousness", "unconscious"],
    "unconscious":    ["unconsciousness"],
    "unresponsive":   ["unconsciousness", "loss of consciousness"],
    "fainted":        ["unconsciousness", "fainting"],
    "faint":          ["fainting", "unconsciousness"],
    "passed out":     ["unconsciousness"],

    # Breathing
    "not breathing":  ["not breathing", "no breathing", "respiratory arrest"],
    "stopped breathing": ["respiratory arrest", "not breathing"],
    "breathing":      ["breathing", "respiration", "airway"],
    "choking":        ["airway obstruction", "choking", "heimlich"],
    "choke":          ["airway obstruction", "choking", "heimlich"],

    # Heart / CPR
    "heart attack":   ["heart attack", "cardiac arrest", "myocardial"],
    "cardiac":        ["cardiac arrest", "heart attack", "CPR"],
    "cpr":            ["CPR", "resuscitation", "chest compressions"],
    "resuscitation":  ["resuscitation", "CPR"],
    "chest pain":     ["chest discomfort", "heart attack", "angina"],

    # Bleeding
    "bleeding":       ["bleeding", "haemorrhage", "hemorrhage"],
    "bleed":          ["bleeding", "haemorrhage"],
    "blood":          ["bleeding", "haemorrhage"],
    "wound":          ["bleeding", "wounds", "cleaning wounds"],
    "cut":            ["bleeding", "wounds"],

    # Burns
    "burn":           ["burn injuries", "burns", "scalds"],

    # CPR explicit mappings (must come before generic terms)
    "cpr":            ["cpr", "cardiopulmonary resuscitation",
                       "cpr (cardiopulmonary resuscitation)"],
    "cardiopulmonary":["cpr", "cardiopulmonary resuscitation",
                       "cpr (cardiopulmonary resuscitation)"],
    "resuscitation":  ["cpr", "resuscitation", "cardiopulmonary resuscitation",
                       "cpr (cardiopulmonary resuscitation)"],
    "compressions":   ["cpr", "chest compressions", "cardiopulmonary resuscitation"],
    "chest compression":["cpr", "cardiopulmonary resuscitation",
                         "cpr (cardiopulmonary resuscitation)"],
    "rescue breath":  ["cpr", "cardiopulmonary resuscitation", "resuscitation"],

    # Psychological / Communication (for resistance, refusal, distress questions)
    "resist":         ["psychological first aid", "reassurance", 
                       "reassure the casualty", "first aid for responsive casualty"],
    "resists":        ["psychological first aid", "reassurance",
                       "reassure the casualty", "first aid for responsive casualty"],
    "refuse":         ["psychological first aid", "reassurance",
                       "reassure the casualty"],
    "refuses":        ["psychological first aid", "reassurance"],
    "distress":       ["psychological first aid", "psychological shock",
                       "reassurance", "reassure the casualty"],
    "panic":          ["psychological first aid", "psychological shock",
                       "reassurance"],
    "scared":         ["psychological first aid", "reassurance"],
    "conscious":      ["assessment of consciousness", "checking consciousness",
                       "first aid for responsive casualty"],
    "responsive":     ["first aid for responsive casualty",
                       "assessment of consciousness"],
    "unresponsive":   ["first aid for unresponsive casualty",
                       "transporting unconscious casualty"],
    "psychological":  ["psychological first aid", "psychological shock"],
    "reassure":       ["reassurance", "reassure the casualty",
                       "psychological first aid"],
    "approach":       ["first aid for responsive casualty",
                       "psychological first aid", "reassurance"],

    # Movement / Transport (critical additions based on CSV analysis)
    "move":           ["transport", "transporting", "moving", "casualty transport",
                       "transport techniques", "transporting a casualty"],
    "moving":         ["transport", "transporting", "casualty transport",
                       "transport techniques", "moving and transporting"],
    "drag":           ["dragging", "transport", "transporting a casualty",
                       "moving and transporting", "casualty transport"],
    "dragging":       ["transport", "transporting a casualty", "drag technique",
                       "moving and transporting"],
    "carry":          ["carrying", "carrying a loaded stretcher", "transport",
                       "transporting a casualty"],
    "lift":           ["lifting", "lifting and lowering a stretcher", "transport"],
    "transport":      ["transport techniques", "transporting a casualty",
                       "casualty transport", "transport to healthcare"],
    "crutch":         ["human crutch technique", "human crutch", "supporting",
                       "assisting in moving"],
    "human crutch":   ["human crutch technique", "supporting injured person",
                       "assisting in moving the person"],
    "grip":           ["human crutch technique", "supporting", "holding"],
    "support":        ["human crutch technique", "supporting injured limbs",
                       "supporting the neck", "supporting the head"],
    "stretcher":      ["stretcher", "lifting and lowering a stretcher",
                       "carrying a loaded stretcher", "loading a stretcher"],
    "spinal":         ["spinal injury", "head neck spinal",
                       "moving and transporting a casualty suspected of a head, neck or spinal injury"],
    "spine":          ["spinal injury", "head neck spinal",
                       "moving and transporting a casualty suspected of a head"],
    "unconscious":    ["unconsciousness", "transporting unconscious casualty",
                       "transporting unconscious victims", "recovery position"],
    "ambulance":      ["loading a stretcher into an ambulance", "transport",
                       "urgent transport to hospital"],
    "accident":       ["transport", "casualty transport", "first aid scene management"],
    "injured":        ["casualty transport", "transporting a casualty",
                       "transport techniques"],
    "casualty":       ["casualty transport", "transporting a casualty",
                       "first aid for responsive casualty",
                       "first aid for unresponsive casualty"],
    "body":           ["transporting a casualty", "moving and transporting",
                       "transport techniques"],
    "level":          ["transporting a casualty", "moving and transporting",
                       "casualty position"],
    "arm":            ["human crutch technique", "supporting", "sling",
                       "first aid for shoulder injuries"],
    "shoulder":       ["first aid for shoulder injuries", "human crutch technique",
                       "supporting injured limbs"],
    "neck":           ["spinal injury", "moving and transporting a casualty suspected",
                       "supporting the neck", "neck injury"],
    "head":           ["head injury", "supporting the head",
                       "moving and transporting a casualty suspected of a head"],
    "fire":           ["burns", "moving the victim to fresh air", "smoke inhalation",
                       "transporting unconscious casualty"],
    "danger":         ["safety", "scene management", "transport",
                       "casualty transport"],
    "burning":        ["burn injuries", "burns"],
    "scald":          ["scalds", "burn injuries"],

    # Seizures / fits
    "seizure":        ["seizures", "fits", "convulsions", "epilepsy"],
    "convulsion":     ["convulsions", "seizures", "fits"],
    "fit":            ["fits", "seizures"],
    "epilepsy":       ["epilepsy", "seizures", "fits"],

    # Fractures / injuries
    "broken":         ["fractures", "fracture"],
    "fracture":       ["fractures"],
    "sprain":         ["sprain", "dislocation"],
    "dislocation":    ["dislocation"],
    "injury":         ["injuries", "trauma"],
    "injured":        ["injuries", "trauma"],

    # Shock
    "shock":          ["shock"],
    "anaphylaxis":    ["anaphylaxis", "allergic reaction", "shock"],
    "allergic":       ["allergic reaction", "anaphylaxis"],

    # Poisoning / overdose
    "poison":         ["poisoning", "poison"],
    "overdose":       ["poisoning", "overdose"],
    "swallowed":      ["poisoning", "swallowed"],

    # Drowning / suffocation
    "drowning":       ["drowning", "suffocation"],
    "drown":          ["drowning", "suffocation"],
    "suffocation":    ["suffocation"],
    "smoke":          ["suffocation by smoke", "smoke inhalation"],

    # Diabetes
    "diabetic":       ["diabetes", "hyperglycaemia", "hypoglycaemia"],
    "diabetes":       ["diabetes", "hyperglycaemia", "hypoglycaemia"],
    "blood sugar":    ["hyperglycaemia", "hypoglycaemia", "diabetes"],

    # Stroke
    "stroke":         ["stroke", "brain"],
    "paralysis":      ["stroke", "paralysis"],

    # Moving / transport
    "move":           ["transporting injured person", "moving"],
    "moving":         ["transporting injured person"],
    "transport":      ["transporting injured person"],
    "carry":          ["transporting injured person"],

    # Recovery position
    "recovery position": ["recovery position"],
    "recovery":       ["recovery position"],

    # Child / infant
    "child":          ["child", "infant", "paediatric"],
    "infant":         ["infant", "child", "baby"],
    "baby":           ["infant", "baby"],

    # General
    "pain":           ["pain"],
    "swelling":       ["swelling"],
    "fever":          ["fever", "temperature", "hyperthermia"],
    "temperature":    ["measuring body temperature", "fever"],
    "vomiting":       ["vomiting", "nausea"],
    "nausea":         ["nausea", "vomiting"],
    "dehydration":    ["dehydration", "rehydration"],
    "bandage":        ["bandaging", "dressing"],
    "splint":         ["splinting", "fractures"],
    "crutch":         ["human crutch", "crutch"],
}

# High-value relationship types (prioritised in traversal)
PRIORITY_RELS = {
    'REQUIRES_PROCEDURE': 3,
    'TREATED_BY':         3,
    'TREATS':             3,
    'HAS_SYMPTOM':        2,
    'INDICATES':          2,
    'CAUSES':             1,
    'AFFECTS':            1,
    'REQUIRES':           2,
    'PART_OF':            1,
}


# ─── KG Retriever (v2) ────────────────────────────────────────────────────────

class KGRetriever:
    """
    Improved retriever with synonym expansion, multi-strategy matching,
    and full traversal logging for visualisation.
    """

    def __init__(self, kg_path: str):
        print(f"  Loading KG from {kg_path}...")
        with open(kg_path, 'r', encoding='utf-8') as f:
            self.kg_data = json.load(f)

        # Node lookup: lowercase id → node
        self.node_map = {n['id'].lower(): n for n in self.kg_data['nodes']}
        # Also index by original case
        self.node_map_orig = {n['id']: n for n in self.kg_data['nodes']}

        # Adjacency list with relationship weights
        self.adjacency = defaultdict(list)
        for edge in self.kg_data['edges']:
            src = edge['source'].lower()
            tgt = edge['target'].lower()
            rel = edge.get('relationship', 'RELATED_TO')
            ctx = edge.get('context', '')
            weight = PRIORITY_RELS.get(rel, 1)
            self.adjacency[src].append({
                'neighbour': tgt, 'relationship': rel,
                'context': ctx, 'direction': 'out', 'weight': weight,
                'source_orig': edge['source'], 'target_orig': edge['target']
            })
            self.adjacency[tgt].append({
                'neighbour': src, 'relationship': rel,
                'context': ctx, 'direction': 'in', 'weight': weight,
                'source_orig': edge['source'], 'target_orig': edge['target']
            })

        stats = self.kg_data.get('statistics', {})
        print(f"  KG loaded: {stats.get('total_nodes', len(self.kg_data['nodes']))} nodes, "
              f"{stats.get('total_edges', len(self.kg_data['edges']))} edges")

    def _tokenise(self, query: str) -> list[str]:
        stopwords = {
            'what', 'when', 'where', 'which', 'who', 'how', 'why', 'should',
            'would', 'could', 'does', 'will', 'can', 'are', 'the', 'and',
            'for', 'you', 'that', 'this', 'with', 'from', 'have', 'not',
            'but', 'they', 'been', 'has', 'was', 'were', 'give', 'make',
            'take', 'get', 'put', 'use', 'help', 'need', 'want', 'first',
            'aid', 'someone', 'person', 'people', 'your', 'their', 'its'
        }
        import re
        tokens = re.sub(r'[^\w\s]', ' ', query.lower()).split()
        return [t for t in tokens if t not in stopwords and len(t) >= 3]

    def _expand_with_synonyms(self, tokens: list[str]) -> list[str]:
        """Expand tokens with synonym map → extra search terms."""
        expanded = list(tokens)
        query_lower = ' '.join(tokens)
        for phrase, expansions in SYNONYMS.items():
            if phrase in query_lower or any(t in phrase for t in tokens):
                expanded.extend(expansions)
        return list(dict.fromkeys(expanded))  # deduplicate, preserve order

    def _score_nodes(self, search_terms: list[str]) -> dict[str, float]:
        """Score every node by how well it matches search terms."""
        scores = defaultdict(float)
        for node_id in self.node_map:
            for term in search_terms:
                if term == node_id:
                    scores[node_id] += 5.0          # exact match
                elif node_id.startswith(term):
                    scores[node_id] += 3.0          # prefix match
                elif term in node_id:
                    scores[node_id] += 2.0          # substring match
                    # Bonus for procedure/condition nodes
                    ntype = self.node_map[node_id].get('attributes', {}).get('type', '')
                    if ntype in ('procedure', 'procedures', 'conditions'):
                        scores[node_id] += 1.0
        return scores

    def _weighted_bfs(self, seed_nodes: list[str],
                      depth: int = 2,
                      traversal_log: list = None) -> tuple[list, list]:
        """
        Weighted BFS — prioritises high-value relationship types.
        traversal_log: if provided, records every step for visualisation.
        """
        visited  = set(seed_nodes)
        frontier = list(seed_nodes)
        all_edges = []

        for d in range(depth):
            next_frontier = []
            # Sort frontier edges by weight (priority rels first)
            candidates = []
            for node_id in frontier:
                for edge in self.adjacency.get(node_id, []):
                    candidates.append((edge['weight'], node_id, edge))
            candidates.sort(key=lambda x: -x[0])

            for weight, from_node, edge in candidates:
                neighbour = edge['neighbour']
                if len(visited) >= 30:
                    break
                edge_record = {
                    'from':         from_node,
                    'to':           neighbour,
                    'relationship': edge['relationship'],
                    'context':      edge['context'],
                    'depth':        d + 1,
                    'weight':       weight,
                    'source_orig':  edge['source_orig'],
                    'target_orig':  edge['target_orig'],
                }
                all_edges.append(edge_record)
                if traversal_log is not None:
                    traversal_log.append(edge_record)
                if neighbour not in visited:
                    visited.add(neighbour)
                    next_frontier.append(neighbour)
            frontier = next_frontier
            if not frontier:
                break

        nodes = [self.node_map[nid] for nid in visited if nid in self.node_map]
        return nodes, all_edges[:40]

    def _extract_procedures(self, nodes: list) -> list[dict]:
        """Extract and rank procedure nodes by completeness."""
        procs = []
        for node in nodes:
            attrs = node.get('attributes', {})
            if attrs.get('type') in ('procedure', 'procedures'):
                steps = attrs.get('steps', [])
                if steps:
                    procs.append({
                        'name':      node['id'],
                        'steps':     steps[:8],
                        'equipment': attrs.get('equipment', []),
                        'warnings':  attrs.get('warnings', []),
                        'seek_help': attrs.get('seek_help', []),
                        'score':     len(steps),  # rank by completeness
                    })
        return sorted(procs, key=lambda x: -x['score'])[:4]

    def retrieve(self, query: str,
                 traversal_log: list = None) -> str:
        """
        Full retrieval → context string.
        If traversal_log list provided, it gets populated with traversal steps.
        """
        tokens       = self._tokenise(query)
        search_terms = self._expand_with_synonyms(tokens)

        if not search_terms:
            return ""

        # Score and rank nodes
        node_scores = self._score_nodes(search_terms)

        # Fallback 1 — ontology search if keyword match is weak
        if len(node_scores) < 2:
            ontology = self.kg_data.get('ontology', {})
            for category, items in ontology.items():
                for item in items:
                    item_lower = item.lower()
                    if any(t in item_lower or item_lower in t
                           for t in search_terms):
                        for nid in self.node_map:
                            if item_lower in nid or nid in item_lower:
                                node_scores[nid] = node_scores.get(nid, 0) + 1.0

        # Fallback 2 — edge context search if still insufficient
        if len(node_scores) < 2:
            for edge in self.kg_data['edges']:
                ctx = edge.get('context', '').lower()
                if not ctx:
                    continue
                score = sum(1 for t in search_terms if len(t) >= 4 and t in ctx)
                if score > 0:
                    for nid in [edge['source'].lower(), edge['target'].lower()]:
                        node_scores[nid] = node_scores.get(nid, 0) + score * 0.5

        if not node_scores:
            return ""

        # Take top seeds weighted by score
        ranked = sorted(node_scores.items(), key=lambda x: -x[1])
        seed_nodes = [nid for nid, _ in ranked[:6]]

        if traversal_log is not None:
            traversal_log.append({
                'stage':       'seed_selection',
                'query':       query,
                'tokens':      tokens,
                'search_terms': search_terms,
                'seeds':       [(nid, score) for nid, score in ranked[:6]],
            })

        # Weighted BFS
        nodes, edges = self._weighted_bfs(seed_nodes, depth=2,
                                          traversal_log=traversal_log)

        # Procedure enrichment
        procedures = self._extract_procedures(nodes)

        # ── Format context ─────────────────────────────────────────────────
        parts = []

        # Grouped entities
        entity_groups = defaultdict(list)
        for node in nodes:
            ntype = node.get('attributes', {}).get('type', 'entity')
            entity_groups[ntype].append(node['id'])

        if entity_groups:
            parts.append("=== Relevant Medical Entities ===")
            for ntype, names in sorted(entity_groups.items()):
                parts.append(f"{ntype.upper()}: {', '.join(names[:6])}")

        # Key relationships (prioritise high-weight ones)
        top_edges = sorted(edges, key=lambda x: -x.get('weight', 1))[:15]
        if top_edges:
            parts.append("\n=== Key Relationships ===")
            for e in top_edges:
                line = (f"• {e['source_orig']} "
                        f"--[{e['relationship']}]--> "
                        f"{e['target_orig']}")
                if e.get('context'):
                    line += f"\n  ↳ {e['context'][:100]}"
                parts.append(line)

        # Step-by-step procedures
        if procedures:
            parts.append("\n=== Step-by-Step Procedures ===")
            for proc in procedures:
                parts.append(f"\n[{proc['name'].upper()}]")
                for i, step in enumerate(proc['steps'], 1):
                    parts.append(f"  Step {i}: {step}")
                if proc['warnings']:
                    parts.append(f"  ⚠ WARNINGS: {'; '.join(proc['warnings'][:3])}")
                if proc['equipment']:
                    parts.append(f"  Equipment: {', '.join(proc['equipment'][:5])}")
                if proc['seek_help']:
                    parts.append(f"  Seek help if: {'; '.join(proc['seek_help'][:2])}")

        return '\n'.join(parts)


# ─── Traversal Visualiser ─────────────────────────────────────────────────────

class TraversalVisualizer:
    """
    Generates human-readable and JSON traversal reports.
    Shows exactly how the KG arrived at its answer.
    """

    @staticmethod
    def format_text(traversal_log: list, question: str, answer: str) -> str:
        """Plain-text traversal report for console/file output."""
        lines = [
            "=" * 70,
            f"KNOWLEDGE GRAPH TRAVERSAL REPORT",
            f"Question: {question}",
            "=" * 70,
        ]

        for entry in traversal_log:
            if entry.get('stage') == 'seed_selection':
                lines.append("\n── STAGE 1: Query Analysis & Seed Selection ──")
                lines.append(f"  Original tokens:  {entry['tokens']}")
                lines.append(f"  After synonyms:   {entry['search_terms'][:10]}")
                lines.append(f"  Top seed nodes:")
                for nid, score in entry['seeds']:
                    lines.append(f"    [{score:.1f}] {nid}")
            else:
                depth = entry.get('depth', '?')
                rel   = entry.get('relationship', '')
                frm   = entry.get('source_orig', entry.get('from', ''))
                to    = entry.get('target_orig', entry.get('to', ''))
                ctx   = entry.get('context', '')
                w     = entry.get('weight', 1)
                priority = '★' * min(w, 3)
                lines.append(
                    f"  [D{depth}] {priority} {frm} --[{rel}]--> {to}"
                )
                if ctx:
                    lines.append(f"       ↳ {ctx[:90]}")

        lines.append(f"\n── FINAL ANSWER ──")
        lines.append(answer)
        lines.append("=" * 70)
        return '\n'.join(lines)

    @staticmethod
    def save_json(traversal_log: list, question: str,
                  answer: str, path: str):
        """Save traversal as JSON for downstream analysis."""
        report = {
            'question': question,
            'answer':   answer,
            'traversal': traversal_log,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)


# ─── Groq Generator ───────────────────────────────────────────────────────────

class GroqKGGenerator:
    """Groq LLM generator — uses KG context, same model as baseline for fair comparison."""
    GROQ_API = "https://api.groq.com/openai/v1/chat/completions"
    MODEL    = "llama-3.1-8b-instant"

    SYSTEM_PROMPT = """You are a first aid expert with access to the Indian Red Cross Society First Aid Manual via a Knowledge Graph.

You are given structured knowledge retrieved from the KG including:
- Relevant medical entities and their relationships
- Step-by-step procedures directly from the RedCross manual
- Warnings and equipment requirements

Your task: write a clear, concise, actionable first aid answer.
Rules:
- Base your answer ONLY on the KG context provided
- Follow the procedure steps in order when available
- Include warnings when present
- Mention when professional help is needed
- Keep it under 120 words
- Write in plain prose, not bullet points"""

    def __init__(self, api_key: str = ''):
        self.api_key = api_key or os.environ.get('GROQ_API_KEY', '')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

    def generate(self, question: str, context: str,
                 max_new_tokens: int = 200) -> str:
        user_content = f"Knowledge Graph Context:\n{context}\n\nQuestion: {question}"
        payload = {
            "model":    self.MODEL,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            "max_tokens":  max_new_tokens,
            "temperature": 0.1,
        }
        for attempt in range(3):
            try:
                resp = requests.post(self.GROQ_API, headers=self.headers,
                                     json=payload, timeout=30)
                if resp.status_code == 429:
                    wait = int(resp.headers.get('retry-after', 15))
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    return ""
                return resp.json()['choices'][0]['message']['content'].strip()
            except Exception:
                if attempt == 2:
                    return ""
                time.sleep(3)
        return ""



# ─── BioMistral Local Generator ───────────────────────────────────────────────

class BioMistralKGGenerator:
    """
    Local BioMistral-7B generator — reuses the singleton from biomistral_backend.py
    so the weights are loaded only once even if KGPipeline is instantiated
    multiple times.  Zero API calls, no rate limits.
    """
    SYSTEM_PROMPT = """You are a first aid expert with access to the Indian Red Cross Society First Aid Manual via a Knowledge Graph.

You are given structured knowledge retrieved from the KG including:
- Relevant medical entities and their relationships
- Step-by-step procedures directly from the RedCross manual
- Warnings and equipment requirements

Your task: write a clear, concise, actionable first aid answer.
Rules:
- Base your answer ONLY on the KG context provided
- Follow the procedure steps in order when available
- Include warnings when present
- Mention when professional help is needed
- Keep it under 120 words
- Write in plain prose, not bullet points"""

    def __init__(self, load_in_4bit: bool = True, groq_api_key: str = ''):
        print("  [BioMistralKGGenerator] Loading BioMistral backend...")
        from biomistral_backend import get_biomistral_backend
        self.backend = get_biomistral_backend(
            prefer_local=True,
            load_in_4bit=load_in_4bit,
            groq_api_key=groq_api_key,
        )
        print(f"  [BioMistralKGGenerator] Backend ready: {self.backend}")

    def generate(self, question: str, context: str,
                 max_new_tokens: int = 200) -> str:
        # Reorder context: procedures first (most valuable for BioMistral)
        lines = context.split('\n')
        proc_start = next(
            (i for i, l in enumerate(lines) if 'Step-by-Step Procedures' in l), None
        )
        rel_start = next(
            (i for i, l in enumerate(lines) if 'Key Relationships' in l), None
        )
        if proc_start is not None:
            proc_section = '\n'.join(lines[proc_start:])
            if rel_start is not None and rel_start < proc_start:
                rel_section = '\n'.join(lines[rel_start:proc_start])[:500]
                context = proc_section + '\n' + rel_section
            else:
                context = proc_section

        user_content = f"Knowledge Graph Context:\n{context}\n\nQuestion: {question}"
        return self.backend.chat(self.SYSTEM_PROMPT, user_content,
                                 max_new_tokens=max_new_tokens)




class KGPipeline:
    """
    KG retrieval + LLM generation. Primary 'Your Method' system.

    prefer_local=True  (default) → uses local BioMistral-7B via biomistral_backend.py
                                    reuses the singleton — no duplicate weight loads
                                    no API calls, no rate limits
    prefer_local=False           → uses Groq API (llama-3.1-8b-instant)
                                    requires GROQ_API_KEY, has rate limits
    """

    def __init__(self, kg_path: str = 'medical_kg.json',
                 groq_api_key: str = '',
                 prefer_local: bool = True,
                 load_in_4bit: bool = True,
                 save_traversals: bool = False,
                 traversal_dir: str = 'traversals'):
        self.name = ("Your Method (RedCross KG + BioMistral-7B)"
                     if prefer_local else
                     "Your Method (RedCross KG + Llama-3.1 Groq)")
        if not os.path.exists(kg_path):
            raise FileNotFoundError(f"KG file not found: {kg_path}")
        self.retriever = KGRetriever(kg_path)

        if prefer_local:
            self.generator = BioMistralKGGenerator(
                load_in_4bit=load_in_4bit,
                groq_api_key=groq_api_key,
            )
        else:
            self.generator = GroqKGGenerator(groq_api_key)

        self.save_traversals = save_traversals
        self.traversal_dir   = traversal_dir
        self._sample_count   = 0
        if save_traversals:
            os.makedirs(traversal_dir, exist_ok=True)

    def generate(self, question: str,
                 max_new_tokens: int = 200) -> tuple[str, str, list]:
        traversal_log = []
        context = self.retriever.retrieve(question,
                                          traversal_log=traversal_log)
        answer  = self.generator.generate(question, context, max_new_tokens)
        return answer, context, traversal_log

    def generate_answer_only(self, question: str,
                             max_new_tokens: int = 200) -> str:
        self._sample_count += 1
        traversal_log = []
        context = self.retriever.retrieve(question,
                                          traversal_log=traversal_log)
        answer  = self.generator.generate(question, context, max_new_tokens)

        if self.save_traversals:
            path = os.path.join(
                self.traversal_dir,
                f"traversal_{self._sample_count:04d}.json"
            )
            TraversalVisualizer.save_json(traversal_log, question, answer, path)

        return answer


# ─── Pure KG Pipeline (no LLM) ───────────────────────────────────────────────

class PureKGFormatter:
    """
    Converts KG retrieval into clean answer prose — zero LLM calls.
    Improved: extracts procedure steps into numbered list format.
    """

    def _is_relevant(self, question: str, context: str) -> bool:
        """
        Check if retrieved context is relevant to the question.
        Uses TF-IDF-style logic: checks DISTINCTIVE question tokens
        (not common words like 'casualty', 'emergency', 'during').
        """
        import re
        # Words that appear in almost every first aid context — not distinctive
        common_firstaid = {
            'what','should','you','do','if','how','when','the','a','an','is',
            'are','and','or','for','to','with','of','in','on','during','while',
            'first','aid','person','being','their','they','them','have','has',
            'been','will','would','that','this','which','from','after','before',
            'then','also','can','may','must','need','make','take','give','keep',
            # Action words that appear in questions but not node IDs
            'perform','treat','handle','apply','provide','someone','swallows',
            'helped','adult','broken','child','minor','someone','other','another',
        }
        q_tokens = set(re.sub(r'[^\w\s]', ' ', question.lower()).split())
        # Keep only distinctive topic words — 4+ chars, not in common set
        distinctive = [t for t in q_tokens
                       if len(t) >= 4 and t not in common_firstaid]
        if len(distinctive) <= 1:
            return True   # not enough signal — allow through
        ctx_lower = context.lower()
        matches = sum(1 for t in distinctive if t in ctx_lower)
        # Need ANY distinctive token to match (very permissive)
        return matches >= 1

    def format(self, question: str, context: str) -> str:
        if not context.strip():
            return "No relevant information found in the knowledge graph."
        # Check relevance — avoid returning wrong procedures
        if not self._is_relevant(question, context):
            return ("The knowledge graph does not contain a specific procedure "
                    "for this scenario. General first aid principles apply: "
                    "ensure scene safety, assess the casualty, and seek "
                    "professional medical help if the situation is unclear.")

        lines = context.split('\n')
        answer_parts = []

        # Extract procedure blocks
        procedure_blocks = []
        current_proc = None
        current_steps = []
        current_warnings = []
        current_seek = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                if current_proc and current_steps:
                    procedure_blocks.append({
                        'name': current_proc,
                        'steps': current_steps,
                        'warnings': current_warnings,
                        'seek': current_seek,
                    })
                current_proc     = stripped[1:-1].title()
                current_steps    = []
                current_warnings = []
                current_seek     = []
            elif 'Step ' in stripped and ':' in stripped and current_proc:
                step_text = stripped.split(':', 1)[-1].strip()
                if step_text:
                    current_steps.append(step_text)
            elif stripped.startswith('⚠') and current_proc:
                current_warnings.append(stripped.replace('⚠ WARNINGS:', '').strip())
            elif stripped.startswith('Seek help') and current_proc:
                current_seek.append(stripped.replace('Seek help if:', '').strip())

        if current_proc and current_steps:
            procedure_blocks.append({
                'name': current_proc,
                'steps': current_steps,
                'warnings': current_warnings,
                'seek': current_seek,
            })

        # Pick most relevant procedure by scoring name against query tokens
        if procedure_blocks:
            import re as _re
            q_tokens = set(_re.sub(r'[^\w\s]', ' ', question.lower()).split())
            for proc in procedure_blocks:
                name_tokens = set(proc['name'].lower().split())
                proc['relevance'] = len(q_tokens & name_tokens)
            best = max(procedure_blocks, key=lambda x: (x['relevance'], len(x['steps'])))
            answer_parts.append(f"{best['name']}:")
            for i, step in enumerate(best['steps'][:6], 1):
                answer_parts.append(f"{i}. {step}")
            if best['warnings']:
                answer_parts.append(f"Warning: {best['warnings'][0]}")
            if best['seek']:
                answer_parts.append(f"Seek medical help if: {best['seek'][0]}")
        else:
            # Fallback: extract meaningful context sentences
            rel_lines = [l for l in lines if '↳' in l]
            for l in rel_lines[:5]:
                ctx = l.replace('↳', '').strip()
                if len(ctx) > 20:
                    answer_parts.append(f"• {ctx}")

        return '\n'.join(answer_parts) if answer_parts else "Consult a medical professional."


class PureKGPipeline:
    """Pure KG pipeline — no LLM, no external calls. Used for ablation study."""

    def __init__(self, kg_path: str = 'medical_kg.json'):
        self.name = "Pure KG (RedCross KG, no LLM)"
        if not os.path.exists(kg_path):
            raise FileNotFoundError(f"KG file not found: {kg_path}")
        self.retriever = KGRetriever(kg_path)
        self.formatter = PureKGFormatter()

    def generate_answer_only(self, question: str, **kwargs) -> str:
        context = self.retriever.retrieve(question)
        return self.formatter.format(question, context)

    def explain(self, question: str) -> str:
        """Returns full traversal report for a single question."""
        traversal_log = []
        context = self.retriever.retrieve(question, traversal_log=traversal_log)
        answer  = self.formatter.format(question, context)
        return TraversalVisualizer.format_text(traversal_log, question, answer)