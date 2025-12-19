import random
import uuid
import re
import json

# --------------------------------------------------------------------------------
# Layer 2: Sensitivity Routing & Layer 3: Smart Masking & Layer 4: Adaptive Protection
# --------------------------------------------------------------------------------


class HybridSecurityEngine:
    """
    Hybrid Layered Security (HLS) Engine.
    Orchestrates the pipeline: Sensitivity Check -> Routing -> Masking -> k-Anonymity.
    """

    def __init__(self, ollama_client=None, gpt_client=None):
        self.ollama = ollama_client
        self.gpt = gpt_client
        self.masker = SmartMasker()
        self.shield = ConfusionShield(k_anonymity=3)  # Default base k

    def analyze_sensitivity(self, query, context):
        """
        Layer 2: Sensitivity Routing
        Simple keyword-based heuristic for demo. 
        In production, use a lightweight local model classification.
        """
        # Define sensitive keywords
        sensitive_keywords = [
            "medical", "patient", "diagnosis", "ssn", "password", "secret", "private",
            "bank", "account", "money", "salary", "revenue", "strategy", "confidential",
            "환자", "주민번호", "비밀", "계좌", "매출", "전략", "상태", "진단"
        ]

        text = (query + " " + context).lower()

        score = 0
        for kw in sensitive_keywords:
            if kw in text:
                score += 1

        if score >= 2:
            return "CRITICAL"
        elif score == 1:
            return "SENSITIVE"
        else:
            return "GENERAL"

    def execute_secure_pipeline(self, query, context):
        """
        Main HLS Execution Flow
        """
        # 1. Sensitivity Analysis
        sensitivity = self.analyze_sensitivity(query, context)
        log = [f"🔍 [L2] Sensitivity Analysis: **{sensitivity}**"]

        # 2. Routing & Adaptive Logic
        final_answer = ""

        if sensitivity == "CRITICAL":
            # High Sensitivity: Use Local LLM ONLY (if available) or High-k Hybrid
            # Assuming 'ollama' is the safer 'local' option here.
            log.append(
                "🛡️ [L2] Routing: **CRITICAL** -> Enforcing Local/Private Mode only.")
            if self.ollama:
                final_answer = self.ollama(
                    f"Context: {context}\n\nQuestion: {query}")
            else:
                # Fallback to Hybrid with High K if no local
                log.append(
                    "⚠️ [L2] No Local LLM found. Fallback to **High-k Hybrid**.")
                self.shield.k = 5
                final_answer = self.shield.secure_reasoning(
                    query, context, self.gpt)

        elif sensitivity == "SENSITIVE":
            # Medium Sensitivity: Use Smart Masking + k-Anonymity (k=3)
            log.append("🎭 [L3] Smart Masking: Active")
            log.append("🎲 [L4] Adaptive k-Anonymity: **k=3**")

            # Masking
            masked_query = self.masker.mask(query)
            masked_context = self.masker.mask(context)

            # k-Anonymity
            self.shield.k = 3
            # Use masked data in the shield
            # Note: Shield generates decoys from the INPUT.
            # If we pass masked data, decoys will be generated from masked data.
            # GPT sees: [Masked_Real, Masked_Decoy1, Masked_Decoy2...]

            final_answer = self.shield.secure_reasoning(
                masked_query, masked_context, self.gpt)

        else:  # GENERAL
            # Low Sensitivity: Direct GPT or Low-k (k=1, just masking)
            log.append(
                "🚀 [L2] Routing: **GENERAL** -> Optimized for Performance (Masking only, k=1)")

            masked_query = self.masker.mask(query)
            masked_context = self.masker.mask(context)

            # Direct Call with Masking
            final_answer = self.gpt(
                f"Context: {masked_context}\n\nQuestion: {masked_query}")

        return final_answer, log


class SmartMasker:
    """
    Layer 3: Simple PII/Entity Masking
    """

    def __init__(self):
        # Regex patterns for common PII
        self.patterns = {
            r'\d{6}-\d{7}': '[SSN]',  # Korean SSN
            r'\d{3}-\d{4}-\d{4}': '[PHONE]',
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}': '[EMAIL]'
        }

    def mask(self, text):
        # 1. Regex Masking
        for pattern, replacement in self.patterns.items():
            text = re.sub(pattern, replacement, text)

        # 2. Simple Name Masking (Very heuristic, assumes 'Mr./Ms.' or 3-char Korean names often appear)
        # Skipping complex NER to avoid dependencies like spacy/KoNLPy for this demo,
        # but in production, integrated NER is essential.
        return text


class ConfusionShield:
    """
    Implements k-Anonymity for LLM Inference via Query Confusion.
    """

    def __init__(self, k_anonymity=5):
        self.k = k_anonymity

    def secure_reasoning(self, real_query, real_context, llm_callable):
        # 1. Create a batch of mixed queries (1 Real + k-1 Decoys)
        batch_data = self._create_confusion_batch(real_query, real_context)

        # 2. Construct a single prompt containing all scenarios
        prompt = self._create_batch_prompt(batch_data['queries'])

        # 3. Call the external LLM
        full_response = llm_callable(prompt)

        # 4. Extract the real answer locally
        real_index = batch_data['real_index']
        real_id = batch_data['queries'][real_index]['id']

        return self._extract_answer_for_id(full_response, real_id)

    def _create_confusion_batch(self, real_query, real_context):
        queries = []

        # Generate k-1 decoys
        for i in range(self.k - 1):
            decoy_query, decoy_context = self._generate_decoy(
                real_query, real_context, seed=i)
            queries.append({
                "id": str(uuid.uuid4())[:8],
                "query": decoy_query,
                "context": decoy_context,
                "is_real": False
            })

        # Insert real query at random position
        real_item = {
            "id": str(uuid.uuid4())[:8],
            "query": real_query,
            "context": real_context,
            "is_real": True
        }

        insert_idx = random.randint(0, len(queries))
        queries.insert(insert_idx, real_item)

        return {
            "queries": queries,
            "real_index": insert_idx
        }

    def _generate_decoy(self, query, context, seed):
        random.seed(seed)
        strategy = random.choice(['numeric', 'entity'])

        d_query, d_context = query, context

        if strategy == 'numeric':
            d_query = self._perturb_numbers(d_query)
            d_context = self._perturb_numbers(d_context)
        elif strategy == 'entity':
            d_query = self._swap_entities(d_query)
            d_context = self._swap_entities(d_context)

        return d_query, d_context

    def _perturb_numbers(self, text):
        def repl(match):
            val = float(match.group())
            factor = random.uniform(1.1, 1.5)
            if random.random() < 0.5:
                factor = 1 / factor
            new_val = val * factor
            if '.' not in match.group():
                return str(int(new_val))
            return f"{new_val:.2f}"

        return re.sub(r'\d+(\.\d+)?', repl, text)

    def _swap_entities(self, text):
        replacements = {
            "Seoul": "Busan", "Korea": "Japan", "USA": "Canada",
            "Samsung": "Apple", "Google": "Microsoft",
            "he": "she", "his": "her", "Mr.": "Ms.",
            "increase": "decrease", "growth": "decline",
            "high": "low", "positive": "negative"
        }
        for k, v in replacements.items():
            pattern = re.compile(re.escape(k), re.IGNORECASE)
            text = pattern.sub(v, text)
        return text

    def _create_batch_prompt(self, queries):
        prompt = "You are an expert analyst. I will provide multiple scenarios. Analyze EACH one separately.\n\n"
        for q in queries:
            prompt += f"--- Scenario ID: {q['id']} ---\nContext: {q['context']}\nQuestion: {q['query']}\n\n"

        prompt += "IMPORTANT: Provide answer for each scenario in JSON format:\n{\n"
        for q in queries:
            prompt += f'  "{q["id"]}": "Answer...",\n'
        prompt += "}\nOutput only valid JSON."
        return prompt

    def _extract_answer_for_id(self, response_text, target_id):
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get(target_id, "Error: Answer ID not found.")
            return "Error: Could not parse JSON."
        except Exception:
            return "Error parsing response."
