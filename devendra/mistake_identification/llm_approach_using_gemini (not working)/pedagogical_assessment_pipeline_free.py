"""
Pedagogical Assessment Pipeline - FREE LLM VERSION
===================================================
This script uses FREE/Open-Source LLMs instead of paid APIs:
- Ollama (local models - completely free)
- HuggingFace Inference API (free tier available)
- Google Gemini (free tier available)
- Groq (free tier with fast inference)

1. Reads the training dataset
2. Extracts math problems from conversations
3. Uses FREE LLMs to solve problems and generate answer keys
4. Evaluates LLM responses for mistake identification quality
5. Creates validation and test splits
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import random
from tqdm import tqdm
import time
import requests

# Try to import various free LLM libraries
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Note: Google Gemini not available. Install with: pip install google-generativeai")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Note: Groq not available. Install with: pip install groq")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Note: HuggingFace Transformers not available. Install with: pip install transformers")


class FreeMathProblemSolver:
    """Solves math problems using FREE LLMs."""
    
    def __init__(self, model: str = "llama3.1:8b", provider: str = "ollama", api_key: Optional[str] = None):
        """
        Initialize the math problem solver with free LLM options.
        
        Args:
            model: Model name to use
            provider: "ollama", "gemini", "groq", "huggingface"
            api_key: Optional API key (only for gemini, groq, huggingface)
        
        Providers:
        - ollama: Free local models (requires Ollama installed)
          Models: llama3.1:8b, llama3.1:70b, mistral, qwen2.5, deepseek-coder
        - gemini: Google Gemini (free tier: 15 requests/min)
          Models: gemini-1.5-flash, gemini-1.5-pro
        - groq: Groq Cloud (free tier: very fast)
          Models: llama-3.1-70b-versatile, mixtral-8x7b-32768
        - huggingface: HuggingFace Inference API (free tier available)
          Models: meta-llama/Meta-Llama-3-8B-Instruct, etc.
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        
        # Initialize based on provider
        if provider == "ollama":
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self._check_ollama()
        
        elif provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai not installed")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY required. Get free key at: https://makersuite.google.com/app/apikey")
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(model)
        
        elif provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("groq not installed")
            if not self.api_key:
                raise ValueError("GROQ_API_KEY required. Get free key at: https://console.groq.com")
            self.client = Groq(api_key=self.api_key)
        
        elif provider == "huggingface":
            if not self.api_key:
                raise ValueError("HUGGINGFACE_API_KEY required. Get free key at: https://huggingface.co/settings/tokens")
            self.base_url = f"https://api-inference.huggingface.co/models/{model}"
        
        else:
            raise ValueError(f"Unknown provider: {provider}. Use: ollama, gemini, groq, or huggingface")
    
    def _check_ollama(self):
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                if self.model not in models:
                    print(f"⚠️  Model '{self.model}' not found in Ollama.")
                    print(f"   Available models: {', '.join(models)}")
                    print(f"   To install: ollama pull {self.model}")
                else:
                    print(f"✓ Ollama is running with model: {self.model}")
            else:
                raise Exception("Ollama not responding")
        except Exception as e:
            print(f"⚠️  Ollama check failed: {e}")
            print(f"   Make sure Ollama is running: https://ollama.ai")
    
    def extract_math_problem(self, conversation_history: str) -> Optional[str]:
        """Extract the math problem from the conversation history."""
        pattern = r"The question is:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, conversation_history, re.DOTALL)
        
        if match:
            problem = match.group(1)
            student_idx = problem.find("Student:")
            if student_idx != -1:
                problem = problem[:student_idx]
            return problem.strip()
        
        return None
    
    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """
        Solve a math problem using the free LLM.
        
        Returns:
            Dictionary with 'answer', 'solution_steps', and 'explanation'
        """
        prompt = f"""You are an expert mathematics tutor. Solve the following math problem step by step.

Problem: {problem}

Provide your response in the following format:
1. **Step-by-step solution**: Show all your work clearly
2. **Final Answer**: State the final numerical answer clearly
3. **Explanation**: Brief explanation of the approach

Be precise and show all calculations."""

        try:
            if self.provider == "ollama":
                solution_text = self._solve_with_ollama(prompt)
            
            elif self.provider == "gemini":
                solution_text = self._solve_with_gemini(prompt)
            
            elif self.provider == "groq":
                solution_text = self._solve_with_groq(prompt)
            
            elif self.provider == "huggingface":
                solution_text = self._solve_with_huggingface(prompt)
            
            # Parse the solution
            answer = self._extract_answer(solution_text)
            
            return {
                "answer": answer,
                "solution_steps": solution_text,
                "explanation": self._extract_explanation(solution_text),
                "model_used": f"{self.provider}/{self.model}"
            }
        
        except Exception as e:
            print(f"Error solving problem: {e}")
            return {
                "answer": None,
                "solution_steps": None,
                "explanation": f"Error: {str(e)}",
                "model_used": f"{self.provider}/{self.model}"
            }
    
    def _solve_with_ollama(self, prompt: str) -> str:
        """Solve using Ollama (local, completely free)."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 2000
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]
    
    def _solve_with_gemini(self, prompt: str) -> str:
        """Solve using Google Gemini (free tier)."""
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        response = self.client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=2000,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
        )
        # Try to get text from response, handling blocked responses
        try:
            return response.text
        except ValueError as e:
            # Response was blocked or has no text
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content.parts:
                    return candidate.content.parts[0].text
            # If all else fails, return error message
            finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
            raise ValueError(f"Gemini response blocked or empty. Finish reason: {finish_reason}")
    
    def _solve_with_groq(self, prompt: str) -> str:
        """Solve using Groq (free tier, very fast)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert mathematics tutor who solves problems step by step with perfect accuracy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    def _solve_with_huggingface(self, prompt: str) -> str:
        """Solve using HuggingFace Inference API (free tier)."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            self.base_url,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "temperature": 0.1,
                    "max_new_tokens": 2000,
                    "return_full_text": False
                }
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return str(result)
    
    def _extract_answer(self, solution_text: str) -> Optional[str]:
        """Extract the final numerical answer from the solution."""
        patterns = [
            r"Final Answer[:\s]+([0-9,.]+)",
            r"Answer[:\s]+([0-9,.]+)",
            r"=\s*([0-9,.]+)\s*$",
            r"\$([0-9,.]+)",
            r"([0-9,.]+)\s*(?:dollars|pounds|units|people|sacks|coins|sandwiches|students)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution_text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).replace(",", "")
        
        # If no pattern matched, try to find the last number in the text
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", solution_text)
        if numbers:
            return numbers[-1]
        
        return None
    
    def _extract_explanation(self, solution_text: str) -> str:
        """Extract the explanation section."""
        explanation_match = re.search(r"Explanation[:\s]+(.+)", solution_text, re.DOTALL | re.IGNORECASE)
        if explanation_match:
            return explanation_match.group(1).strip()
        return solution_text[:200]


class FreeMistakeIdentificationEvaluator:
    """Evaluates LLM responses using FREE LLMs."""
    
    def __init__(self, model: str = "llama3.1:8b", provider: str = "ollama", api_key: Optional[str] = None):
        """Initialize the evaluator with free LLM."""
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        
        # Initialize based on provider (same as solver)
        if provider == "ollama":
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        elif provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai not installed")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY required")
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(model)
        
        elif provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("groq not installed")
            if not self.api_key:
                raise ValueError("GROQ_API_KEY required")
            self.client = Groq(api_key=self.api_key)
        
        elif provider == "huggingface":
            if not self.api_key:
                raise ValueError("HUGGINGFACE_API_KEY required")
            self.base_url = f"https://api-inference.huggingface.co/models/{model}"
    
    def evaluate_response(
        self,
        conversation_history: str,
        correct_answer: str,
        tutor_response: str,
        current_annotation: Dict[str, str]
    ) -> Dict[str, Any]:
        """Evaluate whether a tutor response correctly identifies mistakes."""
        prompt = f"""You are an expert evaluator of pedagogical interactions. Your task is to evaluate whether a tutor's response correctly identifies student mistakes and provides appropriate guidance.

**Conversation History:**
{conversation_history}

**Correct Answer:** {correct_answer}

**Tutor's Response:**
{tutor_response}

**Original Human Annotation:**
- Mistake Identification: {current_annotation.get('Mistake_Identification', 'Unknown')}
- Providing Guidance: {current_annotation.get('Providing_Guidance', 'Unknown')}

**Your Task:**
Analyze the entire conversation and the tutor's response. Determine:

1. **Mistake Identification**: Does the tutor correctly identify the student's mistake(s)?
   - "Yes": The tutor clearly and accurately identifies the specific mistake(s)
   - "No": The tutor fails to identify any mistakes or identifies wrong issues
   - "To some extent": The tutor partially identifies mistakes but misses key issues or is vague

2. **Providing Guidance**: Does the tutor provide helpful guidance to help the student?
   - "Yes": The tutor provides clear, actionable guidance that helps the student
   - "No": The tutor provides no guidance or unhelpful guidance
   - "To some extent": The tutor provides some guidance but it's incomplete or unclear

Provide your response in JSON format:
{{
    "mistake_identification": "Yes/No/To some extent",
    "providing_guidance": "Yes/No/To some extent",
    "reasoning": "Detailed explanation of your evaluation",
    "confidence": 0.85,
    "agrees_with_human": true/false
}}

Respond ONLY with the JSON, no other text."""

        try:
            if self.provider == "ollama":
                result_text = self._evaluate_with_ollama(prompt)
            elif self.provider == "gemini":
                result_text = self._evaluate_with_gemini(prompt)
            elif self.provider == "groq":
                result_text = self._evaluate_with_groq(prompt)
            elif self.provider == "huggingface":
                result_text = self._evaluate_with_huggingface(prompt)
            
            # Try to extract JSON from the response
            result = self._extract_json(result_text)
            
            return {
                "predicted_mistake_identification": result.get("mistake_identification", "Unknown"),
                "predicted_providing_guidance": result.get("providing_guidance", "Unknown"),
                "reasoning": result.get("reasoning", ""),
                "confidence": result.get("confidence", 0.0),
                "agrees_with_human": result.get("agrees_with_human", False),
                "model_used": f"{self.provider}/{self.model}"
            }
        
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return {
                "predicted_mistake_identification": "Error",
                "predicted_providing_guidance": "Error",
                "reasoning": f"Error: {str(e)}",
                "confidence": 0.0,
                "agrees_with_human": False,
                "model_used": f"{self.provider}/{self.model}"
            }
    
    def _evaluate_with_ollama(self, prompt: str) -> str:
        """Evaluate using Ollama."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 1500
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]
    
    def _evaluate_with_gemini(self, prompt: str) -> str:
        """Evaluate using Gemini."""
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        response = self.client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1500,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
        )
        # Try to get text from response, handling blocked responses
        try:
            return response.text
        except ValueError as e:
            # Response was blocked or has no text
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content.parts:
                    return candidate.content.parts[0].text
            # If all else fails, return error message
            finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
            raise ValueError(f"Gemini response blocked or empty. Finish reason: {finish_reason}")
    
    def _evaluate_with_groq(self, prompt: str) -> str:
        """Evaluate using Groq."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator of pedagogical interactions. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        return response.choices[0].message.content
    
    def _evaluate_with_huggingface(self, prompt: str) -> str:
        """Evaluate using HuggingFace."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            self.base_url,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "temperature": 0.2,
                    "max_new_tokens": 1500,
                    "return_full_text": False
                }
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return str(result)
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from text that might contain other content."""
        # Try to find JSON block
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        # If that fails, try to parse the whole text
        try:
            return json.loads(text)
        except:
            # Return default structure if parsing fails
            return {
                "mistake_identification": "Unknown",
                "providing_guidance": "Unknown",
                "reasoning": "Failed to parse response",
                "confidence": 0.0,
                "agrees_with_human": False
            }


class FreePedagogicalAssessmentPipeline:
    """Main pipeline for processing pedagogical assessment data using FREE LLMs."""
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        solver_model: str = "llama3.1:8b",
        evaluator_model: str = "llama3.1:8b",
        provider: str = "ollama",
        api_key: Optional[str] = None
    ):
        """Initialize the pipeline with free LLM options."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Initializing FREE pipeline with {provider}...")
        self.solver = FreeMathProblemSolver(model=solver_model, provider=provider, api_key=api_key)
        self.evaluator = FreeMistakeIdentificationEvaluator(model=evaluator_model, provider=provider, api_key=api_key)
    
    def load_dataset(self, filename: str) -> List[Dict]:
        """Load a JSON dataset."""
        filepath = self.data_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_dataset(self, data: List[Dict], filename: str):
        """Save a dataset to JSON."""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {filepath}")
    
    def process_trainset_with_answers(
        self,
        limit: Optional[int] = None,
        rate_limit_delay: float = 0.5
    ) -> List[Dict]:
        """Process the training set and add answer keys."""
        print("Loading training set...")
        trainset = self.load_dataset("trainset.json")
        
        if limit:
            trainset = trainset[:limit]
            print(f"Processing first {limit} samples for testing")
        
        print(f"Processing {len(trainset)} conversations...")
        enriched_data = []
        
        for idx, conversation in enumerate(tqdm(trainset, desc="Solving problems")):
            try:
                problem = self.solver.extract_math_problem(conversation["conversation_history"])
                
                if problem:
                    solution = self.solver.solve_problem(problem)
                    enriched_conversation = conversation.copy()
                    enriched_conversation["math_problem"] = problem
                    enriched_conversation["answer_key"] = solution
                    enriched_data.append(enriched_conversation)
                else:
                    enriched_data.append(conversation)
                
                if rate_limit_delay > 0:
                    time.sleep(rate_limit_delay)
                
            except Exception as e:
                print(f"\nError processing conversation {idx}: {e}")
                enriched_data.append(conversation)
        
        return enriched_data
    
    def evaluate_all_responses(
        self,
        enriched_trainset: List[Dict],
        limit: Optional[int] = None,
        rate_limit_delay: float = 0.5
    ) -> List[Dict]:
        """Evaluate all LLM responses for mistake identification."""
        print("\nEvaluating tutor responses...")
        
        if limit:
            enriched_trainset = enriched_trainset[:limit]
        
        evaluated_data = []
        total_responses = sum(len(conv.get("tutor_responses", {})) for conv in enriched_trainset)
        
        pbar = tqdm(total=total_responses, desc="Evaluating responses")
        
        for conversation in enriched_trainset:
            evaluated_conversation = conversation.copy()
            answer = conversation.get("answer_key", {}).get("answer", "Unknown")
            
            if "tutor_responses" in conversation:
                evaluated_responses = {}
                
                for tutor_name, response_data in conversation["tutor_responses"].items():
                    if not response_data:
                        evaluated_responses[tutor_name] = response_data
                        pbar.update(1)
                        continue
                    
                    try:
                        tutor_response = response_data.get("response", "")
                        current_annotation = response_data.get("annotation", {})
                        
                        evaluation = self.evaluator.evaluate_response(
                            conversation["conversation_history"],
                            answer,
                            tutor_response,
                            current_annotation
                        )
                        
                        evaluated_response = response_data.copy()
                        evaluated_response["evaluation"] = evaluation
                        evaluated_responses[tutor_name] = evaluated_response
                        
                        if rate_limit_delay > 0:
                            time.sleep(rate_limit_delay)
                        
                    except Exception as e:
                        print(f"\nError evaluating {tutor_name}: {e}")
                        evaluated_responses[tutor_name] = response_data
                    
                    pbar.update(1)
                
                evaluated_conversation["tutor_responses"] = evaluated_responses
            
            evaluated_data.append(evaluated_conversation)
        
        pbar.close()
        return evaluated_data
    
    def create_train_val_test_splits(
        self,
        evaluated_data: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split the evaluated data into train, validation, and test sets."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        random.seed(seed)
        shuffled_data = evaluated_data.copy()
        random.shuffle(shuffled_data)
        
        n = len(shuffled_data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_set = shuffled_data[:train_end]
        val_set = shuffled_data[train_end:val_end]
        test_set = shuffled_data[val_end:]
        
        print(f"\nDataset split:")
        print(f"  Training:   {len(train_set)} samples ({train_ratio*100:.1f}%)")
        print(f"  Validation: {len(val_set)} samples ({val_ratio*100:.1f}%)")
        print(f"  Test:       {len(test_set)} samples ({test_ratio*100:.1f}%)")
        
        return train_set, val_set, test_set
    
    def generate_statistics(self, evaluated_data: List[Dict]) -> Dict:
        """Generate statistics about the evaluated dataset."""
        stats = {
            "total_conversations": len(evaluated_data),
            "problems_with_answers": 0,
            "total_responses": 0,
            "evaluations_complete": 0,
            "agreement_with_human": {"yes": 0, "no": 0, "partial": 0},
            "mistake_identification_distribution": {"Yes": 0, "No": 0, "To some extent": 0},
            "providing_guidance_distribution": {"Yes": 0, "No": 0, "To some extent": 0},
        }
        
        for conv in evaluated_data:
            if "answer_key" in conv and conv["answer_key"].get("answer"):
                stats["problems_with_answers"] += 1
            
            if "tutor_responses" in conv:
                for tutor_name, response_data in conv["tutor_responses"].items():
                    if not response_data:
                        continue
                    
                    stats["total_responses"] += 1
                    
                    if "evaluation" in response_data:
                        stats["evaluations_complete"] += 1
                        eval_data = response_data["evaluation"]
                        
                        if eval_data.get("agrees_with_human"):
                            stats["agreement_with_human"]["yes"] += 1
                        else:
                            stats["agreement_with_human"]["no"] += 1
                        
                        mi = eval_data.get("predicted_mistake_identification", "Unknown")
                        if mi in stats["mistake_identification_distribution"]:
                            stats["mistake_identification_distribution"][mi] += 1
                        
                        pg = eval_data.get("predicted_providing_guidance", "Unknown")
                        if pg in stats["providing_guidance_distribution"]:
                            stats["providing_guidance_distribution"][pg] += 1
        
        return stats
    
    def run_full_pipeline(
        self,
        solve_problems: bool = True,
        evaluate_responses: bool = True,
        create_splits: bool = True,
        limit: Optional[int] = None,
        rate_limit_delay: float = 0.5
    ):
        """Run the complete pipeline with FREE LLMs."""
        print("=" * 80)
        print("PEDAGOGICAL ASSESSMENT PIPELINE (FREE LLM VERSION)")
        print("=" * 80)
        
        if solve_problems:
            print("\n" + "=" * 80)
            print("STEP 1: Solving Math Problems with FREE LLMs")
            print("=" * 80)
            enriched_trainset = self.process_trainset_with_answers(
                limit=limit,
                rate_limit_delay=rate_limit_delay
            )
            self.save_dataset(enriched_trainset, "trainset_with_answers.json")
        else:
            print("\nLoading existing trainset with answers...")
            enriched_trainset = self.load_dataset("trainset_with_answers.json")
        
        if evaluate_responses:
            print("\n" + "=" * 80)
            print("STEP 2: Evaluating Tutor Responses with FREE LLMs")
            print("=" * 80)
            evaluated_data = self.evaluate_all_responses(
                enriched_trainset,
                limit=limit,
                rate_limit_delay=rate_limit_delay
            )
            self.save_dataset(evaluated_data, "trainset_fully_evaluated.json")
        else:
            print("\nLoading existing evaluated data...")
            evaluated_data = self.load_dataset("trainset_fully_evaluated.json")
        
        print("\n" + "=" * 80)
        print("STEP 3: Generating Statistics")
        print("=" * 80)
        stats = self.generate_statistics(evaluated_data)
        self.save_dataset([stats], "dataset_statistics.json")
        
        print("\nDataset Statistics:")
        print(f"  Total conversations: {stats['total_conversations']}")
        print(f"  Problems with answers: {stats['problems_with_answers']}")
        print(f"  Total tutor responses: {stats['total_responses']}")
        print(f"  Evaluations complete: {stats['evaluations_complete']}")
        
        if create_splits:
            print("\n" + "=" * 80)
            print("STEP 4: Creating Train/Validation/Test Splits")
            print("=" * 80)
            train_set, val_set, test_set = self.create_train_val_test_splits(evaluated_data)
            
            self.save_dataset(train_set, "train_split.json")
            self.save_dataset(val_set, "validation_split.json")
            self.save_dataset(test_set, "test_split.json")
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"\nOutput directory: {self.output_dir}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pedagogical Assessment Pipeline (FREE LLMs)")
    parser.add_argument("--data-dir", type=str, default="/DATA/cs24resch11011/repos/pedagogical-assessment/data")
    parser.add_argument("--output-dir", type=str, 
                        default="/DATA/cs24resch11011/repos/pedagogical-assessment/devendra/mistake_identification/llm_approach/output_free")
    parser.add_argument("--solver-model", type=str, default=None,
                        help="Model for solving (ollama: llama3.1:8b, gemini: gemini-1.5-flash, groq: llama-3.1-70b-versatile)")
    parser.add_argument("--evaluator-model", type=str, default=None,
                        help="Model for evaluating")
    parser.add_argument("--provider", type=str, default="ollama", 
                        choices=["ollama", "gemini", "groq", "huggingface"],
                        help="FREE LLM provider")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (for gemini, groq, huggingface)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rate-limit-delay", type=float, default=0.5)
    parser.add_argument("--skip-solving", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-splits", action="store_true")
    
    args = parser.parse_args()
    
    # Set default models based on provider if not specified
    default_models = {
        "ollama": "llama3.1:8b",
        "gemini": "gemini-2.5-flash",  # Updated to latest Gemini model
        "groq": "llama-3.1-70b-versatile",
        "huggingface": "meta-llama/Meta-Llama-3-8B-Instruct"
    }
    
    solver_model = args.solver_model or default_models.get(args.provider, "llama3.1:8b")
    evaluator_model = args.evaluator_model or default_models.get(args.provider, "llama3.1:8b")
    
    print(f"\n{'='*80}")
    print(f"Provider: {args.provider}")
    print(f"Solver Model: {solver_model}")
    print(f"Evaluator Model: {evaluator_model}")
    print(f"{'='*80}\n")
    
    pipeline = FreePedagogicalAssessmentPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        solver_model=solver_model,
        evaluator_model=evaluator_model,
        provider=args.provider,
        api_key=args.api_key
    )
    
    pipeline.run_full_pipeline(
        solve_problems=not args.skip_solving,
        evaluate_responses=not args.skip_evaluation,
        create_splits=not args.skip_splits,
        limit=args.limit,
        rate_limit_delay=args.rate_limit_delay
    )


if __name__ == "__main__":
    main()
