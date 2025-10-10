"""
Pedagogical Assessment Pipeline
================================
This script:
1. Reads the training dataset
2. Extracts math problems from conversations
3. Uses an LLM to solve the math problems and generate answer keys
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

# Try to import OpenAI (can be replaced with other LLM APIs)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not available. Install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: Anthropic library not available. Install with: pip install anthropic")


class MathProblemSolver:
    """Solves math problems using an LLM."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", provider: str = "openai"):
        """
        Initialize the math problem solver.
        
        Args:
            api_key: API key for the LLM provider
            model: Model name to use (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
            provider: "openai" or "anthropic"
        """
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not available")
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic library not available")
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def extract_math_problem(self, conversation_history: str) -> Optional[str]:
        """Extract the math problem from the conversation history."""
        # Look for pattern: "The question is: <problem>"
        pattern = r"The question is:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, conversation_history, re.DOTALL)
        
        if match:
            # Extract until we hit "Student:" or end
            problem = match.group(1)
            student_idx = problem.find("Student:")
            if student_idx != -1:
                problem = problem[:student_idx]
            return problem.strip()
        
        return None
    
    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """
        Solve a math problem using the LLM.
        
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
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert mathematics tutor who solves problems step by step with perfect accuracy."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=2000
                )
                solution_text = response.choices[0].message.content
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.1,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                solution_text = response.content[0].text
            
            # Parse the solution
            answer = self._extract_answer(solution_text)
            
            return {
                "answer": answer,
                "solution_steps": solution_text,
                "explanation": self._extract_explanation(solution_text),
                "model_used": self.model
            }
        
        except Exception as e:
            print(f"Error solving problem: {e}")
            return {
                "answer": None,
                "solution_steps": None,
                "explanation": f"Error: {str(e)}",
                "model_used": self.model
            }
    
    def _extract_answer(self, solution_text: str) -> Optional[str]:
        """Extract the final numerical answer from the solution."""
        # Look for "Final Answer:" or similar patterns
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
        return solution_text[:200]  # Return first 200 chars as fallback


class MistakeIdentificationEvaluator:
    """Evaluates LLM responses for mistake identification quality."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", provider: str = "openai"):
        """Initialize the evaluator."""
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not available")
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic library not available")
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    
    def evaluate_response(
        self,
        conversation_history: str,
        correct_answer: str,
        tutor_response: str,
        current_annotation: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Evaluate whether a tutor response correctly identifies mistakes.
        
        Returns:
            Dictionary with evaluation results including:
            - predicted_mistake_identification: "Yes", "No", or "To some extent"
            - predicted_providing_guidance: "Yes", "No", or "To some extent"
            - reasoning: Explanation of the evaluation
            - confidence: Confidence score (0-1)
        """
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

Provide your response in the following JSON format:
{{
    "mistake_identification": "Yes/No/To some extent",
    "providing_guidance": "Yes/No/To some extent",
    "reasoning": "Detailed explanation of your evaluation",
    "confidence": 0.85,
    "agrees_with_human": true/false
}}"""

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator of pedagogical interactions. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1500,
                    response_format={"type": "json_object"}
                )
                result_text = response.choices[0].message.content
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    temperature=0.2,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result_text = response.content[0].text
            
            # Parse JSON response
            result = json.loads(result_text)
            
            return {
                "predicted_mistake_identification": result.get("mistake_identification", "Unknown"),
                "predicted_providing_guidance": result.get("providing_guidance", "Unknown"),
                "reasoning": result.get("reasoning", ""),
                "confidence": result.get("confidence", 0.0),
                "agrees_with_human": result.get("agrees_with_human", False),
                "model_used": self.model
            }
        
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return {
                "predicted_mistake_identification": "Error",
                "predicted_providing_guidance": "Error",
                "reasoning": f"Error: {str(e)}",
                "confidence": 0.0,
                "agrees_with_human": False,
                "model_used": self.model
            }


class PedagogicalAssessmentPipeline:
    """Main pipeline for processing pedagogical assessment data."""
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        solver_model: str = "gpt-4o",
        evaluator_model: str = "gpt-4o",
        provider: str = "openai"
    ):
        """Initialize the pipeline."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.solver = MathProblemSolver(model=solver_model, provider=provider)
        self.evaluator = MistakeIdentificationEvaluator(model=evaluator_model, provider=provider)
    
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
        rate_limit_delay: float = 1.0
    ) -> List[Dict]:
        """
        Process the training set and add answer keys.
        
        Args:
            limit: Maximum number of samples to process (for testing)
            rate_limit_delay: Delay between API calls in seconds
        """
        print("Loading training set...")
        trainset = self.load_dataset("trainset.json")
        
        if limit:
            trainset = trainset[:limit]
            print(f"Processing first {limit} samples for testing")
        
        print(f"Processing {len(trainset)} conversations...")
        enriched_data = []
        
        for idx, conversation in enumerate(tqdm(trainset, desc="Solving problems")):
            try:
                # Extract problem
                problem = self.solver.extract_math_problem(conversation["conversation_history"])
                
                if problem:
                    # Solve the problem
                    solution = self.solver.solve_problem(problem)
                    
                    # Add to conversation
                    enriched_conversation = conversation.copy()
                    enriched_conversation["math_problem"] = problem
                    enriched_conversation["answer_key"] = solution
                    
                    enriched_data.append(enriched_conversation)
                else:
                    # No math problem found, keep original
                    enriched_data.append(conversation)
                
                # Rate limiting
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
        rate_limit_delay: float = 1.0
    ) -> List[Dict]:
        """
        Evaluate all LLM responses for mistake identification.
        
        Args:
            enriched_trainset: Dataset with answer keys
            limit: Maximum number to evaluate (for testing)
            rate_limit_delay: Delay between API calls
        """
        print("\nEvaluating tutor responses...")
        
        if limit:
            enriched_trainset = enriched_trainset[:limit]
        
        evaluated_data = []
        total_responses = sum(len(conv.get("tutor_responses", {})) for conv in enriched_trainset)
        
        pbar = tqdm(total=total_responses, desc="Evaluating responses")
        
        for conversation in enriched_trainset:
            evaluated_conversation = conversation.copy()
            answer = conversation.get("answer_key", {}).get("answer", "Unknown")
            
            # Evaluate each tutor response
            if "tutor_responses" in conversation:
                evaluated_responses = {}
                
                for tutor_name, response_data in conversation["tutor_responses"].items():
                    if not response_data:  # Skip empty responses
                        evaluated_responses[tutor_name] = response_data
                        pbar.update(1)
                        continue
                    
                    try:
                        tutor_response = response_data.get("response", "")
                        current_annotation = response_data.get("annotation", {})
                        
                        # Evaluate
                        evaluation = self.evaluator.evaluate_response(
                            conversation["conversation_history"],
                            answer,
                            tutor_response,
                            current_annotation
                        )
                        
                        # Add evaluation to response data
                        evaluated_response = response_data.copy()
                        evaluated_response["evaluation"] = evaluation
                        evaluated_responses[tutor_name] = evaluated_response
                        
                        # Rate limiting
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
        """
        Split the evaluated data into train, validation, and test sets.
        
        Args:
            evaluated_data: Fully processed dataset
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        # Shuffle data
        random.seed(seed)
        shuffled_data = evaluated_data.copy()
        random.shuffle(shuffled_data)
        
        # Calculate split indices
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
                        
                        # Track agreement
                        if eval_data.get("agrees_with_human"):
                            stats["agreement_with_human"]["yes"] += 1
                        else:
                            stats["agreement_with_human"]["no"] += 1
                        
                        # Track distributions
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
        rate_limit_delay: float = 1.0
    ):
        """
        Run the complete pipeline.
        
        Args:
            solve_problems: Whether to solve math problems
            evaluate_responses: Whether to evaluate tutor responses
            create_splits: Whether to create train/val/test splits
            limit: Maximum samples to process (for testing)
            rate_limit_delay: Delay between API calls
        """
        print("=" * 80)
        print("PEDAGOGICAL ASSESSMENT PIPELINE")
        print("=" * 80)
        
        # Step 1: Add answer keys
        if solve_problems:
            print("\n" + "=" * 80)
            print("STEP 1: Solving Math Problems and Adding Answer Keys")
            print("=" * 80)
            enriched_trainset = self.process_trainset_with_answers(
                limit=limit,
                rate_limit_delay=rate_limit_delay
            )
            self.save_dataset(enriched_trainset, "trainset_with_answers.json")
        else:
            print("\nLoading existing trainset with answers...")
            enriched_trainset = self.load_dataset("trainset_with_answers.json")
        
        # Step 2: Evaluate responses
        if evaluate_responses:
            print("\n" + "=" * 80)
            print("STEP 2: Evaluating Tutor Responses for Mistake Identification")
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
        
        # Step 3: Generate statistics
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
        print(f"\nAgreement with human annotations:")
        print(f"  Agree: {stats['agreement_with_human']['yes']}")
        print(f"  Disagree: {stats['agreement_with_human']['no']}")
        
        # Step 4: Create splits
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
    
    parser = argparse.ArgumentParser(description="Pedagogical Assessment Pipeline")
    parser.add_argument("--data-dir", type=str, default="/DATA/cs24resch11011/repos/pedagogical-assessment/data",
                        help="Directory containing input data")
    parser.add_argument("--output-dir", type=str, 
                        default="/DATA/cs24resch11011/repos/pedagogical-assessment/devendra/mistake_identification/llm_approach/output",
                        help="Directory for output files")
    parser.add_argument("--solver-model", type=str, default="gpt-4o",
                        help="Model for solving math problems")
    parser.add_argument("--evaluator-model", type=str, default="gpt-4o",
                        help="Model for evaluating responses")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic"],
                        help="LLM provider to use")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--rate-limit-delay", type=float, default=1.0,
                        help="Delay between API calls (seconds)")
    parser.add_argument("--skip-solving", action="store_true",
                        help="Skip solving problems (use existing answers)")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Skip evaluation (use existing evaluations)")
    parser.add_argument("--skip-splits", action="store_true",
                        help="Skip creating train/val/test splits")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PedagogicalAssessmentPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        solver_model=args.solver_model,
        evaluator_model=args.evaluator_model,
        provider=args.provider
    )
    
    # Run pipeline
    pipeline.run_full_pipeline(
        solve_problems=not args.skip_solving,
        evaluate_responses=not args.skip_evaluation,
        create_splits=not args.skip_splits,
        limit=args.limit,
        rate_limit_delay=args.rate_limit_delay
    )


if __name__ == "__main__":
    main()
