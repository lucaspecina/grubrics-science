"""MATH dataset adapter (Hendrycks competition math)."""

from typing import Any, Dict, List, Optional

from ..base import DatasetAdapter


class MATHAdapter(DatasetAdapter):
    """Adapter for Hendrycks' MATH dataset.

    Source: ``hendrycks/competition_math`` on HuggingFace.
    Domain: competition math (AMC 10/12, AIME, etc.), 7 subjects, 5 levels.
    Size: ~12K train / 500 test.
    """

    data_source = "math"
    domain_type = "verifiable"

    def load_raw(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load MATH from HuggingFace datasets."""
        from datasets import load_dataset

        ds_name = path or "hendrycks/competition_math"
        ds = load_dataset(ds_name, split="train")

        items = []
        for row in ds:
            # MATH format: "problem", "solution", "level", "type"
            # Solution contains step-by-step reasoning with \\boxed{answer}.
            solution = row["solution"]
            final_answer = ""
            if "\\boxed{" in solution:
                # Extract content inside \boxed{...}
                start = solution.rfind("\\boxed{") + len("\\boxed{")
                depth = 1
                end = start
                while end < len(solution) and depth > 0:
                    if solution[end] == "{":
                        depth += 1
                    elif solution[end] == "}":
                        depth -= 1
                    end += 1
                final_answer = solution[start:end - 1]

            items.append({
                "question": row["problem"],
                "solution": solution,
                "final_answer": final_answer,
                "level": row.get("level", ""),
                "subject": row.get("type", ""),
            })

        return items

    def to_verl_format(self, item: Dict[str, Any], tokenizer: Any = None) -> Dict[str, Any]:
        """Convert a MATH item to veRL row format."""
        question = item["question"]
        subject = item.get("subject", "mathematics")
        level = item.get("level", "")

        context = f"This is a competition math problem ({subject}"
        if level:
            context += f", {level}"
        context += ") with a precise answer."

        prompt_messages = self.build_rubric_generation_prompt(
            question=question,
            context=context,
        )

        return {
            "data_source": self.data_source,
            "prompt": prompt_messages,
            "reward_model": {
                "ground_truth": item["final_answer"],
                "style": "rubric_for_verifiable",
            },
            "extra_info": {
                "domain_type": self.domain_type,
                "solution": item["solution"],
                "final_answer": item["final_answer"],
                "level": item.get("level", ""),
                "subject": item.get("subject", ""),
                "question": question,
            },
        }
