"""GSM8K dataset adapter (Grade School Math 8K)."""

from typing import Any, Dict, List, Optional

from ..base import DatasetAdapter


class GSM8KAdapter(DatasetAdapter):
    """Adapter for OpenAI's GSM8K dataset.

    Source: ``openai/gsm8k`` on HuggingFace.
    Domain: grade-school math with step-by-step solutions.
    Size: ~7.5K train / ~1.3K test.
    """

    data_source = "gsm8k"
    domain_type = "verifiable"

    def load_raw(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load GSM8K from HuggingFace datasets."""
        from datasets import load_dataset

        ds_name = path or "openai/gsm8k"
        ds = load_dataset(ds_name, "main", split="train")

        items = []
        for row in ds:
            # GSM8K format: "question" and "answer" fields.
            # The answer contains step-by-step reasoning ending with
            # "#### <final_number>".
            answer_text = row["answer"]
            final_answer = ""
            if "####" in answer_text:
                final_answer = answer_text.split("####")[-1].strip()

            items.append({
                "question": row["question"],
                "solution": answer_text,
                "final_answer": final_answer,
            })

        return items

    def to_verl_format(self, item: Dict[str, Any], tokenizer: Any = None) -> Dict[str, Any]:
        """Convert a GSM8K item to veRL row format.

        The prompt asks the model to generate a rubric for evaluating
        an answer to the math question.
        """
        question = item["question"]
        prompt_messages = self.build_rubric_generation_prompt(
            question=question,
            context="This is a grade-school math problem with a unique numerical answer.",
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
                "question": question,
            },
        }
