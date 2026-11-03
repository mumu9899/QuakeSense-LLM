import os
import argparse
import pandas as pd

from model import QuakeSenseModel


def is_csv_file(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(".csv")


def run_single_prediction(args):
    model = QuakeSenseModel(
        model_dir=args.model_dir,
        base_model=args.base_model,
        max_length=args.max_length,
        device=args.device,
        num_labels=args.num_labels
    )

    result = model.predict_one(args.input)

    print(f"Input text: {result['text']}")
    print(f"Predicted label: {result['pred_label']}")
    print(f"Predicted category: {result['pred_name']}")
    print(f"Confidence: {result['confidence']:.4f}")

    prob_items = [
        f"prob_{i}: {result[f'prob_{i}']:.4f}"
        for i in range(args.num_labels)
        if f"prob_{i}" in result
    ]

    print("Class probabilities:")
    print(", ".join(prob_items))


def run_batch_prediction(args):
    df = pd.read_csv(args.input)
    if args.text_column not in df.columns:
        raise ValueError(
            f"Column '{args.text_column}' was not found in input CSV. Available columns: {list(df.columns)}"
        )

    texts = df[args.text_column].astype(str).tolist()

    model = QuakeSenseModel(
        model_dir=args.model_dir,
        base_model=args.base_model,
        max_length=args.max_length,
        device=args.device,
        num_labels=args.num_labels
    )

    results = model.predict_batch(texts, batch_size=args.batch_size)
    result_df = pd.DataFrame(results)
    output_df = pd.concat(
        [
            df.reset_index(drop=True),
            result_df.drop(columns=["text"]).reset_index(drop=True)
        ],
        axis=1
    )
    output_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Batch prediction completed: {len(output_df)} samples")
    print(f"Results saved to: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Post-earthquake social media intensity classification.")

    parser.add_argument("--input", type=str, required=True, help="Input text or path to a CSV file.")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV file path for batch prediction.")
    parser.add_argument("--model_dir", type=str, default="pretrained", help="Path to the fine-tuned model or LoRA adapter directory.")
    parser.add_argument("--base_model", type=str, default=None, help="Base model path or Hugging Face model name. Required when model_dir contains a LoRA adapter.")
    parser.add_argument("--text_column", type=str, default="text", help="Text column name in the input CSV file.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for batch prediction.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--num_labels", type=int, default=4, help="Number of classification labels.")
    parser.add_argument("--device", type=str, default=None, help="Device for inference. Default: auto (cuda if available, otherwise cpu).")
    return parser.parse_args()


def main():
    args = parse_args()
    if is_csv_file(args.input):
        run_batch_prediction(args)
    else:
        run_single_prediction(args)


if __name__ == "__main__":
    main()