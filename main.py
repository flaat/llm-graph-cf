import argparse
from src.llms import get_eplanations

def main():
    parser = argparse.ArgumentParser(description="Process command-line parameters.")

    parser.add_argument(
        '--parameters',
        type=str,
        default='0.5',
        help='Parameters value (default: 0.5)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Temperature value (default: 0.7)'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help='Top-p value (default: 0.8)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='cora',
        help='Dataset name (default: Cora)'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=2048,
        help='Maximum number of tokens (default: 2048)'
    )
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.05,
        help='Repetition penalty value (default: 1.05)'
    )
    
    parser.add_argument(
        '--explainer',
        type=str,
        default="cf-gnnfeatures",
        help='Explanation'
    )

    args = parser.parse_args()

    # Use the parsed arguments as needed
    print(f"Parameters: {args.parameters}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Dataset: {args.dataset}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    
    
    get_eplanations(parameters=args.parameters,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    dataset=args.dataset,
                    max_tokens=args.max_tokens,
                    repetition_penalty=args.repetition_penalty,
                    explainer=args.explainer)

if __name__ == "__main__":
    main()
