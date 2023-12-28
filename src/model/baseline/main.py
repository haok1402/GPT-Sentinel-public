import json
import torch
import argparse
import subprocess
from pathlib import Path
from .dataset import DualDataset
from .utilities import test, save_confusion_matrix, save_pca_analysis
from transformers import RobertaTokenizer, RobertaForSequenceClassification


def test_gpt2(args: argparse.Namespace) -> None:
    pass


def test_gpt3(args: argparse.Namespace) -> None:
    # initialize configuration
    directory = Path("src/model/baseline")
    with open(Path(directory, "config.json")) as fp: 
        testConfig = json.load(fp)["gpt-3.5-turbo"]
    cached_path = Path(directory, "test_gpt3.pt")
    
    if args.R or not cached_path.is_file():
        # download pretrained weight if not found
        if args.V: print("INIT: download detector-base.pt")
        if not Path(directory, "detector-base.pt").is_file():
            subprocess.run([
                "wget", "https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-base.pt", 
                "-O", Path(directory, "detector-base.pt")
            ])

        # initialize dataset, detector
        if args.V: print("INIT: dataset, detector")
        dataset = DualDataset(**testConfig["dataset"])
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        detector_model = RobertaForSequenceClassification.from_pretrained("roberta-base", output_hidden_states=True)

        # load pretrained weight
        if args.V: print("INIT: load pretrained weight")
        detector_state = torch.load(Path(directory, "detector-base.pt"))
        detector_model.load_state_dict(detector_state["model_state_dict"])
        detector_model.to("cuda")

        # run the testing
        if args.V: print("INIT: run testing")
        cached_data = test(dataset, tokenizer, detector_model)
        torch.save(cached_data, cached_path)
    else: 
        if args.V: print("INIT: found cache and load")
        cached_data = torch.load(cached_path)

    # visualize confusion matrix
    if args.V: print("INIT: save_confusion_matrix")
    save_confusion_matrix(
        **cached_data,
        filename="confusion_matrix.png",
        title="Confusion Matrix for GPT2 Detector on GPT3.5 Rephrased Text",
        cached_path="cache/figures/baseline/test_gpt3", 
    )

    # visualize pca projection
    if args.V: print("INIT: save_pca_analysis")
    save_pca_analysis(
        **cached_data,
        filename="pca_projection.png",
        title="PCA Projection for GPT2 Detector on GPT3.5 Rephrased Text",
        cached_path="cache/figures/baseline/test_gpt3", 
    )


def init_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run baseline model and perform different analysis."
    )
    parser.add_argument(
        "-M", type=str, required=True, choices=["GPT-2", "GPT-3"],
        help="name of the model whose output tested against baseline",
    )
    parser.add_argument(
        "-R", action="store_true",
        help="ignore cache and re-run the baseline model"
    )
    parser.add_argument(
        "-V", action="store_true",
        help="verbosely display progress during analysis"
    )
    return parser.parse_args()


def main():
    args = init_args()
    if args.M == "GPT-2":
        test_gpt2(args)
    if args.M == "GPT-3":
        test_gpt3(args)


if __name__ == "__main__":
    main()
