from dataset.main import main as generate_datasets
from performance.main import main as evaluate_performance
from eval.main import main as evaluate_quality


def main():
    generate_datasets()
    evaluate_performance()
    evaluate_quality()


if __name__ == "__main__":
    main()
