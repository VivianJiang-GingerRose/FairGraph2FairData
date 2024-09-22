# FaGGM (Fairness-aware Generative Graphical Models)

## Description

**FaGGM** (Fairness-aware Generative Graphical Model) is a novel approach designed to address biases in machine learning models that arise from historical prejudices and underrepresented data groups. These biases can lead to discrimination, either by perpetuating existing prejudices or by failing to accurately identify patterns for certain groups due to insufficient data.

FaGGM tackles these challenges by incorporating a fairness regularization term into its graph structure learning algorithm. This integration ensures that the learned models are fairer and more equitable across different groups. Additionally, FaGGM serves as a fair data generator, enhancing data representation for underrepresented groups and improving overall model performance.

Key features of FaGGM include:

- **Fairness Integration**: Compatible with most fairness definitions, it seamlessly integrates fairness considerations into the model learning process.
- **Flexible Structure Learning**: Works with various score-based structure learning algorithms, providing flexibility in model development.
- **High-Quality Synthetic Data Generation**: Produces synthetic data that mitigates bias while maintaining data quality.
- **Improved Model Fairness**: Experimental results show significantly increased fairness scores with minimal reductions in accuracy compared to other bias mitigation methods.

By using FaGGM, developers and researchers can build machine learning models that are not only accurate but also fair, promoting equitable decision-making in applications that impact people's lives.


## Repository Structure
- `data/`: Raw and processed data used in the experiments
- `ExistingBiasMitigationMethods/`: Implementations of existing bias mitigation methods: Reweighing, Disparate Impact Remover(DIR), and Learn Fair Representations (LFR).
- `experiments/`: Scripts to run experiments and generate results. It first discretize the data. Then for each dataset, perform fairness-aware structure learning using FairnessAwareHC and existing bias mitigation methods. Finally, evaluate the performance of the models using the generated data.
- `FairnessAwareHC/`: Implementation of fairness-aware Hill Climbing search algorithm. The algorithm is based on the Hill Climbing search algorithm in the pgmpy library. [@pgmpy]


## Project Setup

This project is managed using [Poetry](https://python-poetry.org/), a tool for dependency management and packaging in Python. Follow the instructions below to set up the project.

1. **Install Poetry** (if you haven't already):

Make sure you have Poetry installed. If not, you can install it by following the instructions [here](https://python-poetry.org/docs/#installation).

2. **Clone the repository** 
3. **Install project dependencie** by running the following command in the project root directory:

```bash
poetry install
```
This command will create a virtual environment and install all required dependencies specified in `pyproject.toml`.

4. **Activate the virtual environment** with the following command:

```bash                     
poetry shell
```
Now you're ready to run the project within the virtual environment.

## Usage


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Languages

Python

## Citation

If you use this code or our method in your research, please consider citing our paper:

```bibtex
@article{jiang2024FaGGM,
  title={Charting a Fair Path: FaGGM Fairness-aware Generative Graphical Models},
  author={Vivian Wei Jiang, Gustavo Batista, and Michael Bain},
  year={2024},
}

## References
[@pgmpy]: https://github.com/pgmpy/pgmpy "Kumar, Ankur, and Nayyar, Abinash. 'pgmpy: Probabilistic Graphical Models using Python.'"