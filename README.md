# 🏷️ POS Tagging Model Collections

This repository contains machine learning models of POS Tagging, designed to be deployed using ONNX and utilized in a Streamlit-based web application. The app provides an interactive interface for performing this task using neural network architectures. [Check here to see other ML tasks](https://github.com/verneylmavt/ml-model).

For more information about the training process, please check the `pos-tagging.ipynb` file in the `training` folder.

## 🎈 Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://verneylogyt-pos-tagging.streamlit.app/)

![Demo GIF](https://github.com/verneylmavt/st-pos-tagging/blob/main/assets/demo.gif)

If you encounter message `This app has gone to sleep due to inactivity`, click `Yes, get this app back up!` button to wake the app back up.

<!-- [https://verneylogyt.streamlit.app/](https://verneylogyt.streamlit.app/) -->

## ⚙️ Running Locally

If the demo page is not working, you can fork or clone this repository and run the application locally by following these steps:

<!-- ### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- pip (Python Package Installer)

### Installation Steps -->

1. Clone the repository:

   ```bash
   git clone https://github.com/verneylmavt/st-pos-tagging.git
   cd st-snt-analysis
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

Alternatively you can run `jupyter notebook demo.ipynb` for a minimal interface to quickly test the model (implemented w/ `ipywidgets`).

## ⚖️ Acknowledgement

I acknowledge the use of the **Penn Treebank (PTB)** dataset provided by the **Linguistic Data Consortium (LDC)**. This dataset has been instrumental in conducting the research and developing this project.

- **Dataset Name**: Penn Treebank (PTB)
- **Source**: [https://catalog.ldc.upenn.edu/LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42)
- **License**: The dataset is available under the LDC User Agreement for Non-Members. Please refer to the [LDC website](https://www.ldc.upenn.edu/) for licensing details.
- **Description**: This dataset contains over one million words of text from the 1989 Wall Street Journal, annotated for part-of-speech (POS) information and skeletal syntactic structure. It is widely used for training and evaluating natural language processing models.

I deeply appreciate the efforts of Linguistic Data Consortium in making this dataset available.
