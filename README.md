# Trust-Aware LLM-Based Conversational Recommendation for Personalized Product Selection in E-Commerce Platforms

## Overview

This project implements a **Trust-Aware LLM-Based Conversational Recommendation Framework** for reliable and personalized product recommendation in e-commerce.

The core proposed model is the **Preference-Aware Trust Transformer (PATT)**, which jointly learns:

- User preferences
- Product semantic information
- User-product interaction patterns
- Review-derived trust information
- Preference-guided attention

The trained PATT model produces trust-aware recommendation scores for **Top-K product ranking**. A conversational layer then uses the ranked products to provide product comparison, recommendation explanations, and purchase-decision support.

---

## Dataset

### Data Source

The implementation uses the **Amazon Reviews 2023** dataset released by the **McAuley Lab, UC San Diego**.

- **Official Dataset Website:** https://amazon-reviews-2023.github.io/
- **Official Hugging Face Repository:** https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- **Dataset Used in This Implementation:** Amazon Beauty category/subset

The implementation uses an **Amazon Reviews 2023 – Beauty sample** containing:

### Reviews
`amazon_beauty_100k_reviews.parquet`

Important fields:

- `user_id`
- `parent_asin`
- `rating`
- `title`
- `text`
- `timestamp`
- `helpful_vote`
- `verified_purchase`

### Product Metadata
`amazon_beauty_product_metadata.parquet`

Important fields:

- `parent_asin`
- `title`
- `main_category`
- `features`
- `description`
- `price`
- `store`
- `categories`
- `details`
- `average_rating`
- `rating_number`

Reviews and product metadata are integrated using `parent_asin`.

---

## Proposed Workflow

```text
Amazon Reviews + Product Metadata
                |
                v
Data Cleaning and Integration
                |
                v
+---------------------------------------------+
| Feature Learning                            |
|                                             |
| User Preference Modelling                   |
| Product Semantic Representation             |
| Dynamic Trust Estimation                    |
| Interaction Representation                  |
+---------------------------------------------+
                |
                v
+---------------------------------------------+
| PATT                                        |
| Preference-Aware Trust Transformer          |
|                                             |
| 1. Preference Alignment                     |
| 2. Trust-Aware Feature Learning             |
| 3. Preference-Guided Multi-Head Attention   |
+---------------------------------------------+
                |
                v
Trust-Aware Prediction
                |
                v
Recommendation Score
                |
                v
Top-K Product Ranking
                |
                v
Conversational Recommendation / Explanation
                |
                v
User Feedback and Adaptive Refinement
```

---

## PATT Architecture

### 1. User Preference Modelling
Learns preference information from user interaction history, ratings, verification behaviour, and historical positive-interaction patterns.

### 2. Product Semantic Representation
Represents product information using product title, features, description, and related textual metadata.

### 3. Dynamic Trust Estimation
Trust information is derived from available signals including:

- Review semantics
- Verified-purchase status
- Helpful votes
- Rating behaviour
- Temporal information

### 4. Preference Alignment
Measures compatibility between user preference representations and candidate product representations.

### 5. Trust-Aware Feature Learning
Allows highly reliable review/product information to contribute more strongly to the learned representation.

### 6. Preference-Guided Multi-Head Attention
Jointly attends to preference, semantic, interaction, and trust features.

### 7. Trust-Aware Prediction
Produces the final relevance probability used by the recommender.

---

## Evaluation

Binary relevance is defined as:

```text
Relevant     : rating >= 4
Not Relevant : rating < 4
```

The notebook includes:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Specificity
- Classification report
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Training and validation curves
- Attention-weight visualization
- Preference-alignment visualization
- Trust-score analysis
- Top-K recommendation visualization

### Executed Test Results

| Metric | Score |
|---|---:|
| Accuracy | 92.35% |
| Precision | 94.86% |
| Recall | 94.76% |
| F1 Score | 94.81% |
| ROC-AUC | 96.89% |
| Specificity | 85.59% |

These values are generated from the executed notebook and are not hard-coded evaluation results.

---

## Top-K Recommendation

The recommendation stage ranks candidate products using the trained PATT model.

The ranking stage avoids using the target user's future review text or rating as candidate input. Candidate information is generated from product metadata and training-derived product/review aggregates.

Example output:

```text
User Query
    |
    v
Preference / Query Representation
    |
    v
PATT Recommendation Scoring
    |
    v
Top-K Products
    |
    v
Personalized Explanation
```

---

## Conversational Layer

The notebook contains an **offline conversational explanation fallback** so the complete notebook can run without requiring a paid API key.

PATT remains the actual recommendation and ranking engine.

For deployment, the structured PATT Top-K output can be connected to a GPT/LLM API for:

- Natural-language preference extraction
- Product comparison
- Personalized explanations
- Purchase-decision support
- Conversational feedback collection

---

## Important Implementation Note

The notebook uses a **Logistic Regression semantic-prior component** as an auxiliary semantic relevance signal before PATT refinement.

It is **not used or reported as a baseline comparison model**.

The final evaluated proposed recommendation model is **PATT**.

For a fully deep-learning-only research version, this auxiliary semantic prior can later be replaced with a trainable neural semantic encoder.

---

## Project Files

Recommended structure:

```text
project/
|
|-- Trust_Aware_LLM_PATT_Full_Executed_Clear_Visuals.ipynb
|-- README.md
|-- requirements.txt
|
|-- Amazon_Reviews_2023-20260811T053649Z-1-001.zip
|
`-- patt_amazon_beauty_work/
    `-- artifacts/
        |-- patt_model.pt
        |-- test_metrics.csv
        |-- attention_weights.csv
        |-- demo_top10_recommendations.csv
        `-- run_summary.json
```

---

## Installation

### Google Colab

Upload the dataset ZIP and notebook, then run:

```python
!pip install -r requirements.txt
```

Most packages are already available in standard Google Colab environments.

### Local Environment

Create a virtual environment:

```bash
python -m venv venv
```

Activate it and install dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook
```

---

## Running the Notebook

1. Place the Amazon Reviews ZIP in the working directory or upload it to Colab.
2. Open `Trust_Aware_LLM_PATT_Full_Executed_Clear_Visuals.ipynb`.
3. Run the notebook from top to bottom.
4. The notebook automatically:
   - locates the supplied ZIP,
   - extracts the Parquet files,
   - preprocesses and integrates the data,
   - performs EDA,
   - estimates trust,
   - creates semantic/user/interaction features,
   - trains PATT,
   - evaluates the model,
   - generates visualizations,
   - produces Top-K recommendations,
   - saves model and result artifacts.

---

## Saved Artifacts

After execution, the notebook saves:

- `patt_model.pt` — trained PATT model
- `test_metrics.csv` — final test metrics
- `attention_weights.csv` — learned attention information
- `demo_top10_recommendations.csv` — demonstration recommendations
- `run_summary.json` — dataset and evaluation summary

---

## Reproducibility

The implementation uses:

```text
Random seed = 42
```

Random seeds are set for NumPy and PyTorch to improve reproducibility.

---

## Hardware

The implementation supports:

- CPU execution
- CUDA-enabled GPU execution
- Google Colab

A GPU is recommended for faster PATT training, but the notebook can run on CPU.

---

## Research Scope

The implementation focuses on the proposed **Trust-Aware PATT framework** and does not perform baseline-model comparison.

The major contribution is the joint integration of:

```text
User Preference
      +
Product Semantics
      +
Interaction Information
      +
Dynamic Review Trust
      +
Preference-Guided Attention
      |
      v
Reliable Personalized Recommendation
```
