# Fine-Tuning Report: FLAN-T5 Model for Meeting Summarization

## Project Overview

This project focuses on fine-tuning the `google/flan-t5-small` language model to generate summaries of meeting transcripts. The objective was to enhance the model's ability to produce coherent and concise summaries. We also developed a custom model, `AyeshaFayyaz/flan-T5_summarizer`, to compare performance. The dataset used for this task is `microsoft/MeetingBank-QA-Summary`.

## App Interface

Here are some screenshots of the app interface:

### Main Screen
![Main Screen]<img width="1438" alt="App#1" src="https://github.com/user-attachments/assets/db6aae7a-92a6-432f-93c6-f49e84a83d0a">


### Training Progress
![Evaluation Results#1]<img width="1440" alt="App#2" src="https://github.com/user-attachments/assets/c9e8dd93-bc2b-4b4b-b8dc-5f3a8f35300e">


### Evaluation Results
![Evaluation Results#2]<img width="1440" alt="App#3" src="https://github.com/user-attachments/assets/84d1959f-77d5-4643-ba26-fb5d6115b656">


## Fine-Tuning Process

### 1. Dataset Preparation
- **Dataset**: Loaded using Hugging Face's datasets library.
- **Pre-processing**: Included data cleaning, tokenization, and preparation for training.

### 2. Model Selection
- **Initial Model**: `google/flan-t5-small`.
- **Custom Model**: Fine-tuned `AyeshaFayyaz/flan-T5_summarizer` to evaluate improvements.

### 3. Tokenization
- **Tool**: Utilized `AutoTokenizer` class.
- **Handling Length**: Applied truncation to ensure sequences did not exceed 512 tokens.

### 4. Training Setup
- **Configuration**: Set up learning rate, batch size, and number of epochs.
- **Training**: Managed using `Trainer` class from Hugging Face Transformers.

### 5. Evaluation
- **Metrics**: Performance evaluated using ROUGE-1, ROUGE-2, and ROUGE-L.
- **Results**: Precision, recall, and F1-score of the generated summaries were assessed.

## Challenges Faced

### 1. GPU Limitation
- **Issue**: Problems with MPS (Metal Performance Shaders) on Apple Silicon.
- **Workaround**: Used CPU fallback, which led to slower performance.

### 2. Tokenization Issues
- **Issue**: Sequences exceeding 512 tokens caused indexing errors.
- **Solution**: Implemented truncation to manage longer sequences.

### 3. Performance Constraints
- **Issue**: Training on CPU resulted in longer training times and slower inference.
- **Solution**: Managed computational resources to balance performance and time.

## Key Observations

### 1. Tokenization
- **Insight**: Proper tokenization and handling of sequence length are crucial for model performance.
- **Outcome**: Truncation ensured that sequences remained within acceptable limits.

### 2. Model Performance
- **Insight**: Despite challenges, the model generated coherent and useful summaries.
- **Metrics**: ROUGE scores indicated a good balance of precision and recall.

### 3. Resource Management
- **Insight**: Efficient management of resources is critical for model performance.
- **Future Work**: Optimize for Apple Silicon and explore advanced tokenization techniques.

## Conclusion

The fine-tuning of `google/flan-t5-small` and `AyeshaFayyaz/flan-T5_summarizer` models was successful, achieving satisfactory performance despite GPU limitations and tokenization issues. Future work should focus on optimizing models for better compatibility with Apple Silicon and enhancing tokenization techniques for improved performance.

## Installation & Usage

To replicate or build upon this work, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/your-repo.git
